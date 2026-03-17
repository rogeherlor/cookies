"""
ResNet1D for TLIO inertial odometry — faithful copy of external/tlio/src/network/model_resnet.py.

Credit / License
----------------
Original implementation by Wenxin Liu et al.
Repository: https://github.com/CathIAS/TLIO (MIT License)
Paper: Liu et al., "TLIO: Tight Learned Inertial Odometry", IEEE RA-L 2020.

This file is reproduced here verbatim so that tlio_runner.py can load weights
without triggering external/tlio/src/network/__init__.py's transitive imports,
which conflict with other packages on sys.path.
"""

import torch.nn as nn


def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes * self.expansion)
        self.bn2 = nn.BatchNorm1d(planes * self.expansion)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class FcBlock(nn.Module):
    def __init__(self, in_channel, out_channel, in_dim):
        super().__init__()
        self.prep_channel = 128
        self.fc_dim = 512
        self.prep1 = nn.Conv1d(in_channel, self.prep_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.prep_channel)
        self.fc1 = nn.Linear(self.prep_channel * in_dim, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.fc3 = nn.Linear(self.fc_dim, out_channel)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.bn1(self.prep1(x))
        x = self.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.relu(self.fc2(self.dropout(x)))
        return self.fc3(self.dropout(x))


class ResNet1D(nn.Module):
    """
    Parameters
    ----------
    block_type  : BasicBlock1D
    in_dim      : input channels (6 for IMU)
    out_dim     : output dimension (3)
    group_sizes : list of 4 ints, blocks per residual group  (e.g. [2,2,2,2])
    inter_dim   : temporal size after backbone = W // 32 + 1
    """

    def __init__(self, block_type, in_dim, out_dim, group_sizes, inter_dim,
                 zero_init_residual=False):
        super().__init__()
        self.base_plane = 64
        self.inplanes = self.base_plane

        self.input_block = nn.Sequential(
            nn.Conv1d(in_dim, self.base_plane, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.base_plane),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.residual_groups = nn.Sequential(
            self._make_group(block_type, 64,  group_sizes[0], stride=1),
            self._make_group(block_type, 128, group_sizes[1], stride=2),
            self._make_group(block_type, 256, group_sizes[2], stride=2),
            self._make_group(block_type, 512, group_sizes[3], stride=2),
        )

        self.output_block1 = FcBlock(512 * block_type.expansion, out_dim, inter_dim)
        self.output_block2 = FcBlock(512 * block_type.expansion, out_dim, inter_dim)

        self._initialize(zero_init_residual)

    def _make_group(self, block, planes, group_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, group_size):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)
        return self.output_block1(x), self.output_block2(x)
