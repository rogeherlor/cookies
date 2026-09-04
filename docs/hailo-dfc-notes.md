# Hailo Dataflow Compiler User Guide

1. ONNX model
2. Parsing to HAILO .har format
3. Optimization
    3.1 Full precision optimization: Equalization / TSE / Pruning
    3.2 Model script modifications
    3.3 Quantization: IBC [Finkelstein2019], AdaRound [Nagel2020], FineTune and QFT [McKinstry2019]
4. Evaluation

## Parser
hailo visualizer *.tar
hailo parser -compare

## Optimization
1. SDK_NATIVE emulator before any changes
2. optimize_full_precision() to apply the model script and the full precision optimizations
3. SDK_FP_OPTIMIZED emulator. Perform full precision validation taking into account the model modifications (normalization, resize, color conversions, etc.)
4. optimization API. GPU machine and a dataset with at least 1024 entries for calibration
5. SDK_QUANTIZED emulator. Optimization and Compression levels 0-5 (important when each level)

* Before calling the optimize() API, you might call load_model_script() to load a model script (.alls file) that includes commands that modify the model, affect the basic quantization flow and additional algorithms to improve the accuracy and optimize the running time.

Debugging Accuracy
    hailo analyze-noise har_path –data-path data_path
    Configure 16-bit output: quantization_param(output_layer1, precision_mode=a16_w16)
    activation clipping
    Use different loss type in Finetune
    QAT
    ...

## Compilation

## Inference

## Layer Noise Analysis
Accuracy analysis: This step is the heart of the tool, and computes the quantization noise of each layer output.
For each layer, the layer under analysis is the only quantized layer, while the rest of the model is kept in full precision.
This highlights the quantization sensitivity of the model to the noise of that specific layer.

## QAT
Different from fine-tunning


# Big models
Multi-context
Each context consists of five phases:
    Config time
    Load time
    Inference time
    Drainage time
    Overhead time


# Hailo RT User Guide

Obtain stats, info from layers, info from quantization and quantized layers, graph visualization, etc.
hailortcli commands

Performance power mode (default) / Ultra-performance power mode
Hailo ONNX Runtime is a fork of ONNX Runtime modified to work on Hailo devices.

hailo_status hailo_create_ethernet_device()

# Install
Download: hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl
Install: sudo apt-get update; sudo apt-get install -y graphviz libgraphviz-dev python3-tk
pip install hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl