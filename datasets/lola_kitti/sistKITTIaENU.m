% Conversion sistema de coordenadas de KITTI a sistema ENU

function [enu]=sistKITTIaENU(KITTI,rpy)

    % KITTI:datos en coordenadas flu
    % rpy:datos de 'roll','pitch','yaw'

    enu=zeros(size(KITTI,1),3); %vector para guardar datos en sistema ENU

    for i=1:size(KITTI,1) % se multiplica cada dato por su respectiva matriz de rotacion (R)

        yaw=rpy(i,3); % angulo 'yaw', guardado en la tercera columna de 'rpy'
        Ryaw=[cos(yaw) -sin(yaw) 0;sin(yaw) cos(yaw) 0;0 0 1];
        R=Ryaw; 
        enu(i,:)=(R*KITTI(i,:)')'; % conversion

    end

end