%Funcion a la que le pasas los parametros y calcula el error RMS del filtro
function [pRMS1,pRMS2,vRMS]=Copy_of_RMS_KalmanErrores(settings,t1,d,accel_flu,gyro_flu,vel_enu,lla,rpy)
    %Inicializar objeto insfilterMARG
    n_settings=size(settings,2);
    pRMS1=zeros(n_settings,1);
    pRMS2=zeros(n_settings,1);
    vRMS=zeros(n_settings,1);

    for ii=1:n_settings
        % t1: instante inicial de perdida de GPS en segundos
        % d: duracion del tramo de perdida de GPS en segundos
        % Si no se quiere simular un tramo de perdida de GPS: t1=0, d=0
        
        %%%%%%%%%%%%% DATOS Y VARIABLES PARA IMPLEMENTAR EL SISTEMA %%%%%%%%%%%
    
        % DATOS VARIOS
    
        NN = size(lla,1); % numero de datos
        frecIMU = 10; % 10 Hz
        frecGPS = 1; % 1 Hz 
        t2 = t1+d; % instante de final de perdida de GPS en segundos
        A = t1*frecIMU; % nº de dato de inicio de perdida de GPS
        if(d==0)
            B = 0;
        else
            B = t2*frecIMU; % nº de dato de fin de perdida de GPS
        end
        Ts= 1/frecIMU; % tiempo de muestreo
        g = [0;0;9.81]; % aceleracion de la gravedad
        lla0 = [49.01 8.43 116.43]; % origen de coordenada para el sistema ENU,
                                    % se escoge una coordenada cercana a las
                                    % trayectorias de KITTI
        sigma = 7.2921151467e-5; % rotacion de la Tierra [rad/s]
        a = 6378e3; % radio ecuatorial [m]
        b = 6357e3; % radio polar [m]
        ecc = sqrt(1-b^2/a^2); % eccentricidad
    
        % VARIABLES PARA GUARDAR LOS ESTADOS DE NAVEGACION ESTIMADOS
    
        p = zeros(NN,3); % para guardar las posiciones estimadas 
        v = zeros(NN,3); % para guardar las velocidades estimadas 
        r = zeros(NN,3); % para guardar las orientaciones estimadas
        
        % Como primer estado de navegacion se guardan los datos de KITTI
        % Esta no es la inicialización del estado de navegación; estas
        % variables son exclusivamente para guardar resultados
        p(1,:) = lla2enu(lla(1,:),lla0,'ellipsoid');
        v(1,:) = vel_enu(1,:);
        r(1,:) = rpy(1,:);
    
        % INICIALIZACION DEL ESTADO DE NAVEGACION
    
        %Solo se inicializan la posicion y la velocidad con valores conocidos
    
        pIMU = lla2enu(lla(1,:),lla0,'ellipsoid')'; % posicion inicial conocida
        vIMU = vel_enu(1,:)'; % velocidad inicial conocida
        ypr=zeros(3,1); % orientacion inicial desconocida, ypr = (yaw, pitch roll)
       
    
        % INICIALIZACION DE COMPONENTES DEL FILTRO (se usan los valores de
        % settings)
    
        % Estado
    
        x = zeros(15,1); % estado del filtro de Kalman
                         % x = (errorPos, errorVel, errorOrient, accBias, gyrBias)
        accBias = settings(ii).accBias;
        gyrBias = settings(ii).gyrBias;
        x(10:12) = accBias;
        x(13:15) = gyrBias;
        estado = zeros(NN,15); % para guardar los vectores de estado del filtro
                               % de Kalman generados durante toda la trayectoria 
        estado(1,:) = x';
        
        % Coeficientes de Gauss-Markov
    
        beta_acc = settings(ii).beta_acc;
        beta_gyr = settings(ii).beta_gyr;
    
        % Matriz de ruido de medicion (R)
    
        Rpos = settings(ii).Rpos;
        R = zeros(3,3);
        R(1:3,1:3) = eye(3)*Rpos;
    
        % Matriz de ruido del proceso (Q)
    
        Qpos = settings(ii).Qpos;
        Qvel = settings(ii).Qvel;
        Qorient = settings(ii).Qorient;
        Qacc = settings(ii).Qacc;
        Qgyr = settings(ii).Qgyr;
        Q = zeros(15,15);
        Q(1:3,1:3) = eye(3)*Qpos;
        Q(4:6,4:6) = eye(3)*Qvel;
        Q(7:9,7:9) = eye(3)*Qorient;
        Q(10:12,10:12) = eye(3)*Qacc;
        Q(13:15,13:15) = eye(3)*Qgyr;
    
        % Matrices de covarianza del estado (P) y de ganancia de Kalman (K)
        % iniciales
    
        P = zeros(15,15);
        K = zeros(15,3);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%% IMPLEMENTACION DEL SISTEMA %%%%%%%%%%%%%%%%%%%%
    
        for i=1:(NN-1) 
    
            % CORRECION DE MEDIDAS DE LA IMU 
            % con el accBias y el gyrBias
            % estimados por el EKF
    
            accX = accel_flu(i,1) + x(10);
            accY = accel_flu(i,2) + x(11);
            accZ = accel_flu(i,3) + x(12);
            acc = [accX;accY;accZ]; % medidas corregidas de aceleracion
            gyrX = gyro_flu(i,1) + x(13);
            gyrY = gyro_flu(i,2) + x(14);
            gyrZ = gyro_flu(i,3) + x(15);
            gyr = [gyrX;gyrY;gyrZ]; % medidas corregidas de velocidad angular
            
    
            % ECUACIONES DE NAVEGACION
    
            yaw=ypr(1); % angulos de Euler
            pitch=ypr(2);
            roll=ypr(3);
    
            Rbn=[cos(yaw) -sin(yaw) 0;sin(yaw) cos(yaw) 0; 0 0 1]; % matriz de rotacion
    
            accENU = Rbn*acc; % aceleracion en coordenadas ENU
            accE = accENU(1);
            accN = accENU(2);
            accU = accENU(3);
            
            pIMU = pIMU + Ts*vIMU + Ts^2/2*(accENU-g); % posicion en coordenadas ENU
            vIMU = vIMU + Ts*(accENU-g); % velocidad en coordenadas ENU
            ypr = ypr + Ts*[0 sin(roll)*sec(pitch) cos(roll)*sec(pitch);0 cos(roll) -sin(roll);1 sin(roll)*tan(pitch) cos(roll)*tan(pitch)]*gyr; % orientacion
         
            % Se modifican los angulos de Euler para que esten dentro de los
            % limites de los datos de KITTI y el calculo de los errores de
            % orientacion sea mas exacto.
            % El unico motivo de esta modificacion es el calculo de errores, no
            % tiene nigun efecto en la implementacion del sistema.
            % Solo se modifica el angulo yaw porque es el unico que puede
            % llegar a sobrepasar los limites en el caso de un vehiculo
            % terrestre.
    
            % Modificacion del angulo yaw con limites (-pi, +pi)
            if(ypr(1)<-3.14)
                ypr(1) = 2*pi+ypr(1);
            end
            if(ypr(1)>=3.14)
                ypr(1) = -2*pi+ypr(1);
            end
          
            % FILTRO DE KALMAN (EKF)
            
            % Variables necesarias para la matriz dinamica del sistema
    
            llaIMU = enu2lla(pIMU',lla0,'ellipsoid'); % posicion estimada en coordenadas geodesicas
            lat = llaIMU(1,1);
            alt = llaIMU(1,3);
    
            M = a*(1-ecc^2)/((1-ecc^2*sind(lat)^2)^(3/2)); % radio de curvatura en el meridiano
            N = a/sqrt(1-ecc^2*sind(lat)^2); % radio de curvatura en el plano vertical principal
    
            % Matriz dinamica del sistema (F)
    
            F = zeros(15,15);
            F(1:3,4:6) = eye(3);
            F(4,8) = accU;
            F(4,9) = -accN;
            F(5,7) = -accU;
            F(5,9) = accE;
            F(6,7) = accN;
            F(6,8) = -accE;
            F(7,5) = 1/(M+alt);
            F(8,4) = 1/(N+alt);
            F(9,4) = tand(lat)/(N+alt);
            F(4:6,10:12) = Rbn;
            F(7:9,13:15) = Rbn; 
            F(10:12,10:12) = -beta_acc*eye(3);
            F(13:15,13:15) = -beta_gyr*eye(3);
    
            % Prediccion
            
            F = eye(15) + F*Ts;
            P = F*P*F' + Q;
            x = F * x;
    
            % Actualizacion
    
            if((i<A) || (i>B)) % si estamos fuera del tramo de perdida de GPS
            
                if ~mod(i,frecIMU/frecGPS) % si se recibe medida de GPS (frecGPS = 1 Hz)
                    
                    
                    H = [eye(3) zeros(3,12)]; % Matriz de observacion
                    K = (P*H')/(H*P*H' + R); % Ganancia de Kalman
                    z = lla2enu(lla(i,:),lla0,'ellipsoid')' - pIMU; % Error de posicion entre medida de GPS en coordenadas ENU 
                                                                    % y posicion calculada por ecuaciones de navegacion
                    x = x + K*(z - H*x); 
                    P = (eye(15) - K*H)*P;
                    
                    % Correccion de estado de navegacion (cuando se completa la
                    % actualizacion)
                    pIMU = pIMU + x(1:3); % posicion corregida 
                    vIMU = vIMU + x(4:6); % velocidad corregida 
                    ypr(1,1) = ypr(1,1) + x(9); % orientacion corregida
                    ypr(2,1) = ypr(2,1) + x(8);
                    ypr(3,1) = ypr(3,1) + x(7);           
                    
                end
            
            end
    
            % GUARDAR RESULTADOS DE LA ITERACION ACTUAL
            
            p(i+1,:) = pIMU'; % posicion
            v(i+1,:) = vIMU'; % velocidad
            r(i+1,1) = ypr(3); % orientacion
            r(i+1,2) = ypr(2);
            r(i+1,3) = ypr(1);
            estado(i+1,:) = x'; % estado del EKF
    
            % REINICIO DE ERRORES DE NAVEGACION:
            % cuando se produce la actualizacion del filtro y se corrigen el
            % estado de navegacion, se asume que los errores de navegacion son
            % nulos tras la correccion
    
             if((i<A) || (i>B))
                if ~mod(i,frecIMU/frecGPS)
                    x(1:9) = zeros(9,1);
                end
             end
            
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        %Errores RMS
        %De posicion en coordenadas ENU
        posErr1 = lla2enu(lla,lla0,'ellipsoid') - p;
        ErrX1=sqrt(mean(posErr1(:,1).^2));
        ErrY1=sqrt(mean(posErr1(:,2).^2));
        ErrZ1=sqrt(mean(posErr1(:,3).^2));
        pRMS1(i) = sqrt(ErrX1^2+ErrY1^2+ErrZ1^2);
        
        %De posicion en coordenadas lla
        posErr2 = lla - enu2lla(p,lla0,'ellipsoid');
        ErrX2=sqrt(mean(posErr2(:,1).^2));
        ErrY2=sqrt(mean(posErr2(:,2).^2));
        ErrZ2=sqrt(mean(posErr2(:,3).^2));
        pRMS2(i) = sqrt(ErrX2^2+ErrY2^2+ErrZ2^2);
    
        %De velocidad en coordenadas ENU
        velErr = vel_enu - v;
        ErrVX=sqrt(mean(velErr(:,1).^2));
        ErrVY=sqrt(mean(velErr(:,2).^2));
        ErrVZ=sqrt(mean(velErr(:,3).^2));
        vRMS(i) = sqrt(ErrVX^2+ErrVY^2+ErrVZ^2);

    end
end