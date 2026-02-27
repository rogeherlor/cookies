function [p,estado]=EKF(t1,d,accel_flu,gyro_flu,vel_enu,lla,rpy)
    
    % t1: instante inicial de perdida de GPS en segundos
    % d: duracion del tramo de perdida de GPS en segundos
    % Si no se quiere simular un tramo de perdida de GPS: t1=0, d=0
    
    %%%%%%%%%%%%% DATOS Y VARIABLES PARA IMPLEMENTAR EL SISTEMA %%%%%%%%%%%

    % DATOS VARIOS

    NN = size(lla,1); % numero de datos en la trayectoria
    frecIMU = 10; % 10 Hz
    frecGPS = 1; % 1 Hz (para crear el dataset de entrenamiento de la LSTM se pone a 10 Hz)
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
    ypr=zeros(3,1); % orientacion inicial desconocida, ypr = (yaw, pitch, roll)
    
    % % Se inicializa la orientacion como conocida cuando se crea el dataset
    % % de entrenamiento de la LSTM
    % ypr(1,1)=rpy(1,3);
    % ypr(2,1)=rpy(1,2);
    % ypr(3,1)=rpy(1,1);

    % INICIALIZACION DE COMPONENTES DEL FILTRO

    % Estado

    x = zeros(15,1); % estado del filtro de Kalman
                     % x = (errorPos, errorVel, errorOrient, accBias, gyrBias)
    accBias = [1e-6;1e-6;1e-6]; % bias inicial
    gyrBias = [1e-7;1e-7;1e-7]; % bias inicial
    x(10:12) = accBias;
    x(13:15) = gyrBias;
    estado = zeros(NN,15); % para guardar los vectores de estado del filtro
                           % de Kalman generados durante toda la trayectoria 
    estado(1,:) = x';
    
    % Coeficientes de Gauss-Markov

    beta_acc = 3.7e-7;
    beta_gyr = 2.9e-1;

    % Matriz de ruido de medicion (R)

    Rpos = 2;
    R = zeros(3,3);
    R(1:3,1:3) = eye(3)*Rpos;

    % Matriz de ruido del proceso (Q)

    Qpos = 2;
    Qvel = 2;
    QorientX = 0.0002;
    QorientY = 0.0002;
    QorientZ = 0.2;
    Qacc = 0.1;
    QgyrX = 0.0001;
    QgyrY = 0.0001;
    QgyrZ = 0.1;
    Q = zeros(15,15); 
    Q(1:3,1:3) = eye(3)*Qpos;
    Q(4:6,4:6) = eye(3)*Qvel;
    Q(7:9,7:9) = [QorientX 0 0;0 QorientY 0;0 0 QorientZ];
    Q(10:12,10:12) = eye(3)*Qacc;
    Q(13:15,13:15) = [QgyrX 0 0;0 QgyrY 0;0 0 QgyrZ];

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

        accE = accENU(1);
        accN = accENU(2);
        accU = accENU(3);
        
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

    %%%%%%%%%%%%%%%%%%% MOSTRAR RESULTADOS Y ERRORES %%%%%%%%%%%%%%%%%%%%%

    % DIBUJAR LA TRAYECTORIA

    p_lla=enu2lla(p,lla0,'ellipsoid'); % pasar las posiciones estimadas de ENU a geodesicas
    lon2=p_lla(:,2); % longitudes de posiciones estimadas
    lat2=p_lla(:,1); % latitudes de posiciones estimadas
    lat1=lla(:,1); % latitudes de posiciones dadas por el GPS (trayectoria real)
    lon1=lla(:,2); % longitudes de posiciones dadas por el GPS (trayectoria real)
    geoplot(lat1,lon1,'k');% trayectoria real en negro
    hold on;
    geoplot(lat2,lon2,'b');% trayectoria estimada por el sistema en azul
    hold on;
    if((A ~= 0) || (B ~= 0))% tramo perdida de datos de GPS en lineas mas gruesas
        geoplot(lat1(A:B,:),lon1(A:B,:),'k','LineWidth',1.5);
        geoplot(lat2(A:B,:),lon2(A:B,:),'b','LineWidth',1.5);
    end

    % ERRORES (RMSE)

    % De posicion en coordenadas ENU
    posErr1 = lla2enu(lla,lla0,'ellipsoid') - p;
    pRMS1 = sqrt(mean(posErr1.^2));
    
    % De posicion en coordenadas geodesicas
    posErr2 = lla - p_lla;
    pRMS2 = sqrt(mean(posErr2.^2));

    % De velocidad en coordenadas ENU
    velErr = vel_enu - v;
    vRMS = sqrt(mean(velErr.^2));

    % De orientacion en angulos de Euler
    rErr = rpy - r; 
    rRMS = sqrt(mean(rErr.^2));
    
    fprintf('Error RMS de posicion\n\tEn coordenadas geodesicas:\n\t\tlat: %.2f, lon: %.2f, alt: %.2f\n\tEn coordenadas ENU:\n\t\tX: %.2f, Y: %.2f, Z: %.2f\n\n',pRMS2(1),pRMS2(2),pRMS2(3),pRMS1(1),pRMS1(2),pRMS1(3));
    fprintf('Error en velocidad\n\tEn coordenadas ENU:\n\t\tE: %.2f, N:%.2f, U: %.2f\n',vRMS(1),vRMS(2),vRMS(3)); 
    fprintf('Error de orientacion\n\tEn rad:\n\t\tRoll: %.2f, Pitch:%.2f, Yaw:%.2f\n',rRMS(1),rRMS(2),rRMS(3)); 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
    