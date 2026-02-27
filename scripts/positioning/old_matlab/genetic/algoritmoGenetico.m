% Se guardan en 'settings' las 5 mejores combinaciones de parametros
settings = importdata("best_settings.mat");

% Se generan nuevas combinaciones a partir de las 5 mejores, sin perder
% estas 5 mejores
n_settings = 50;
[settings] = generate_random_settings(n_settings, settings);

% Se calcula el error del filtro con cada una de las combinaciones
% pRMS1: error en coordenadas ENU
% pRMS2: error en coordenadas lla
% vRMS: error en velocidad
% rRMS: error en orientacion
[pRMS1,pRMS2,vRMS]=RMS_KalmanErrores(settings,0,0,accel_flu,gyro_flu,vel_enu,lla,rpy);

%Se ordenan los errores en sentido ascendente
[pRMS1_sort, index] = sort(pRMS1,'ascend');

%Guardar las 5 combinaciones de parametros con el menor error
for j = 1:5
    best_settings(j) = settings(index(j));
    best_pRMS1(j) = pRMS1(index(j));
end
save("best_settings", "best_settings");
save("best_pRMS1", "best_pRMS1");