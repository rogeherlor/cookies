-----------------------------------------------------------------------------------------------
-
- El código incluye implementación de EKF y DKF; y pruebas asociadas con los algoritmos
- El siguiente código ha sido desarrollado en la Universidad Politécnica de Madrid (UPM)
- Versión actualizada 10.2025
-
-----------------------------------------------------------------------------------------------
 
A continuacion se presenta una breve explicacion de los contenidos de esta carpeta.


MATLAB


‘Matlab’: scripts y datasets en Matlab.


‘AlgoritmoGenetico’: se ejecuta el script ‘algoritmoGenetico.m’ y se guardan los datos que dan mejores resultados en ‘best_setting.mat’ y sus errores correspondientes en ‘best_pRMS1.mat’.


‘dataKITTI’: contiene los datasets de las trayectorias de KITTI en formato .mat. ‘mergeTXT.m’ es el script para crear el .mat a partir de una trayectoria descargada de KITTI, que utiliza ‘sistKITTIaENU.m’ para obtener datos en coordenadas ENU.


‘EKF.m’: es el codigo del filtro de Kalman extendido estandar. Se abre uno de los .mat con datos de una trayectoria en el workspace y se llama a la funcion EKF, indicando el tramo de perdida de GPS con las variables ‘t1’ y ‘d’.


PYTHON


‘Python’: codigo en Python para trabajar con redes neuronales usando tensorflow. Los datasets de las trayectorias se cargan de ‘dataKITTI’ indicando el directorio y el nombre de la trayectoria con la que se quiere trabajar (en el ejemplo se trabaja con la trayectoria 10_03_0034).


‘EKF.py’: es el codigo del filtro de Kalman extendido estandar (identico a ‘EKF.m’).


‘DKF.py’: es el EKF que utiliza una LSTM en la fase de prediccion.


‘DKF_solosinGPS.py’: utiliza el EKF cuando hay señal de GPS y la LSTM cuando no la hay.


Sobre el entrenamiento de la red: con ‘EKF.m’, inicializando la orientacion y utilizando 10 Hz para el GPS, se crea el dataset de entrenamiento ‘estado’ para una trayectoria, guardandolo en el .mat de dicha trayectoria (en el ejemplo se ha guardado ‘estado’ para las trayectorias 10_03_0027 y 10_03_0034). En ‘DKF.py’ y ‘DKF_solosinGPS.py’, ‘estado’ se utiliza para entrenar la red (en el ejemplo se utiliza ‘estado’ de la trayectoria 10_03_0027 para el entrenamiento).