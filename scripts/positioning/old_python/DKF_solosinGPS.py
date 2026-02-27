#%%
# Configuracion
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
from math import sqrt
from math import sin
from math import cos
from math import tan
from math import radians
import pymap3d as pm
from pandas import DataFrame
from pandas import concat
from scipy.io import loadmat
from pathlib import Path
#%%
################### CARGA DE DATOS ###########################
# Load raw KITTI ('10_03_0034.mat')
base_dir = Path(__file__).parent
data_path = os.path.join(base_dir, '../../../', 'datasets','raw_kitti', '10_03_0034.mat')
data = loadmat(data_path)

accel_flu = data['accel_flu']
gyro_flu = data['gyro_flu']
vel_enu = data['vel_enu']
lla = data['lla']
orient = data['rpy']
estado = data['estado']

accel_flu = pd.DataFrame(accel_flu)
accel_flu.columns = ['accelX','accelY','accelZ']
accel_flu.to_csv('accel_flu.csv')
accel_flu = read_csv('accel_flu.csv', header=0, index_col=0)
accel_flu = accel_flu.values

gyro_flu = pd.DataFrame(gyro_flu)
gyro_flu.columns = ['gyroX','gyroY','gyroZ']
gyro_flu.to_csv('gyro_flu.csv')
gyro_flu = read_csv('gyro_flu.csv', header=0, index_col=0)
gyro_flu = gyro_flu.values

vel_enu = pd.DataFrame(vel_enu)
vel_enu.columns = ['velE','velN','velU']
vel_enu.to_csv('vel_enu.csv')
vel_enu = read_csv('vel_enu.csv', header=0, index_col=0)
vel_enu = vel_enu.values

lla = pd.DataFrame(lla)
lla.columns = ['lat','lon','alt']
lla.to_csv('lla.csv')
lla = read_csv('lla.csv', header=0, index_col=0)
lla = lla.values

orient = pd.DataFrame(orient)
orient.columns = ['roll','pitch','yaw']
orient.to_csv('orient.csv')
orient = read_csv('orient.csv', header=0, index_col=0)
orient = orient.values

estado = pd.DataFrame(estado)
estado.columns = ['posErrX','posErrY','posErrZ','velErrX','velErrY','velErrZ','orientErrR','orientErrP','orientErrY','accBiasX','accBiasY','accBiasZ','gyrBiasX','gyrBiasY','gyrBiasZ']
estado.to_csv('estado.csv')
estado = read_csv('estado.csv', header=0, index_col=0)
estado = estado.values

# Cargar datos de la trayectoria de KITTI con los datos de entrenamiento de la red LSTM de la carpeta 'dataKITTI' (ejemplo: '10_03_0027.mat')
data_train_path = os.path.join(base_dir, '../../../', 'datasets','raw_kitti', '10_03_0027.mat')
data_train = loadmat(data_train_path)

estado_train = data_train['estado']
estado_train = pd.DataFrame(estado_train)
estado_train.columns = ['posErrX','posErrY','posErrZ','velErrX','velErrY','velErrZ','orientErrR','orientErrP','orientErrY','accBiasX','accBiasY','accBiasZ','gyrBiasX','gyrBiasY','gyrBiasZ']
estado_train.to_csv('errores.csv')
dataset_train = read_csv('errores.csv', header=0, index_col=0)

#%%
############################## DISEÑO Y ENTRENAMIENTO DE LA RED ###################################

# PREPARACION DE DATOS DE ENTRENAMIENTO

# Funcion para crear las ventanas para el entrenamiento

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# Secuencia de entrada (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# Secuencia predicha (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	agg = concat(cols, axis=1)
	agg.columns = names

	if dropnan:
		agg.dropna(inplace=True)
	return agg

values_train = dataset_train.values
values_train = values_train.astype('float32')
n_in = 10 # numero de pasos de tiempo de entrada a la red
values_train = series_to_supervised(values_train, n_in, 1)

# Dividir en train (70 %)  y test (30 %)

values = values_train.values
n_train = int(len(values)*0.7)
train = values[:n_train, :]
test = values[n_train:, :]

# Clasificar en entrada y salida de la red:
# los ultimos 15 datos corresponden al vector de errores de salida de la red,
# los datos anteriores son los 10 vectores de errores de entrada

train_X, train_y = train[:, :-15], train[:, -15:]
test_X, test_y = test[:, :-15], test[:, -15:]

# Dimensionar a 3D [samples, timesteps, features=15]

train_X = train_X.reshape((train_X.shape[0], n_in, 15))
test_X = test_X.reshape((test_X.shape[0], n_in, 15))

# DISEÑO DE LA RED

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.callbacks import LambdaCallback
from tensorflow.keras.layers import InputLayer

# Caracteristicas

activation_function = 'sigmoid'
optimizer = 'adam'
loss_function = 'mean_squared_error'
batch_size = 1
num_units = 40
num_epochs = 50
dropout=0.2

# Montar las capas

model = Sequential()
model.add(InputLayer(batch_input_shape=(batch_size,10,15)))
model.add(LSTM(units = num_units, activation = activation_function, stateful=True, return_sequences=True,recurrent_dropout=dropout))
model.add(LSTM(units = num_units, activation = activation_function, stateful=True,return_sequences=True,recurrent_dropout=dropout))
model.add(LSTM(units = num_units, activation = activation_function, stateful=True,return_sequences=False,recurrent_dropout=dropout))
model.add(Dense(units = 15))

# Compilar

model.compile(optimizer = optimizer, loss = loss_function)

# Funcion para resetear los estados en el entrenamiento (se usa 'stateful=True')
def reset_rnn_states():
    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()
            
resetCallback = LambdaCallback(on_epoch_begin=lambda epoch,logs: reset_rnn_states())

# Entrenar

history = model.fit(train_X,train_y,batch_size=batch_size, validation_data=(test_X , test_y) , epochs = num_epochs, callbacks=[resetCallback],verbose=1, shuffle=False)

# Mostrar los resultados de entrenamiento graficamente

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
#%% 
########################## DATOS Y VARIABLES PARA IMPLEMENTAR EL SISTEMA ###############

# DATOS VARIOS

t1 = 0  # instante inicial de perdida de GPS en segundos
d = 0   # duracion del tramo de perdida de GPS en segundos
        # Si no se quiere simular un tramo de perdida de GPS: t1=0, d=0
       
NN = lla.shape[0]   # numero de datos en la trayectoria
frecIMU = 10        # 10 Hz
frecGPS = 1         # 1 Hz

t2 = t1+d                               # instante de final de perdida de GPS en segundos
A = t1*frecIMU                          # nº de dato de inicio de perdida de GPS
B = t2*frecIMU                          # nº de dato de fin de perdida de GPS
Ts= 1/frecIMU                           # tiempo de muestreo
g = np.array([0,0,9.81])                # aceleracion de la gravedad
lla0 = np.array([49.01,8.43,116.43])    # origen de coordenada para el sistema ENU,
                                        # se escoge una coordenada cercana a las
                                        # trayectorias de KITTI
sigma = 7.2921151467e-5                 # rotacion de la Tierra [rad/s]
a = 6378e3                              # radio ecuatorial [m]
b = 6357e3                              # radio polar [m]
ecc = sqrt(1-b**2/a**2)                 # eccentricidad

# VARIABLES PARA GUARDAR LOS ESTADOS DE NAVEGACION ESTIMADOS

# Los primeros n_in instantes de tiempo de la trayectoria se utilizan como entrada de la LSTM en la primera iteracion;
# la implementacion del sistema empieza en el instante de tiempo n_in+1 (posicion n_in en los vectores de datos de KITTI)

p = np.zeros((NN,3)) # para guardar las posiciones estimadas
v = np.zeros((NN,3)) # para guardar las velocidades estimadas
r = np.zeros((NN,3)) # para guardar las orientaciones estimadas

# Como primer estado de navegacion se guardan los datos de KITTI
# Esta no es la inicialización del estado de navegación; estas
# variables son exclusivamente para guardar resultados
p[0,:] = pm.geodetic2enu(lla[n_in,0],lla[n_in,1],lla[n_in,2],lla0[0],lla0[1],lla0[2])
v[0,:] = vel_enu[n_in,:]
r[0,:] = orient[n_in,:]

# INICIALIZACION DEL ESTADO DE NAVEGACION

pIMU = p[0,:].T     # posicion inicial conocida
vIMU = v[0,:].T     # velocidad inicial conocida
ypr = np.zeros(3)   # orientacion inicial desconocida, ypr = (yaw, pitch, roll)

# INICIALIZACION DE COMPONENTES DEL FILTRO

# Estado

x = np.zeros(15) # estado del filtro de Kalman
                 # x = (errorPos, errorVel, errorOrient, accBias, gyrBias)
accBias = 1e-6   # bias inicial
gyrBias = 1e-7   # bias inicial
x[9:12] = accBias
x[12:15] = gyrBias

# Entrada a la red LSTM

x_prev = np.zeros((15,n_in))

# Coeficientes de Gauss-Markov

beta_acc = 3.7e-7
beta_gyr = 2.9e-1

# Matriz de ruido de medicion (R)

Rpos = 2
R = np.eye(3)*Rpos

# Matriz de ruido del proceso (Q)

Qpos = 2
Qvel = 2
QorientX = 0.0002
QorientY = 0.0002
QorientZ = 0.2
Qacc = 0.1
QgyrX = 0.0001
QgyrY = 0.0001
QgyrZ = 0.1
Q = np.zeros((15,15))
Q[0:3,0:3] = np.eye(3)*Qpos
Q[3:6,3:6] = np.eye(3)*Qvel
Q[6:9,6:9] = np.array([[QorientX,0,0],[0,QorientY,0],[0,0,QorientZ]])
Q[9:12,9:12] = np.eye(3)*Qacc
Q[12:16,12:16] = np.array([[QgyrX,0,0],[0,QgyrY,0],[0,0,QgyrZ]])

# Matriz de observacion (H)

H = np.zeros((3,15))
H[0:3,0:3]=np.eye(3)

# Matrices de covarianza del estado (P) y de ganancia de Kalman (K) iniciales

P = np.zeros((15,15))
K = np.zeros((15,3))
#%%
################## IMPLEMENTACION DEL SISTEMA #########################

for i in range(0,NN-1):
    
   # CORRECION DE MEDIDAS DE LA IMU 
   # con el accBias y el gyrBias
   # estimados por el EKF
        
    accX = accel_flu[i,0] + x[9]
    accY = accel_flu[i,1] + x[10]
    accZ = accel_flu[i,2] + x[11]
    acc = np.array([accX,accY,accZ]) # medidas corregidas de aceleracion
    gyrX = gyro_flu[i,0] + x[12]
    gyrY = gyro_flu[i,1] + x[13]
    gyrZ = gyro_flu[i,2] + x[14]
    gyr = np.array([gyrX,gyrY,gyrZ]) # medidas corregidas de velocidad angular

    # ECUACIONES DE NAVEGACION
    
    yaw = ypr[0] # angulos de Euler
    pitch = ypr[1]
    roll = ypr[2]
    
    Rbn=np.array([[cos(yaw),-sin(yaw),0],[sin(yaw),cos(yaw),0],[0,0,1]]) # matriz de rotacion
    
    accENU = Rbn@acc # aceleracion en coordenadas ENU
        
    pIMU = pIMU + Ts*vIMU + Ts**2/2*(accENU-g) # posicion en coordenadas ENU
    vIMU = vIMU + Ts*(accENU-g) # velocidad en coordenadas ENU
    ypr = ypr + Ts*np.array([[0,sin(roll)/cos(pitch),cos(roll)/cos(pitch)],[0,cos(roll),-sin(roll)],[1,sin(roll)*tan(pitch),cos(roll)*tan(pitch)]])@gyr # orientacion
    
    # Se modifican los angulos de Euler para que esten dentro de los
    # limites de los datos de KITTI y el calculo de los errores de
    # orientacion sea mas exacto.
    # El unico motivo de esta modificacion es el calculo de errores, no
    # tiene nigun efecto en la implementacion del sistema.
    # Solo se modifica el angulo yaw porque es el unico que puede
    # llegar a sobrepasar los limites en el caso de un vehiculo
    # terrestre.

    # Modificacion del angulo yaw con limites (-pi, +pi)
    
    if(ypr[0]<-3.14):
        ypr[0] = 2*np.pi+ypr[0]
    if(ypr[0]>=3.14):
        ypr[0] = -2*np.pi+ypr[0]
    

    # FILTRO DE KALMAN (EKF)
    
    # Variables necesarias para la matriz dinamica del sistema
    
    accE = accENU[0]
    accN = accENU[1]
    accU = accENU[2]
    
    llaIMU = pm.enu2geodetic(pIMU[0],pIMU[1],pIMU[2],lla0[0],lla0[1],lla0[2]) # posicion estimada en coordenadas geodesicas
    lat = radians(llaIMU[0])
    alt = llaIMU[2]
    
    M = a*(1-ecc**2)/((1-ecc**2*sin(lat)**2)**(3/2)) # radio de curvatura en el meridiano
    N = a/sqrt(1-ecc**2*sin(lat)**2) # radio de curvatura en el plano vertical principal

    # Matriz dinamica del sistema (F)

    F = np.zeros((15,15))
    F[0:3,3:6] = np.eye(3)
    F[3,7] = accU
    F[3,8] = -accN
    F[4,6] = -accU
    F[4,8] = accE
    F[5,6] = accN
    F[5,7] = -accE
    F[6,4] = 1/(M+alt)
    F[7,3] = 1/(N+alt)
    F[8,3] = tan(lat)/(N+alt)
    F[3:6,9:12] = Rbn
    F[6:9,12:15] = Rbn
    F[9:12,9:12] = -beta_acc*np.eye(3)
    F[12:15,12:15] = -beta_gyr*np.eye(3)
        
    # Prediccion
    
    F = np.eye(15) + F*Ts
    P = F@P@F.T + Q
    x_pred = F@x
    
    
    # Actualizacion
    
    if ((i+1)<A or (i+1)>B): # si estamos fuera del tramo de perdida de GPS
    
        if not((i+1)%(frecIMU/frecGPS)): # si se recibe medida de GPS (frecGPS = 1 Hz)
            
            K = P@H.T@np.linalg.inv(H@P@H.T + R) # Ganancia de Kalman
            z = pm.geodetic2enu(lla[i,0],lla[i,1],lla[i,2],lla0[0],lla0[1],lla0[2]) - pIMU # Error de posicion entre medida de GPS en coordenadas ENU 
                                                                                           # y posicion calculada por ecuaciones de navegacion
            x = x + K@(z - H@x)
            P = (np.eye(15) - K@H)@P
    
            # CORRECCION DE ESTADO DE NAVEGACION (cuando se completa la actualizacion)
            pIMU = pIMU + x[0:3]    # posicion corregida
            vIMU = vIMU + x[3:6]    # velocidad corregida
            ypr[0] = ypr[0] + x[8]  # orientacion corregida
            ypr[1] = ypr[1] + x[7]
            ypr[2] = ypr[2] + x[6]
            
        else:
            x = x_pred
    else: # cuando estamos en el tramo de perdida de GPS
    
        reset_rnn_states()
        values = x_prev.astype('float32') 
        values = values.reshape(n_in,values.shape[0]) # Redimensionar los datos de entrada a la red
        x_deep = model.predict(values.reshape(1,n_in,values.shape[1]))
        x_pred = x_deep.reshape(15,) # Redimensionar la salida de la red
        
        # CORRECCION DE ESTADO DE NAVEGACION (cuando la red hace la estimacion)
        pIMU = pIMU + x[0:3] # posicion corregida
        vIMU = vIMU + x[3:6] # vekicidad corregida
        ypr[0] = ypr[0] + x[8] # orientacion corregida
        ypr[1] = ypr[1] + x[7]
        ypr[2] = ypr[2] + x[6]
    
    # GUARDAR NUEVO VECTOR DE ERRORES EN LA ENTRADA DE LA RED (en 'x_prev' estan los n_in ultimos vectores de errores)
       
    for ii in range(0,n_in-1):
        x_prev[:,ii] = x_prev[:,ii+1]
    x_prev[:,n_in-1] = x
        
    # GUARDAR RESULTADOS DE LA ITERACION ACTUAL
    
    p[i+1,:] = pIMU.T # posicion
    v[i+1,:] = vIMU.T # velocidad
    r[i+1,0] = ypr[2] # orientacion
    r[i+1,1] = ypr[1]
    r[i+1,2] = ypr[0]

    # REINICIO DE ERRORES DE NAVEGACION:
    # cuando se produce la actualizacion del filtro y se corrigen el
    # estado de navegacion, se asume que los errores de navegacion son
    # nulos tras la correccion
    if ((i+1)<A or (i+1)>B):
        if not((i+1)%(frecIMU/frecGPS)):
            x[0:9] = 0
#%%
########################### MOSTRAR RESULTADOS Y ERRORES ########################

# DIBUJAR LA TRAYECTORIA

f = pm.geodetic2enu(lla[:,0],lla[:,1],lla[:,2],lla0[0],lla0[1],lla0[2]) # pasar la trayectoria real de geodesicas a ENU

plt.plot(f[0],f[1],'k',label='Trayectoria') # trayectoria real en negro
plt.plot(p[:,0],p[:,1],'b',label='EKF + LSTM') # trayectoria estimada por el sistema en azul

if A!=0 or B!=0: # tramo perdida de datos de GPS 
    plt.plot(p[A:B,0],p[A:B,1],'r',label='EKF + LSTM en pérdida GPS') # salida del sistema en rojo
    plt.plot(f[0][A:B],f[1][A:B],'g',label='Trayectoria en pérdida GPS' ) # trayectoria real en verde
plt.legend()
plt.show()

# ERRORES 

# Error posicion en coordenadas ENU
pRMS_E = sqrt(mean_squared_error(f[0],p[:,0]))
pRMS_N = sqrt(mean_squared_error(f[1],p[:,1]))
pRMS_U = sqrt(mean_squared_error(f[2],p[:,2]))
    
# Error velocidad en coordenadas ENU
vRMS_E = sqrt(mean_squared_error(vel_enu[:,0],v[:,0]))
vRMS_N = sqrt(mean_squared_error(vel_enu[:,1],v[:,1]))
vRMS_U = sqrt(mean_squared_error(vel_enu[:,2],v[:,2]))

# Error orientacion en angulos de Euler
rRMS_R = sqrt(mean_squared_error(orient[:,0],r[:,0]))
rRMS_P = sqrt(mean_squared_error(orient[:,1],r[:,1]))
rRMS_Y = sqrt(mean_squared_error(orient[:,2],r[:,2]))
    
print('Error en posicion E: %.3f' % pRMS_E)
print('N: %.3f' %pRMS_N)
print('U: %.3f' %pRMS_U)

print('Error en velocidad E: %.3f' % vRMS_E )
print('N: %.3f' %vRMS_N)
print('U: %.3f' %vRMS_U)

print('Error en orientacion roll: %.3f' % rRMS_R)
print('pitch: %.3f' %rRMS_P)
print('yaw: %.3f' %rRMS_Y)

# Errores absolutos de posicion 
error_E_DKF2=np.zeros((p.shape[0],1))
error_N_DKF2=np.zeros((p.shape[0],1))

for i in range(0,p.shape[0]-1):
    error_E_DKF2[i]= abs(f[0][i]-p[i,0])
    error_N_DKF2[i]= abs(f[1][i]-p[i,1])
        
# Errores absolutos de orientacion 
error_yaw_DKF2=np.zeros((1000,1))

for i in range(0,1000):
    error_yaw_DKF2[i]= abs(orient[i,2]-r[i,2])


