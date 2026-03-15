% Script para extraer los datos de los archivos de un dataset de KITTI 

% Directorio del dataset de KITTI:
directoryTXT = '~/Downloads/2011_09_26/2011_09_26_drive_0001_sync/oxts/data';
% Directorio donde se guardaran los datos en formato double:
directoryDOUBLE = '.';

% Lista de los archivos .txt en directoryTXT
fileList = dir(fullfile(directoryTXT, '*.txt'));

% Para guardar el contenido de cada archivo
fileContents = cell(1, length(fileList));

% Leer el contenido de cada archivo
for i = 1:length(fileList)
    % Construir el file path
    filePath = fullfile(directoryTXT, fileList(i).name);
    
    % Leer el contenido
    fileContent = fileread(filePath);
    
    % Guardar el contenido
    fileContents{i} = fileContent;
end

% Unir el contenido de todos los archivos con un espacio entre ellos
mergedContent = strjoin(fileContents, ' ');

% Directorio donde se guarda el nuevo .txt con todos los datos
mergedFilePath = fullfile(directoryDOUBLE, 'mergedFile.txt');

% Meter todos los datos unidos en el archivo 'mergedFile.txt'
fid = fopen(mergedFilePath, 'w');
if fid == -1
    error('Cannot open file for writing: %s', mergedFilePath);
end

fprintf(fid, '%s', mergedContent);
fclose(fid);

% Meter los datos del .txt en la variable data1 en formato double
data1=textscan(fopen('mergedFile.txt'),'%f','Delimiter','\t');
data1=cell2mat(data1);

% Reordenar los datos de data1 para que quede un tipo de dato por columna
% nº columnas=nº de tipo de dato (en los dataset de KITTI es 30)
% nº filas=nº de archivos con datos proporcionados en el dataset
% los datos ordenados se guardan en data
N=size(data1,1);
n = N/30; % numero de archivos del dataset
data=zeros(n,30); % donde se guardan los datos en formato double
k=1;
for i=1:n
    data(i,:)=data1(k:k+29,1);
    k=k+30;
end

% Extraer los datos de data

lla=data(:,1:3);
rpy=data(:,4:6);
vel_ne=data(:,7:8);
vel_flu=data(:,9:11);
accel_xyz=data(:,12:14);
accel_flu=data(:,15:17);
gyro_xyz=data(:,18:20);
gyro_flu=data(:,21:23);
pos_accuracy=data(:,24);
vel_accuracy=data(:,25);
navstat=data(:,26);
numsats=data(:,27);
posmode=data(:,28);
velmode=data(:,29);
orimode=data(:,30);

% Pasar los datos de velocidad a coordenadas ENU
[vel_enu]=sistKITTIaENU(vel_flu,rpy);

% Habra que extraer del workspace las variables que nos interesan:
% data,lla,rpy,accel_flu,accel_xyz,gyro_flu,gyro_xyz,
%  vel_flu,vel_enu,vel_ne,pos_accuracy,vel_accuracy,navstat,numsats,
% posmode,velmode,orimode

% Borrar 'mergedFile.txt' del directorio