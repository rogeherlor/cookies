// MAC Mode Device Sample Application
//
// Copyright 2017 Silicon Laboratories, Inc.                                *80*
#define TENGO_GPS 1				// Habilito o deshabilito directamente el GPS desde macro

#define LIMITE_DC 10			// Numero de errores de comunicacion

#if !defined(UNIX_HOST) && !defined(CORTEXM3_EMBER_MICRO)
#include "em_device.h"
#include "em_chip.h"
#include "bsp.h"
#endif

#include <icm20648_r.h>
#include "imu.h"
#include "icm20648_config.h"
#include "gps-uart.h"
#include "i2cspm.h"

#include PLATFORM_HEADER
#include CONFIGURATION_HEADER

#include "flex-callbacks.h"

#include "stack/include/ember.h"
#ifndef UNIX_HOST
#include "heartbeat/heartbeat.h"
#include "wstk-sensors/wstk-sensors.h"
#endif
//#include "poll/poll.h"

#include "hal/hal.h"
//#include "command-interpreter/command-interpreter2.h"
//#include "debug-print/debug-print.h"

#include "protocolored.h"
#include <string.h>
#include "rtcdriver.h"
#include <stdlib.h>
#include <stdio.h>

// EKF
#include "ekf/cookie_ekf.h"

typedef uint8_t EmberStatus;

#define NETWORK_UP_LED BOARDLED0

extern uint16_t destino;
extern uint16_t origen;
extern uint8_t paquete[75];							//Feb23 Tama�o paquete de datos
extern EmberMessageOptions txOptions;
extern EmberNetworkParameters parameters;

//static EmberMessageOptions txOptions = EMBER_OPTIONS_ACK_REQUESTED;
//EmberEventControl reportControl;
#if (RED_USADA!=RED_DEMO2)
uint16_t sensorReportPeriodMs =  250; 				// Periodo de "reportes" AKA envio de sensores, por defecto 250 ms inicialmente
#else
static uint16_t sensorReportPeriodMs =  1000;		// para RED_DEMO2 quiero envios cada 1 segundo
#endif
uint8_t flagStandby = 0;							// Oct22 0 manda datos, 1 no manda. No deshabilita la radio ni su consumo (pendiente)
//extern EmberStatus emberAfPluginWstkSensorsGetSample(uint32_t *rhData, int32_t *tData);

//////////////   GNSS   /////////////////			//Feb23
extern uint8_t PDOP[6];
extern uint8_t vel_GNSS[6];
uint16_t PDOP_u = 0;
uint16_t vel_GNSS_u = 0;
uint8_t n_modes = 7;								//n_modes en set_GNSS_mode
uint8_t GNSS_mode = 7;								//modo del GNSS
uint16_t GNSS_mode_check;							//al recibir mensaje comprobar que el modo no supera n_modes
uint8_t mode_selector_GNSS = 7;						//en 0 no se busca. Es el modo en el que se va a buscar. Se puede buscar modo la primera vez que se recibe GPS valido.
uint8_t const mode_selector_GNSS_min = 7;			//el valor minimo desde el cual se empieza a buscar a no ser que se fuerce por comando serial. Si es 0 no se busca.
uint16_t cont_selector_GNSS = 0;
uint16_t cont_selector_cicles = 240;				//Numero de ciclos que se prueba el PDOP de un modo durante la busqueda del mejor modo
uint8_t best_mode_selector_GNSS = 7;				//Por defecto 7
uint16_t PDOP_best = 65535;
uint16_t PDOP_lim = 400;							//PDOP umbral*100 que hace que se aumente el contador para buscar el mejor GNSS
uint8_t cont_bad_PDOP = 0;							//Numero de veces que PDOP recibido y valido supera el umbral
uint8_t cont_bad_PDOP_lim = 60;						//Umbral del numero de veces que se puede dar mal PDOP hasta que se busca nuevo
////////////////////////////////////////

extern uint8_t tengo_gps;
uint8_t tengo_datos_gps = 0;
extern uint8_t validez;
extern uint8_t tiempo[10];
extern uint8_t tiempo_ultimo_valido[10];
uint8_t tiempo_out[10] = "000000.000";
uint8_t fecha_out[6] = "000000"; 		//Mar23
//extern uint8_t tiemponuevo[10];
extern uint8_t tiempo2[10];
extern uint8_t latitud[9];
extern uint8_t norte_sur;
extern uint8_t longitud[10];
extern uint8_t este_oeste;
//extern uint8_t tamanoaltitud;
extern uint8_t altitud[8];
extern uint8_t metros;
uint8_t cuentanodos = 0;
static uint8_t tipo;
extern uint16_t cuentaDC;
uint8_t EnablePrint_H = 1;
uint8_t EnablePrint_T = 1;
uint8_t EnablePrint_A = 1;
uint8_t EnablePrint_G = 1;
uint8_t EnablePrint_GPS = 1;

static uint32_t tiempo_referencia = 0;
static uint32_t tiempo_actual = 0;
static uint32_t tiempo_anterior = 0;
static uint32_t tiempo_incremento = 0;

uint16_t rango_ack;
uint8_t flagConfig = 0;
uint16_t conf_orig;
uint16_t conf_rang;
uint16_t conf_padre;
int8_t conf_rssi;

int8_t original_data_rssi;	//rssi del nodo padre que se envia en el paquete de datos
int8_t original_data_rssi_init = 0;

//////////////   EKF   /////////////////
EKF_Context_t myEKF;
bool ekf_initialized = false;
////////////  END EKF   ////////////////


static float Ts1_sum = 0.001;
uint16_t t_max_ciclo=10;			//CUIDADO. Maxima duracion de una iteracion del kalman. Valor hardcodeado. Se utiliza para que el kalman no retrase el tiempo de envio. Se podria eliminar.
uint8_t cont_valid_gps = 0;
uint16_t cont_invalid_gps = 0;
bool flag_first_gps = true; 		//Primer gps valido cuando cumple varios gps seguidos validos
bool flag_gps0 = true;



void reportHandler(void)
{
	if (NODO_QUE_ENVIA == 0){return;}
	if (flagStandby == 1){flag_first_gps=true; cont_valid_gps=0; return;} 	//No manda datos, pero no apaga radio

	tiempo_anterior = tiempo_actual;
	tiempo_actual = RTCDRV_GetWallClockTicks32()/4;							// cojo el tiempo actual, en cada paquete

	//validez = 0x56;
	int32_t tempData = 0;
	uint32_t rhData = 0;
	// For navigation we want high resolution (low range) and LPF with small frequency
	// For integrity we prefer higher range for detecting spykes, more sampling, and LPF with high frequency.
	static float acelflo[3]={0,0,0}; 		// imu.c ICM20648_ACCEL_FULLSCALE_2G, ICM20648_ACCEL_BW_6HZ for navigation (for m/s2 *9.81)
	static float gyroflo[3]={0,0,0};  		// imu.c ICM20648_GYRO_FULLSCALE_250DPS, ICM20648_GYRO_BW_6HZ (deg/s)
	static int32_t acelint[3] = {0,0,0};
	static int32_t gyroint[3] = {0,0,0};
	float Ts1 = 0.001;
	bool Correct_GPS1 = false;

	float latitud_s = 0;
	float longitud_s = 0;
	float altitud_s = 0;
	float latitud_k = 0;
	float longitud_k = 0;
	float altitud_k = 0;
	float vel_k = 0;

	Ts1 = (float) (tiempo_actual - tiempo_anterior)/1000; 				//tiempo desde mi ultima referencia, en s
	Ts1_sum = Ts1_sum + Ts1;

	if (mi_panID != 0xFFFF && (mi_rango < 0xFFFF)){   					//solo envio si estoy conectado a algo
		IMU_getAccelerometerData(acelflo);
		IMU_getGyroData(gyroflo);

		for (int i = 0; i < 3; i++){
			acelint[i] = (int32_t) 1000*acelflo[i]; 					//Valor enviado no esta en m/s2
			gyroint[i] = (int32_t) gyroflo[i];
		}

		// Units Reference rotation
		acelflo[0] = -acelflo[0]*9.81f;		//conversion a m/s2
		float aux_acc = acelflo[1];
		acelflo[1] = -acelflo[2]*9.81f;
		acelflo[2] = -aux_acc*9.81f;

		// Units
		gyroflo[0] = -gyroflo[0]*PI/180; 	//conversion a radianes
		float aux_gyr = gyroflo[1];
		gyroflo[1] = -gyroflo[2]*PI/180;
		gyroflo[2] = -aux_gyr*PI/180;

		emberAfCorePrint("\nA = %ld, %ld, %ld;", acelint[0], acelint[1], acelint[2]);
		emberAfCorePrint("G = %ld, %ld, %ld;", gyroint[0], gyroint[1], gyroint[2]);
		emberAfCorePrint("Ts1 = %ld; Sum = %ld;", (int32_t)(Ts1*1000), (int32_t)(Ts1_sum*1000));

		if (ekf_initialized) {
			EKF_Predict(&myEKF, acelflo, gyroflo, Ts1);
			
			// Update output variables
			float lla_out[3];
			ENU_to_LLA(myEKF.pos_enu, myEKF.lla0[0], myEKF.lla0[1], myEKF.lla0[2], lla_out);
			latitud_k = lla_out[0];
			longitud_k = lla_out[1];
			altitud_k = lla_out[2];
			vel_k = sqrtf(myEKF.vel_enu[0]*myEKF.vel_enu[0] + myEKF.vel_enu[1]*myEKF.vel_enu[1] + myEKF.vel_enu[2]*myEKF.vel_enu[2]);
			emberAfCorePrint("\nlat_k: %ld; lon_k: %ld; alt_k: %ld;  vel_k: %ld", (int32_t)latitud_k, (int32_t)longitud_k, (int32_t)altitud_k, (int32_t)vel_k);
		}

		//emberAfPluginWstkSensorsGetSample(&rhData, &tempData); //Dic22 Comentado puesto que consume mucho tiempo y no se utiliza esta info
		//emberAfCorePrint("\nH = %lu.%lu (%%)\n", rhData/1000, (rhData-1000*(rhData/1000)));
		//emberAfCorePrint("T = %ld.%ld (�C)\n", tempData/1000, (tempData-1000*(tempData/1000)));
		//emberAfCorePrint("A = %ld, %ld, %ld (g x10-3)\n", acelint[0], acelint[1], acelint[2]);
		//emberAfCorePrint("G = %ld, %ld, %ld (deg/s)\n", gyroint[0], gyroint[1], gyroint[2]);


	if(Ts1_sum*1000 >= sensorReportPeriodMs - t_max_ciclo ){ //Si el tiempo de ejecucion supera el tiempo de muestreo menos un offset opcional para no retrasar el envio se mira el gps
	#if TENGO_GPS
		#if PROBANDO == 1
			  tengo_gps=1;
		#endif
			//memcpy(&tiempo, "000000.000", sizeof(tiempo));
		tengo_datos_gps = 0;
		if(tengo_gps){
			tengo_datos_gps = 1;
			if (busca2()){			// esto me saca el valor de tiempo en "tiempo", lo volcare a "tiempo ultimo valido". Serial GPS, true cuando hay valor nuevo.
									// NOTA: estamos tomando tiempo SIEMPRE QUE TENGAMOS UNO, sea valido o no (con 1 satelite ya tenemos tiempo)
				memcpy(&tiempo_ultimo_valido, &tiempo, sizeof(tiempo));	// solo vuelco el tiempo de GPS al "ultimo valido" cuando tenga un tiempo GPS bueno
				tiempo_referencia = RTCDRV_GetWallClockTicks32()/4;		// cojo el nuevo tiempo de referencia (ultimo valor de tiempo del gps)

				if (validez == 0x41){
					Correct_GPS1 = true;											//Feb23 Update del kalman
					PDOP_u = atof(PDOP)*100;										//Feb23 PDOP*100
				}
				  //////////// Algoritmo selector GNSS con PDOP ////////////
				  if(PDOP_u>0 && PDOP_u<PDOP_best && mode_selector_GNSS>0){ 		//Feb23 selector del mejor durante la busqueda. Se hace cuando hay nuevo valor
					  PDOP_best=PDOP_u;
					  best_mode_selector_GNSS=mode_selector_GNSS;
				  }

				  if(PDOP_u>0 && PDOP_u<PDOP_lim){cont_bad_PDOP=0;}					//Feb23 si valor de PDOP malo n veces supera PDOP_lim, se busca nuevo mejor GNSS
				  if((PDOP_u==0 || PDOP_u>PDOP_lim) && mode_selector_GNSS==0){		//Si se supera y se esta en modo normal de funcionamiento (sin buscar nueva se�al GNSS)
					  cont_bad_PDOP++;
					  if(cont_bad_PDOP>cont_bad_PDOP_lim){
						  cont_bad_PDOP=0;
						  mode_selector_GNSS=mode_selector_GNSS_min;
					  }
				  }
				  //////////////////////////////////////////////////////////

			}
			//emberAfCorePrint("\nValid? = %s\n",validez== 0x41 ? "YES" : "NO");

			////////////     Algoritmo selector GNSS     ////////////
			if(mode_selector_GNSS>0){												//Feb23. Se hace por numero de ciclos y no por numero de se�ales nuevas por si algun modo no recibe se�al
				if (cont_selector_GNSS==0){
					set_GNSS_mode(mode_selector_GNSS);
					GNSS_mode=mode_selector_GNSS;
				}
				cont_selector_GNSS++;
				if (cont_selector_GNSS>cont_selector_cicles){
					cont_selector_GNSS=0;
					mode_selector_GNSS++;
					if(mode_selector_GNSS>n_modes) {
						mode_selector_GNSS=0;
						set_GNSS_mode(best_mode_selector_GNSS);
						GNSS_mode=best_mode_selector_GNSS;
						emberAfCorePrint("Best Mode %u, Best PDOP %u\n", best_mode_selector_GNSS, PDOP_best);
						best_mode_selector_GNSS=7;	//Por defecto 7
						PDOP_best=65535;
					}
				}
			}
			//////////////////////////////////////////////////////////

			if (validez == 0x41){									//0x41='A' //Feb23 Los mensajes de GNSS guardan los valores antiguos hasta que hay uno nuevo, por lo tanto validez no significa que haya un dato nuevo. Dato nuevo es en busca2()=true

				  latitud_s = atof(latitud);
				  longitud_s = atof(longitud);
				  altitud_s = atof(altitud);
				  vel_GNSS_u = atof(vel_GNSS)*100/1.94384;			//Feb23 knots to m/s *100

				  if(latitud_s==0.0f || longitud_s==0.0f){			//Ene 23. Comprobacion de valor valido. Se puede incluso restringir la zona de operacion a la del pais.
					  flag_gps0=true; validez=0x41; latitud_s=0.0; longitud_s=0.0; altitud_s=0.0;
				  }else{
					  flag_gps0 = false;
					  cont_invalid_gps = 0;
					  if(cont_valid_gps<10){cont_valid_gps++;}

					  if(flag_first_gps && cont_valid_gps>=10){		//Kalman Initialization
						  flag_first_gps = false;
						  /////////////////////////////////////////////////
						  //              KALMAN INITIALIZATION         //
						  ////////////////////////////////////////////////

						  EKF_Init(&myEKF, latitud_s/100.0f, longitud_s/100.0f, altitud_s);
						  ekf_initialized = true;
					  }

					  if (ekf_initialized && Correct_GPS1) {
						  EKF_Update(&myEKF, latitud_s/100.0f, longitud_s/100.0f, altitud_s);
						  Correct_GPS1 = false;
						  emberAfCorePrint("\nKUpd; lat_s: %ld; lon_s: %ld; alt_s: %ld;  vel_s: %ld", (int32_t)latitud_s, (int32_t)longitud_s, (int32_t)altitud_s, (int32_t)vel_GNSS_u);
					  }


				}
			}else{
				cont_valid_gps = 0;
				if(cont_invalid_gps<200){cont_invalid_gps++;}else{flag_first_gps=true;cont_invalid_gps=0;mode_selector_GNSS=mode_selector_GNSS_min;} //200 gps invalidos seguidos reset kalman
			}
		}
		if (!tengo_gps){			// creo que esto es totalmente innecesario, si no tengo gps siempre el tiempo estara a "000000.000"
			memcpy(&tiempo, "000000.000", sizeof(tiempo));
		}
	}


	if(Ts1_sum*1000 >= sensorReportPeriodMs - t_max_ciclo ){ //Si el tiempo de ejecucion supera el tiempo de muestreo menos un offset opcional para no retrasar el envio se envian datos

			tiempo_incremento = (RTCDRV_GetWallClockTicks32()/4) - tiempo_referencia;			// esto es el tiempo desde mi ultima referencia, en ms. Tiempo interno

			uint16_t parte_horas, parte_minutos;
			uint32_t parte_resto, parte_segundos, parte_milisegundos;
			uint16_t parte_dia, parte_mes, parte_ano;
			uint8_t strhoras[3] = {0}, strminutos[3] = {0}, strresto[5] = {0}, strsegundos[3] = {0}, strmilisegundos[4] = {0}, strdia[3] = {0}, strmes[3] = {0}, strano[3] = {0};

			memcpy(&strhoras, &tiempo_ultimo_valido[0], 2);
			memcpy(&strminutos, &tiempo_ultimo_valido[2], 2);
			memcpy(&strresto[0], &tiempo_ultimo_valido[4] , 2);
			memcpy(&strresto[2], &tiempo_ultimo_valido[7] , 3);
			memcpy(&strdia, &fecha[0], 2);
			memcpy(&strmes, &fecha[2], 2);
			memcpy(&strano, &fecha[4], 2);

			parte_horas = atoi(strhoras);
			parte_minutos = atoi(strminutos);
			parte_resto = atoi(strresto);
			parte_dia = atoi(strdia);
			parte_mes = atoi(strmes);
			parte_ano = atoi(strano);

			parte_resto += tiempo_incremento;				// los segundos + milisegundos ya sumados, en ms


			parte_segundos = parte_resto / 1000;
			parte_milisegundos = parte_resto % 1000;
			while(parte_segundos >= 60){			// si tengo MAS de 60s, me sumo al minuto y ajusto
				parte_segundos -= 60;
				parte_minutos += 1;
			}
			while(parte_minutos >= 60){
				parte_horas += 1;
				parte_minutos -= 60;
			}
			while(parte_horas >= 24){
				parte_dia += 1;
				parte_horas -= 24;
			}
			if((parte_mes == 2)&&(parte_dia > 28)){
				parte_dia -= 28;
				parte_mes += 1;
			}
			if(((parte_mes == 4)||(parte_mes == 6)||(parte_mes == 9)||(parte_mes == 11))&&(parte_dia > 30)){
				parte_dia -= 30;
				parte_mes += 1;
			}
			if(((parte_mes == 1)||(parte_mes == 3)||(parte_mes == 5)||(parte_mes == 7)||(parte_mes == 8)||(parte_mes == 10)||(parte_mes == 12))&&(parte_dia > 31)){
				parte_dia -= 31;
				parte_mes += 1;
			}
			if (parte_mes > 12){
				parte_ano += 1;
				parte_mes -= 12;
			}


			sprintf(strmilisegundos, "%03lu", parte_milisegundos); 	//Mar23 Necesario un byte adicional para '\0' con sprintf
			sprintf(strsegundos, "%02lu", parte_segundos);
			sprintf(strminutos, "%02u", parte_minutos);
			sprintf(strhoras, "%02u", parte_horas);
			sprintf(strdia, "%02u", parte_dia);
			sprintf(strmes, "%02u", parte_mes);
			sprintf(strano, "%02u", parte_ano);

			memcpy(&tiempo_out, "000000.000", 10);
			memcpy(&fecha_out, "000000", 6);
			memcpy(&tiempo_out[0], &strhoras, 2);
			memcpy(&tiempo_out[2], &strminutos, 2);
			memcpy(&tiempo_out[4], &strsegundos, 2);
			memcpy(&tiempo_out[7], &strmilisegundos, 3);
			memcpy(&fecha_out[0], &strano, 2);
			memcpy(&fecha_out[2], &strmes, 2);
			memcpy(&fecha_out[4], &strdia, 2);

			//sprintf(tiempo_out, "%02u%02u%02lu.%03lu", parte_horas, parte_minutos, parte_segundos, parte_milisegundos); //lento
			//sprintf(fecha_out, "%02u%02u%02u", strano, strmes, strdia);

			#if VERBOSEO==1
				//emberAfCorePrint("Parte horas = %lu\n", parte_horas);
				//emberAfCorePrint("Parte minutos = %lu\n", parte_minutos);
				//emberAfCorePrint("Parte dia, numero = %lu\n", parte_dia);
				//emberAfCorePrint("Parte dia, string = %s\n", strdia);
				//emberAfCorePrint("Parte mes, numero = %lu\n", parte_mes);
				//emberAfCorePrint("Parte mes, string = %s\n", strmes);
				//emberAfCorePrint("Parte a�o, numero = %lu\n", parte_ano);
				//emberAfCorePrint("Parte a�o, string = %s\n", strano);
				emberAfCorePrint("Parte dia, numero = %lu\n", parte_dia);
				emberAfCorePrint("Parte dia, string = %s\n", strdia);
				emberAfCorePrint("Parte mes, numero = %lu\n", parte_mes);
				emberAfCorePrint("Parte mes, string = %s\n", strmes);
				emberAfCorePrint("Parte a�o, numero = %lu\n", parte_ano);
				emberAfCorePrint("Parte a�o, string = %s\n", strano);
			#endif


		#endif

			memcpy(paquete, &rhData, sizeof(uint32_t));
			memcpy(paquete+4, &tempData, sizeof(int32_t));
			memcpy(paquete+8, &acelint[0], sizeof(int32_t));
			memcpy(paquete+12, &acelint[1], sizeof(int32_t));
			memcpy(paquete+16, &acelint[2], sizeof(int32_t));
			memcpy(paquete+20, &original_data_rssi_init, sizeof(int8_t));

			memcpy(paquete+26, &latitud_k, sizeof(float));
			memcpy(paquete+35, &longitud_k,sizeof(float));
			memcpy(paquete+44, &altitud_k, sizeof(float));
			memcpy(paquete+50, &vel_k, sizeof(float));

			memcpy(paquete+72, &GNSS_mode, sizeof(uint8_t));
			memcpy(paquete+73, ",", sizeof(char)); // ",," al final del paquete
			memcpy(paquete+74, ",", sizeof(char)); // ",," al final del paquete

		#if TENGO_GPS
					if(tengo_datos_gps){
						// hasta aqui lo que ira en TODOS los paquetes. ahora, si la lectura GPS es valida, iran mas cosas
						// termina en una coma pq es mas facil buscar asi en la recepcion (y es menos guarro)
						memcpy(paquete+21, &validez, sizeof(validez));
						if (validez == 0x41){	//validez = A (0x41)
								memcpy(paquete+22, &latitud_s, sizeof(float));
								memcpy(paquete+30, &norte_sur, sizeof(norte_sur));
								memcpy(paquete+31, &longitud_s,sizeof(float));
								memcpy(paquete+39, &este_oeste, sizeof(este_oeste));
								memcpy(paquete+40, &altitud_s, sizeof(float));
								memcpy(paquete+48, &vel_GNSS_u, sizeof(uint16_t));
								memcpy(paquete+54, &tiempo_out, sizeof(tiempo_out));
								memcpy(paquete+64, &fecha_out, sizeof(fecha_out));
								memcpy(paquete+70, &PDOP_u, sizeof(uint16_t));

							  if(Correct_GPS1){
								int32_t print_lat = latitud_s*10000;
								emberAfCorePrint("{GP:LA=%ld;",print_lat);
								int32_t print_lon = longitud_s*10000;
								emberAfCorePrint("LO=%ld;",print_lon);
								int32_t print_alt = altitud_s*10000;
								emberAfCorePrint("AL=%ld;",print_alt);
								emberAfCorePrint("VL=%lu;",vel_GNSS_u);
								emberAfCorePrint("TS=");
								for (int i=0;i<sizeof(tiempo_out);i++){
									if (tiempo_out[i]!='x'){
										emberAfCorePrint("%c",tiempo_out[i]);
									}
								}
								emberAfCorePrint(";DT=");
								for (int i=0;i<sizeof(fecha_out); i++){
									emberAfCorePrint("%c", fecha_out[i]);
								}
								emberAfCorePrint(";Mod=%u",GNSS_mode);
								emberAfCorePrint(";PDOP=%u",PDOP_u);
								emberAfCorePrint("}\n");
							  }

						} //else {
			#if (RED_USADA==RED_DEMO2)
							memcpy(paquete+28, ",", sizeof(char));
							memcpy(paquete+29, ",", sizeof(char));
							memcpy(paquete+30, ",", sizeof(char));
							memcpy(paquete+31, ",", sizeof(char));
							memcpy(paquete+32, ",", sizeof(char));
							memcpy(paquete+33, ",", sizeof(char));
			#endif
						//}
						} else{		//del if tengo gps
							memcpy(paquete+21, "V", sizeof(char));

			#if (RED_USADA==RED_DEMO2)
							memcpy(paquete+28, ",", sizeof(char));
							memcpy(paquete+29, ",", sizeof(char));
							memcpy(paquete+30, ",", sizeof(char));
							memcpy(paquete+31, ",", sizeof(char));
							memcpy(paquete+32, ",", sizeof(char));
							memcpy(paquete+33, ",", sizeof(char));
			#endif
						}
			#endif
				#if RED_USADA == RED_DEMO2
						uint8_t despl_gyro = ((tengo_gps==1)&&(validez == 0x41)) ? (72-33) : 0;
						memcpy(paquete+34 + despl_gyro, &gyroint[0], sizeof(gyroint[0]));
						memcpy(paquete+38 + despl_gyro, ",", sizeof(char));
						memcpy(paquete+39 + despl_gyro, &gyroint[1], sizeof(gyroint[1]));
						memcpy(paquete+43 + despl_gyro, ",", sizeof(char));
						memcpy(paquete+44 + despl_gyro, &gyroint[2], sizeof(gyroint[2]));
						memcpy(paquete+48 + despl_gyro, ",", sizeof(char));

				#endif

		origen = mi_nodeID;		// origen es la var. global que ira dentro del payload, no es lo mismo que el "origen" del envio nodo-a-nodo
		rangorigen = mi_rango;
		destino = 0x0000;
		emberAfCorePrint("Sending from %2x to %2x\n", origen, nodo_padre.shortAddress);		//Julio 22, se imprimia el destino pero siempre es 0000, quiero ver a que padre manda
		emberAfCorePrint("My range %2x\n", mi_rango);
		tipo = PQT_DATOS;		// esto es una chapuza enorme
		manda_datos(mi_nodeID, nodo_padre.shortAddress);	//los paquetes de datos SIEMPRE se envian al nodo padre (con destino el coordinador 0x0000)
	}
}
else{	//Jul22 Si se ha borrado el padre y mi_rango es 0xffff, se deja de enviar y se aumenta la cuenta para volver a intentar reconectarse
	cuentaDC ++;
	emberAfCorePrintln("State reset, disconnection count = %lu / %lu", cuentaDC, LIMITE_DC);
	#if NODO_QUE_ENVIA
		  if(cuentaDC >= LIMITE_DC){
			  emberAfCorePrint("\n Deleting parent node, reconnecting to network\n");
			  //arrancar_red();
			  borrar_padre();
			  manda_request();
			  manda_repair_global();
			  cuentaDC = 0;
			  flag_first_gps=true; cont_valid_gps=0;
		  }
	#endif
}
//if(cuentaDC >= LIMITE_DC){
	//arrancar_red();
//} else {
if(Ts1_sum*1000 >= sensorReportPeriodMs - t_max_ciclo ){ //Resta de tiempo max de ciclo
	Ts1_sum=0;
	if (tipo == PQT_DATOS){
		if(numpaq < 0xffff){
				emberAfCorePrintln("Packets sent: %lu", numpaq);
				numpaq++;				// usamos el numpaq para contar los paquetes enviados //es generado pero no mandado
		} else {
			numpaq = 1;
		}
	}

	//Dic22. Este delay era fijo sin tener en cuenta el tiempo de ejecucion
	//emberEventControlSetDelayMS(reportControl, sensorReportPeriodMs);		// al final de un envio (report) programamos el siguiente en 200ms
	//emberEventControlSetDelayMS(reportControl, 2000);
}
}



void emberAfChildJoinCallback(EmberNodeType nodeType,
                              EmberNodeId nodeId)
{
  emberAfCorePrintln("Node joined with short address 0x%2x", nodeId);
}


/////// MAC mode incoming message handler ////////
void emberAfIncomingMacMessageCallback(EmberIncomingMacMessage *message)
{
/*
#if (RED_USADA!=RED_JAIME)
  uint8_t recepcion[1*sizeof(uint8_t)+7*sizeof(uint16_t) //de las cabeceras
#if !SOLO_CABEZA
  				+ (1*sizeof(uint32_t) + 4*sizeof(int32_t))/sizeof(char) + 53*sizeof(char) + 1*sizeof(int8_t)  // del propio payload
#endif
#if (RED_USADA == RED_DEMO2)
  + 3*sizeof(char) + 3*sizeof(uint32_t)			 //de tres comas y la aceleracion angular (gyroX,Y,Z) para la DEMO2
#endif
  ] = {0};  //89
#else
  uint8_t recepcion[1024];
#endif
 */
uint8_t recepcion[90];	//Feb23 15+75

  //uint8_t i;
  uint16_t rango_entrante;
  uint16_t panID_entrante;
  uint16_t numero_paquete;					// numero_paquete: esto es interno a la funcion de callback, para contar los paq recibidos
  uint16_t secuencia_origen;				// secuencia_origen: interno a la funcion, para llevar el n� secuencia del nodo original

  /*
   * 		AQUI VAMOS A GUARDAR LOS ELEMENTOS DE LA CABECERA, QUE NOS DARAN INFO DEL MENSAJE Y SU ORIGEN/DESTINO
   */

  memcpy(&tipo, message->payload, 1);
  //tipo = message->payload[0];							// bit 0 del payload es el TIPO de mensaje
  memcpy(&rango_entrante, message->payload + 1, 2);				// bit 1 del payload es el RANGO del nodo emisor
  //emberAfCorePrintln("\nRango entrante: %2x", rango_entrante);
  memcpy(&destino, message->payload + 3, 2);						// bit 2-3 del payload es el DESTINO (corto) al que quiere llegar el mensaje entrante
  //emberAfCorePrintln("Destino: %2x", destino);
  memcpy(&panID_entrante, message->payload + 5, 2);				// bit 4-5 del payload es la panID del que ha enviado (para saber si pertenecemos a la misma)
  //emberAfCorePrintln("pan ID entrante: %2x", panID_entrante);
  memcpy(&origen, message->payload + 7, 2);						// bit 6-7 del payload es el nodo ORIGEN del envio, NUNCA lo voy a modificar
  memcpy(&numero_paquete, message->payload + 9, 2);						// bit 8-9 del payload es el numero de paquete, para el recuento en en coordinador
  memset(&rangorigen, 0, 2);
  memcpy(&rangorigen, message->payload + 11, 2);				// bit 11-12 del payload es el rango del emisor original del mensaje
  memcpy(&secuencia_origen, message->payload + 13, 2);			// bit 13-14 del payload es el numero de secuencia del emisor original del mensaje
  // emberAfCorePrintln("Nodo de origen %2x con rango %lu ", origen, rangorigen);		// ESTO SOLO PARA PRUEBAS INTERNAS

  memset(&recepcion, 0, sizeof(recepcion));						// primero borro el "mensaje" guardado localmente, sea lo que sea
  memcpy(&recepcion, message->payload, message->length);			// luego copio el payload que me ha entrado en el "mensaje", y sobre eso leere/modificare cosas

//  emberAfCorePrintln("\nMensaje recibido:");
//  for (int i=0;i<sizeof(recepcion);i++){
//	emberAfCorePrint("%x", recepcion[i]);
//  }
  switch(tipo){

	case PQT_DISCOVERY:
	{
		//jul 22 Si uno perdido y reseteado con el mismo rango le manda un discovery, entra y su cuentaDC se pone a 0 por lo que resetea. A�adida condicion rango distinto 0xFFFF
		//emberAfCorePrintln("\nTest Received discovery from: %2x", message->macFrame.srcAddress.addr.shortAddress);
		//si me llega un discovery de algun nodo aguas abajo, lo ignoro
		if((((rango_entrante < nodo_padre.rango)) || ((rango_entrante == nodo_padre.rango) && (message->rssi > nodo_padre.rssi))) //Encontrar padre. Si el rango del que me envia es MENOR al del padre actual (es mejor padre) lo cambiare
			&&  //(nodo_padre.shortAddress != message->macFrame.srcAddress.addr.shortAddress) &&  //jul22 ignorar discoveries del padre
			((mi_panID == panID_entrante)||(mi_panID == 0xFFFF)) && (panID_entrante != 0xFFFF) && (rango_entrante != 0xFFFF)) {
			// esto pierde sentido en la version 1 con solo una PAN, siempre sera igual que la entrante y distinta de 0xffff, condicion innecesaria pero ahi queda

  /*------ RESUMEN: condicion para cambiar de padre es tener rango menor, o mismo rango pero mejor rssi, en la misma panID (o si no tenia inicialmente) ------*/
			nodo_padre.rango = rango_entrante;
			nodo_padre.shortAddress = message->macFrame.srcAddress.addr.shortAddress;
			nodo_padre.rssi = message->rssi;		//julio 22 antes no se actualizaba
			emberAfCorePrintln("\nAccepted discovery from: %2x", nodo_padre.shortAddress);
			mi_rango = rango_entrante + SALTO_RANGO;

  /*----- hasta aqui el cambio de padre, aqui el cambio de PAN si es necesario (en version 1 simplemente no aplica) -----*/
			if(mi_panID == 0xFFFF){
				parameters.panId = panID_entrante;					// aqui es donde un nodo hijo adquiere su panID, para siempre
				mi_panID = panID_entrante;
				emLocalPanId = panID_entrante;						// esto parece/deberia ser la clave
				emberAfCorePrint("new  pan ID (parameters.panId) = %2x\n", parameters.panId);
				//emberAfCorePrint("nueva?? pan ID (emLocalPanId) = %2x\n", emLocalPanId);
				//memcpy(&mi_panID, message->payload + 5, 2);			// aqui es donde un nodo hijo adquiere su panID, para siempre
				emberAfCorePrint("new pan ID (mi_panID) = %2x\n", mi_panID);
			}

			//emberResetNetworkState();			// ESTO ES LO QUE HACE QUE CRASHEE AL RECIBIR MUCHOS PQT_DISCOVERY
			emberJoinCommissioned(6, mi_nodeID, &parameters );	//arranque de la red

			cuentaDC = 0;				//al recibir un ack de este mensaje ya deberia ponerse a 0 la cuentaDC
		//manda_confirmacion(mi_nodeID, nodo_padre.shortAddress); //ene23 aunque viene en el protocolo no se utiliza este mensaje
		numpaq = 1;

		if (NODO_QUE_ENVIA){			//jul22 para que no pierdan la ventana al resetearse se pasa durante el discovery
			memcpy(&sensorReportPeriodMs, &recepcion[15], sizeof(sensorReportPeriodMs));
			emberAfCorePrint("Sensor report period ms: %u\n", sensorReportPeriodMs);
		}

		}

		if (nodo_padre.shortAddress == message->macFrame.srcAddress.addr.shortAddress){ //ene23 discoveries del padre propagados para que hijos encuentren mejores opciones
			nodo_padre.rssi = message->rssi;
			manda_discovery(mi_nodeID, 0xFFFF);
		}

	}			// fin de este case
		break;
	case PQT_CONFIRMACION:											// no se para que vale la confirmacion si no almaceno hijos, pero OK
	{
		//emberAfCorePrintln("\nTest Received confirmation from: %2x", message->macFrame.srcAddress.addr.shortAddress);
	}			// fin de este case
		break;
	case PQT_DATOS:													// si es para mi (soy el coord) lo proceso. si no, debo rebotarlo aguas arriba
	{
		//emberAfCorePrintln("\nTest Received data from: %2x", message->macFrame.srcAddress.addr.shortAddress);
		if(rango_entrante <= mi_rango){
			if(mi_rango != 0xFFFF){
			  emberAfCorePrint("Sending repair\n");
			manda_repair(mi_nodeID, message->macFrame.srcAddress.addr.shortAddress);
			// si ha habido error, ni siquiera proceso el paquete. mando a reparar y listos
			}else{
				emberAfCorePrintln("\nData received from: %2x, but mi_rango: 0xffff", message->macFrame.srcAddress.addr.shortAddress);
				//jul22 pendiente mandar repair si no hace que se produzcan muchos discoveries
			}
		} else{ 													// si el rango es correcto respecto al mio, leo/reboto el paquete
			if(destino != mi_nodeID){							// si no es para mi, lo reboto
				/* Pablo, daba error y hacia reseteratse el nodo
				memset(paquete, 0, sizeof(paquete));
				memcpy(paquete, recepcion + 15, sizeof(recepcion) - 15);	// OJO al forzar aqui, esos 15 son el tama�o de la cabeza puesto a mano
				emberAfCorePrint("Relaying data message from %2x to %2x\n", message->macFrame.srcAddress.addr.shortAddress, nodo_padre.shortAddress);
				uint16_t paqtemporal = numpaq;
				uint16_t seqtemporal = mi_numseq;
				numpaq = numero_paquete;			// lo que envio es numpaq, que a su vez es el contador "interno" de envios de origen PROPIO.
				mi_numseq = secuencia_origen;		// hago lo mismo para la secuencia, guardo la mia en la variable auxiliar "___temporal" y reboto la entrante
				manda_datos( mi_nodeID, nodo_padre.shortAddress); 	// PROBANDO
				numpaq = paqtemporal;
				mi_numseq = seqtemporal;
				*/
				//Jul22
				memset(paquete, 0, sizeof(paquete));

				if(mi_rango == rangorigen-1){	//actualizar rssi del paquete si el rango origen es justo uno menos de este
					memcpy(&recepcion[35], &(message->rssi), 1);	//byte de rssi 15+20
				}

				memcpy(&paquete, &recepcion[15], sizeof(paquete));	//jul22 recepcion mas grande que paquete crasheaba
				emberAfCorePrint("Relaying data message from %2x to %2x\n", message->macFrame.srcAddress.addr.shortAddress, nodo_padre.shortAddress);
				manda_datos(mi_nodeID, nodo_padre.shortAddress);
				//numpaq y numseq pendiente

			} else {											// si es para mi, lo leo/proceso
				  //emberAfCorePrint("Leemos el mensaje\n");
	//// flagH = 0, flagT = 0, flagA = 0, flagC1 = 0, flagC2 = 0, flagC3 = 0;
	//// uint8_t flagC4 = 0, flagC5 = 0, flagC6 = 0, flagC7 = 0, flagC8 = 0, flagC9 = 0, flagC10 = 0, flagC11 = 0;
	uint32_t humedad = 0;
	int32_t temperatura = 0, acelX = 0, acelY = 0, acelZ = 0;
#if RED_USADA == RED_DEMO2
	int32_t gyroX = 0, gyroY = 0, gyroZ = 0;
	uint8_t flagGyro1 = 0, flagGyro2 = 0, flagGyro3 = 0;
#endif
	uint8_t valid = 0x56, nortesur = 0, esteoeste = 0; //0x56='V'
	float latit=0.0, longit=0.0, altit=0.0;
	float latit_k=0.0, longit_k=0.0, altit_k=0.0;
	float velocidad_k = 0;
	int32_t print_lat=0, print_lon=0, print_alt=0, print_latk=0, print_lonk=0, print_altk=0, print_vel_k=0;
	uint16_t velocidad=0, PDOP_r =0;
	uint8_t GNSS_mode_r = 0;
	uint8_t tiem[10]={0}, date[6]= {0};

//		emberAfCorePrint("\nRecibido datos: ");
//	for (int i=0;i<sizeof(recepcion);i++){
//		emberAfCorePrint("%c", recepcion[i]);
//	}
//		emberAfCorePrint("\n");
		for (int i=0;i<sizeof(tiem);i++){
							if(i==6){tiem[i]='.';} else {tiem[i] = '0';}
							//longit[i] = 'x';
							//if(i<sizeof(altit)){ altit[i] = 'x';	}
							//if(i<sizeof(latit)){ latit[i]='x';		}
							if(i<sizeof(date)){ date[i]='0';		}
						}
#if !SOLO_CABEZA
/*////
	for (int i=0;i<sizeof(recepcion);i++){
					if((recepcion[i]==0x48)&&(flagH==0)){
						flagH = i;
						flagT = i+5;
						flagA = i+10;
						flagC1 = i+15;
						flagC2 = i+20;
						flagC3 = i+25;
						flagC4 = i+27;
						i+=27;
						} else if (sizeof(recepcion)>45){
							if((i>flagC4)&&(flagC4!=0)&&(recepcion[i]==0x2c)&&(flagC5==0)){ //Ene23 0x2c deteccion de comas ','
							flagC5 = i;
						} else if((i>flagC5)&&(flagC5!=0)&&(recepcion[i]==0x2c)&&(flagC6==0)){
							flagC6 = i;
						} else if((i>flagC6)&&(flagC6!=0)&&(recepcion[i]==0x2c)&&(flagC7==0)){
							flagC7 = i;
						} else if((i>flagC7)&&(flagC7!=0)&&(recepcion[i]==0x2c)&&(flagC8==0)){
							flagC8 = i;
						} else if((i>flagC8)&&(flagC8!=0)&&(recepcion[i]==0x2c)&&(flagC9==0)){
							flagC9 = i;
						} else if((i>flagC9)&&(flagC9!=0)&&(recepcion[i]==0x2c)&&(flagC10==0)){
							flagC10 = i;
						}
#if RED_USADA != RED_DEMO2
						else if ((i>flagC10)&&(flagC10!=0)&&(recepcion[i]==0x2c)&&(flagC11==0)){
							flagC11 = i;
						}
#endif
#if RED_USADA == RED_DEMO2
						  else if ((i>flagC10)&&(flagC10!=0)&&(recepcion[i]==0x2c)&&(flagGyro1==0)) {
							flagGyro1 = i;
						} else if ((i>flagGyro1)&&(flagGyro1!=0)&&(recepcion[i]==0x2c)&&(flagGyro2==0)) {
							flagGyro2 = i;
						} else if ((i>flagGyro2)&&(flagGyro2!=0)&&(recepcion[i]==0x2c)&&(flagGyro3==0)) {
							flagGyro3 = i;
						}
#endif
					} else{
					GPIO_PinOutSet(gpioPortA,5);
					}
				}


				for (int i = flagH+1;i<sizeof(recepcion);i++){
			//		emberAfCorePrint("%x",recepcion[i]);
					if (i < flagT){
						int desp = i - flagH - 1;
						humedad = humedad | recepcion[i]<<8*desp;
					}else if ((i>flagT)&&(i<flagA)){//(i<flagA){
						int desp = i - flagT - 1;
						temperatura = temperatura | recepcion[i]<<8*desp;
					}else if((i>flagA)&&(i<flagC1)){//(i<flagC1){
						int desp = i - flagA - 1;
						acelX = acelX | recepcion[i]<<8*desp;
					}else if((i>flagC1)&&(i<flagC2)){//i<flagC2){
						int desp = i - flagC1 - 1;
						acelY = acelY | recepcion[i]<<8*desp;
					}else if ((i>flagC2)&&(i<flagC3)){//(i<flagC3){
						int desp = i - flagC2 - 1;
						acelZ = acelZ | recepcion[i]<<8*desp;
					}else if ((i>flagC3)&&(i<flagC4)){//(i<flagC4){
						valid = recepcion[i];
						if ((valid!=0x41)&&(valid!=0x56)){
							valid = 0x56;
						}
					}else if ((i>flagC4)&&(i<flagC5)){//(i<flagC5){
						int desp = i - flagC4 - 1;
						latit[desp] = recepcion[i];
					}else if ((i>flagC5)&&(i<flagC6)){//(i<flagC6){
						nortesur = recepcion[i];
					}else if ((i>flagC6)&&(i<flagC7)){//(i<flagC7){
						int desp = i - flagC6 - 1;
						longit[desp] = recepcion[i];
					}else if ((i>flagC7)&&(i<flagC8)){//(i<flagC8){
						esteoeste = recepcion[i];
					}else if ((i>flagC8)&&(i<flagC9)){//(i<flagC9){
						int desp = i - flagC8 - 1;
						altit[desp] = recepcion[i];
					}else if ((i>flagC9)&&(i<flagC10)){//(i<flagC10){
						int desp = i - flagC9 - 1;
						tiem[desp] = recepcion[i];
					}
#if RED_USADA != RED_DEMO2
					else if ((i>flagC10)&&(i<flagC11)){
						int desp = i - flagC10 - 1;
						date[desp] = recepcion[i];
					}
					if (strncmp(date, "Xinval", 6) == 0){			// fix para errores de trama al saltarse algun campo
						memcpy(&date, "000000", 6);
					}
#endif
#if RED_USADA == RED_DEMO2
					else if ((i>flagC10)&&(i<flagGyro1)){
						int desp = i - flagC10 - 1;
						gyroX = gyroX | recepcion[i]<<8*desp;
					} else if ((i>flagGyro1)&&(i<flagGyro2)){
						int desp = i - flagGyro1 - 1;
						gyroY = gyroY | recepcion[i]<<8*desp;
					} else if ((i>flagGyro2)&&(i<flagGyro3)){
						int desp = i - flagGyro2 - 1;
						gyroZ = gyroZ | recepcion[i]<<8*desp;
					}
#endif
				}


//// */
		//Ene23 Buscar por comas tiene sentido cuando se reconstruye un paquete que se lee de un serial o algun sitio variable. Si se lee un paquete que se ha construido
		// asignando directamente los valores en las posiciones de memoria, con confirmar la longitud del paquete deberia ser suficiente.
		if(recepcion[88] == 0x2c && recepcion[89] == 0x2c){ 	//Final del paquete acaba en ',,'
			memcpy(&humedad, &recepcion[15], 4);
			memcpy(&temperatura, &recepcion[19], 4);
			memcpy(&acelX, &recepcion[23], 4);
			memcpy(&acelY, &recepcion[27], 4);
			memcpy(&acelZ, &recepcion[31], 4);
			memcpy(&valid, &recepcion[36], 1);
			memcpy(&latit, &recepcion[37], 4);
			memcpy(&latit_k, &recepcion[41], 4);
			memcpy(&nortesur, &recepcion[45], 1);
			memcpy(&longit, &recepcion[46], 4);
			memcpy(&longit_k, &recepcion[50], 4);
			memcpy(&esteoeste, &recepcion[54], 1);
			memcpy(&altit, &recepcion[55], 4);
			memcpy(&altit_k, &recepcion[59], 4);
			memcpy(&velocidad, &recepcion[63], 2);
			memcpy(&velocidad_k, &recepcion[65], 4);
			memcpy(&tiem, &recepcion[69], 10);
			memcpy(&date, &recepcion[79], 6);
			memcpy(&PDOP_r, &recepcion[85], 2);
			memcpy(&GNSS_mode_r, &recepcion[87], 1);
		}


#endif		// del !SOLO_CABEZA
//		emberAfCorePrint("\nH %u, T %u, A %u, C1%u, C2%u, C3%u, C4%u, C5%u, C6%u, C7%u, C8%u, C9%u, ",flagH,flagT,flagA,flagC1,flagC2,flagC3,flagC4,flagC5,flagC6,flagC7,flagC8,flagC9);
//		emberAfCorePrint("\n"); */

/************				AQUI PRINTEAMOS LA SALIDA				****************/
#if !SOLO_CABEZA
#if (RED_USADA == RED_DEMO2)
				    emberAfCorePrint("Packet from: 90FD.9FFF.FE19.%2x\n",origen);	//rhData/1000, (rhData-1000*(rhData/1000))
					if (EnablePrint_H!=0){emberAfCorePrint("Relative humidity = %lu.%lu (%%)\n",humedad/1000, (humedad-1000*(humedad/1000)));}
					if (EnablePrint_T!=0){emberAfCorePrint("Temperature = %ld.%ld (�C)\n",temperatura/1000, (temperatura-1000*(temperatura/1000)));}
					if (EnablePrint_A!=0){emberAfCorePrint("Linear acceleration (x,y,z) = %ld, %ld, %ld (g x10-3)\n",acelX,acelY,acelZ);}
					if (EnablePrint_G!=0){emberAfCorePrint("Angular acceleration (x,y,z) = %ld, %ld, %ld (deg/s)\n",gyroX,gyroY,gyroZ);}
					if (EnablePrint_GPS!=0){
							if (valid == 0x41){	emberAfCorePrint("GPS = %s", valid==0x41?"Valid":"Invalid/None"); }
					}
#else
/*				    emberAfCorePrint("Paquete de: 90FD.9FFF.FE19.%2x\n",origen);
					emberAfCorePrint("H = %lu\n",humedad);
					emberAfCorePrint("T = %ld\n",temperatura);
					emberAfCorePrint("A = %ld, %ld, %ld\n",acelX,acelY,acelZ);
					emberAfCorePrint("G = 0, 0, 0\n");
					emberAfCorePrint("GPS = %c",valid);
*/
					/***** Trama InSecTT: Se quitan los \n para que quede todo en una linea ******************************************/
					emberAfCorePrint("#Paquete de: 90FD.9FFF.FE19.%2x;",origen);
					//emberAfCorePrint("H = %lu;",humedad);			//Feb23 se envia por si en algun momento se quiere mandar, pero no se imprime porque ahora esta a 0 siempre
					//emberAfCorePrint("T = %ld;",temperatura);		//Feb23 se envia por si en algun momento se quiere mandar, pero no se imprime porque ahora esta a 0 siempre
					emberAfCorePrint("A = %ld, %ld, %ld;",acelX,acelY,acelZ);
					//emberAfCorePrint("G = 0, 0, 0;");	//Ene23
					//GPS Kalman
					emberAfCorePrint("GPS_k = ");
					print_latk = latit_k*10000;
					emberAfCorePrint("%ld, ",print_latk);
					print_lonk = longit_k*10000;
					emberAfCorePrint("%ld, ",print_lonk);
					print_altk = altit_k*10000;
					emberAfCorePrint("%ld;",print_altk);
					print_vel_k = velocidad_k*100;
					emberAfCorePrint("Vel_k = %ld;",print_vel_k);

					//GPS serial
					emberAfCorePrint("Mod = %u; ",GNSS_mode_r);
					emberAfCorePrint("GPS = %c;",valid);
#endif
#if (RED_USADA == RED_DEMO2)
					if (EnablePrint_GPS!=0){
						if ((valid==0x41)&&((nortesur!=0)&&(esteoeste!=0))){
												emberAfCorePrint("\nLatitude: ");
												for (int i=0;i<sizeof(latit);i++){
													emberAfCorePrint("%c",latit[i]);
												}
												emberAfCorePrint(" %c",nortesur);
												emberAfCorePrint("\nLongitude: ");
												for (int i=0;i<sizeof(longit);i++){
													emberAfCorePrint("%c",longit[i]);
												}
												emberAfCorePrint(" %c",esteoeste);
												emberAfCorePrint("\nAltitude: ");
												for (int i=0;i<sizeof(altit);i++){
													if (altit[i]!='x'){
														emberAfCorePrint("%c",altit[i]);
													}
												}
												emberAfCorePrint("\nGPS Timestamp: ");
												for (int i=0;i<sizeof(tiem);i++){
													if (tiem[i]!='x'){
														emberAfCorePrint("%c",tiem[i]);
													}
												}
												emberAfCorePrint("\n");
						} else {
							emberAfCorePrint("\n");
						}
					} else {
						emberAfCorePrint("\n");
					}
#else
				if (EnablePrint_GPS!=0){
					if ((valid==0x41)&&((nortesur!=0)&&(esteoeste!=0))){
						//emberAfCorePrint(", ");
						print_lat = latit*10000;
						emberAfCorePrint("%ld, ",print_lat);
						emberAfCorePrint("%c, ",nortesur);
						print_lon = longit*10000;
						emberAfCorePrint("%ld, ",print_lon);
						emberAfCorePrint("%c, ",esteoeste);
						print_alt = altit*10000;
						emberAfCorePrint("%ld, ",print_alt);
						for (int i=0;i<sizeof(tiem);i++){
							if (tiem[i]!='x'){
								emberAfCorePrint("%c",tiem[i]);
							}
						}
						emberAfCorePrint(";");//\n");
						emberAfCorePrint("D = ");
						for (int i=0;i<sizeof(date); i++){
							emberAfCorePrint("%c", date[i]);
						}
						emberAfCorePrint(";");
						emberAfCorePrint("Vel = %d;",velocidad);
						emberAfCorePrint("PDOP = %lu;",PDOP_r);
					} else{
						emberAfCorePrint(";");//"\n");
					}
				} else {
					emberAfCorePrint(";");//"\n");
				}
#endif

/* Ene23
#if (RED_USADA !=RED_DEMO2)
		  //emberAfCorePrint("LQI = %d\n", message->lqi);
			emberAfCorePrint("LQI = %d;", message->lqi);
#else
		  //emberAfCorePrint(" (dBm)\n\n");
		  emberAfCorePrint(" (dBm);");
#endif

			emberAfCorePrint("TM = ");
			for (int i=0;i<sizeof(tiem); i++){
				emberAfCorePrint("%c", tiem[i]);
			}
			//emberAfCorePrint("\n");
			emberAfCorePrint(";");
*/

		    //emberAfCorePrint("RSSI = %d\n", message->rssi);			//jul22 comentado
			if(rangorigen == 1){									//aqui entramos solo en el coord
				emberAfCorePrint("RSSI = %d;", message->rssi);		//rssi recibido por el coord
			}else if(rangorigen > 1){
				original_data_rssi = recepcion[35];
				emberAfCorePrint("RSSI = %d;", original_data_rssi);		//rssi del padre original 15+74
			}else{
				emberAfCorePrint("RSSI = %-99;");
			}

			emberAfCorePrint("\n"); // InSecTT: Este si que se queda como el final del mensaje

#else
		    emberAfCorePrint("Paquete de: 90FD.9FFF.FE19.%2x\n",origen);
			emberAfCorePrint("RSSI = %d\n\n", message->rssi);
#endif
/* ----- AQUI HACEMOS EL RECUENTO DE PAQUETES RECIBIDOS ----- */
		  uint8_t banderafea = 0;					// banderafea nos dice si el paquete es de alguien conocido (1) o si es de un nodo nuevo (0)
		  for(int i=0; i < cuentanodos; i++){
			  //emberAfCorePrint("No me fio de esta mierda: origen %2x y nodo en lista %2x\n", origen, listanodos[i].shortAddress);
			  if(listanodos[i].shortAddress == origen){
				  if(listanodos[i].recuento_ultimo == numero_paquete){	// NADA. Si cae aqui quiere decir que el paquete era repetido
					  banderafea = 1;
					  // emberAfCorePrint("Paquete repetido");			// ESTO SOLO PARA PRUEBAS INTERNAS
				  } else {
				  if(listanodos[i].paq_recibidos < 0xffff){
				  listanodos[i].paq_recibidos++;
				  listanodos[i].secuencia = secuencia_origen;
				  listanodos[i].rango = rangorigen;
				  } else {
					  listanodos[i].paq_recibidos = 0;
				  }
				  if(listanodos[i].recuento_ultimo < numero_paquete){
					  listanodos[i].recuento_ultimo = numero_paquete;
				  } else {
					  listanodos[i].recuento_ultimo++;				// OJO PRUEBA
				  }
//				  listanodos[i].recuento_ultimo = numero_paquete;	// OJO ORIGINALMENTE ESTO, PRUEBA CON LO DE ABAJO

//				  if(numero_paquete==1) {						// OJO PRUEBA
//					  listanodos[i].paq_recibidos = 1;			// OJO PRUEBA
//				  }												// OJO PRUEBA
/*		SOLO HABILITAMOS ESTO PARA MEDICIONES INTERNAS, PARA FUNCIONAMIENTO NORMAL NO
				  emberAfCorePrint("Paquetes recibidos del nodo %2x : %lu de %lu enviados\n", listanodos[i].shortAddress, listanodos[i].paq_recibidos, numero_paquete);
				  emberAfCorePrint("%% de recepci�n del nodo %2x : %lu.%lu %%\n\n", listanodos[i].shortAddress, (100*(listanodos[i].paq_recibidos))/numero_paquete,(100*(listanodos[i].paq_recibidos))%numero_paquete);
*/
				  banderafea = 1;
				  break;
				  }
			  }				// Nota: numpaq o numero_paquete empiezan la cuenta en 0, recibidos empieza en 1. Por eso se le resta 1, si no siempre ira 1 por delante
		  }
		  if (!banderafea){
			  listanodos[cuentanodos].shortAddress = origen;
			  listanodos[cuentanodos].paq_recibidos = 1;
			  listanodos[cuentanodos].secuencia = secuencia_origen;
			  listanodos[cuentanodos].rango = rangorigen;
			  if(cuentanodos<20){
				  cuentanodos++;
			  }
		  }
/* ----- AQUI HACEMOS EL RECUENTO DE PAQUETES RECIBIDOS ----- */

																	// tras rebotar/procesar el mensaje, salgo
		}
		}
	}			// fin de este case
	break;
	case PQT_REPAIR_BROADCAST:
	{
		//emberAfCorePrintln("\nTest Received repair broadcast from: %2x", message->macFrame.srcAddress.addr.shortAddress);
		if(message->macFrame.srcAddress.addr.shortAddress == nodo_padre.shortAddress){		// cuando un nodo deba repararse y emita un repair_broadcast, solo sus hijos deberan hacerle caso
			emberAfCorePrint("Received repair (broadcast)\n");
			borrar_padre();
			manda_request();						// insertar aqui funcion para solicitar un discovery
			manda_repair_global();
		}
	}			// fin de este case
	break;
	case PQT_REPAIR_UNICAST:						// si algo fue mal, se le mandara al nodo malo un repair_unicast (que disparara un broadcast desde ese nodo)
	{
		//emberAfCorePrintln("\nTest Received repair unicast from: %2x", message->macFrame.srcAddress.addr.shortAddress);
		if(message->macFrame.srcAddress.addr.shortAddress == nodo_padre.shortAddress){
			emberAfCorePrint("Received repair (unicast)\n");
			borrar_padre();
			manda_request();						// insertar aqui funcion para solicitar un discovery
			manda_repair_global();
		}
	}			// fin de este case
	break;
	case PQT_REQUEST:				// en caso de request (al unirse un nodo nuevo o perder/reiniciar su conexion a la red), hacemos un discovery unicast
	{
		//emberAfCorePrintln("\nTest Received request from: %2x", message->macFrame.srcAddress.addr.shortAddress);
		if ((message->macFrame.srcAddress.addr.shortAddress != nodo_padre.shortAddress)&&(mi_panID != 0xFFFF)&& mi_rango!=0xFFFF && mi_rango<=rango_entrante){	//jul22 no responde si esta perdido
#if NODO_QUE_ENVIA
			emberAfCorePrint("Received request from: %2x\n", message->macFrame.srcAddress.addr.shortAddress);
#endif
			manda_discovery(mi_nodeID, message->macFrame.srcAddress.addr.shortAddress);
		}
	}
	break;
	case PQT_SIMPLE:
	{
		if(destino != mi_nodeID){							// si no es para mi, lo reboto
			memset(paquete, 0, sizeof(paquete));
			memcpy(paquete, recepcion + 15, sizeof(recepcion) - 15);	// OJO al forzar aqui
			manda_simple(nodo_padre.shortAddress);
		} else {
		emberAfCorePrint("\nReceived simple message: ");
		for(int i=0; i<message->length-15; i++){
			emberAfCorePrint("%x",recepcion[i+15]);			// TODO: BUSCARAQUI
		}
		emberAfCorePrint("\n");
		}
	}
	break;

	case PQT_CONFIG:{		//Jul22 //Nodos reciben paquete de conf de ventana o autopos. Coord recibe ventana por serial
		//for(int i=0; i < sizeof(recepcion); i++){emberAfCorePrint("%x", recepcion[i]);}	//lectura paquete

		if(recepcion[15] == 1){			//window
			if (NODO_QUE_ENVIA==1){																	//Asegurar. El coordinador no debe procesarlo
				if(message->macFrame.srcAddress.addr.shortAddress == nodo_padre.shortAddress){		//asegurarse cuando solo quiero recibir del padre
					nodo_padre.rssi = message->rssi;//Actualizar al recibir un mensaje
					memcpy(&sensorReportPeriodMs, &recepcion[16], sizeof(sensorReportPeriodMs));
					emberAfCorePrint("Sensor report period ms: %u", sensorReportPeriodMs);
					flagConfig = 1;
					manda_config(mi_nodeID, 0xFFFF);
					flagConfig = 0;		//asegurar
				}
			}
		}
		else if(recepcion[15] == 2 || recepcion[15] == 3){	//autopos, se recibe de un broadcast. Si viene de hijo se lee o se propaga hasta el coord, si viene de padre se considera peticion para mandar los datos del nodo
			if(message->macFrame.srcAddress.addr.shortAddress == nodo_padre.shortAddress){	//si viene del padre, propago en broadcast
				nodo_padre.rssi = message->rssi;//Actualizar al recibir un mensaje
				if (recepcion[15] == 2) {
					flagConfig = 2;
					manda_config(mi_nodeID, 0xFFFF);
					flagConfig = 0;
				}
			}
			else if(rango_entrante!=0xFFFF && rango_entrante>mi_rango){						// si es de rango inferior y no eres el coordinador rebotas. el coord lo lee
				conf_padre = 0xfffe; 		//valor por defecto
				if (NODO_QUE_ENVIA==1){		//cada nodo no conoce sus hijos por lo que se rebotan todos los mensajes de rango inferior. Pendiente, se podria leer el padre del paquete.
					emberAfCorePrint("Relaying config message from %2x to %2x\n", message->macFrame.srcAddress.addr.shortAddress, nodo_padre.shortAddress);
					memset(paquete, 0, sizeof(paquete));
					memcpy(&paquete, &recepcion[15], sizeof(paquete));
					flagConfig = 3;
					manda_config(mi_nodeID, nodo_padre.shortAddress);
					flagConfig = 0;
				}
				else{
					memcpy(&conf_orig, &recepcion[16], sizeof(conf_orig));
					memcpy(&conf_rang, &recepcion[18], sizeof(conf_rang));
					memcpy(&conf_padre, &recepcion[20], sizeof(conf_padre));
					memcpy(&conf_rssi, &recepcion[22], sizeof(conf_rssi));
					emberAfCorePrint("\nAuto-pos: Origen = %2x;", conf_orig);
					emberAfCorePrint(" Rango = %lu;", conf_rang);
					emberAfCorePrint(" Padre = %2x;", conf_padre);
					emberAfCorePrint(" RSSI = %d;\n", conf_rssi);
				}
			}
		}
		else if(recepcion[15] == 4){
			if (NODO_QUE_ENVIA==1){																	//Asegurar. El coordinador no debe procesarlo
				if(message->macFrame.srcAddress.addr.shortAddress == nodo_padre.shortAddress){		//asegurarse cuando solo quiero recibir del padre
					memcpy(&flagStandby, &recepcion[16], sizeof(flagStandby));
					emberAfCorePrint("Standby set to: %u", flagStandby);
					flagConfig = 4;
					manda_config(mi_nodeID, 0xFFFF);
					flagConfig = 0;	//asegurar
				}
			}
		}
		else if(recepcion[15] == 5){
			if (NODO_QUE_ENVIA==1){																	//Asegurar. El coordinador no debe procesarlo
				if(message->macFrame.srcAddress.addr.shortAddress == nodo_padre.shortAddress){		//asegurarse cuando solo quiero recibir del padre
					memcpy(&GNSS_mode_check, &recepcion[16], sizeof(uint8_t));
					if(GNSS_mode_check<(n_modes+1)) memcpy(&GNSS_mode, &recepcion[16], sizeof(uint8_t));
					memcpy(&mode_selector_GNSS, &recepcion[17], sizeof(uint8_t));
					memcpy(&cont_selector_cicles, &recepcion[18], sizeof(uint16_t));
					memcpy(&PDOP_lim, &recepcion[20], sizeof(uint16_t));
					memcpy(&cont_bad_PDOP_lim, &recepcion[22], sizeof(uint8_t));
					if(GNSS_mode_check<(n_modes+1)) set_GNSS_mode(GNSS_mode);
					emberAfCorePrint("GNSS mode: %u, selector: %u, cont_selector: %u, PDOP_lim: %u, cont_PDOP_lim: %u\n", GNSS_mode, mode_selector_GNSS, cont_selector_cicles, PDOP_lim, cont_bad_PDOP_lim);
					flagConfig = 5;
					manda_config(mi_nodeID, 0xFFFF);
					flagConfig = 0;	//asegurar
				}
			}
		}

	}
	break;

  }				// cierro el switch-case

}

// MAC mode message sent handler
void emberAfMacMessageSentCallback(EmberStatus status,
                                   EmberOutgoingMacMessage *message)
{
  if ( status != EMBER_SUCCESS ) {
#if NODO_QUE_ENVIA
    emberAfCorePrintln("MAC TX: 0x%x", status);
#endif
   if (status == 0x40){						// un status de 0x40 es el NO_ACK
    GPIO_PinOutSet(gpioPortA,5);
    GPIO_PinOutSet(gpioPortA,6);
#if NODO_QUE_ENVIA
    cuentaDC ++;
    emberAfCorePrintln("NO_ACK error, disconnection count = %lu / %lu", cuentaDC, LIMITE_DC);
#endif
   }
  } else {						// si el status del envio es SUCCESS
	  //GPIO_PinOutClear(gpioPortA,5);
	  //GPIO_PinOutSet(gpioPortA,6);
#if NODO_QUE_ENVIA
	  memcpy(&rango_ack, message->payload + 1, 2);	//rango del mensaje en este ack de que un mesaje se entrega con exito
	  //emberAfCorePrintln("Test rango entrante ack %2x", rango_ack);
	  if(rango_ack != 0xFFFF){		//evitar que se resetee la cuenta por un mensaje de un nodo perdido.
		  cuentaDC = 0;
	  }
#endif
  }
#if NODO_QUE_ENVIA		// no quiero que el coordinador me ande printeando cada envio que haga, sus print deberian ser minimos fuera del mensaje de trama
#if VERBOSEO
  	  emberAfCorePrint("\n");
  for (int i=0;i<message->length;i++){
      	emberAfCorePrint("%x",message->payload[i]);
      }
      emberAfCorePrint("\n");
#endif
#endif
#if NODO_QUE_ENVIA
      if(cuentaDC >= LIMITE_DC){		// PROBANDO
    	  emberAfCorePrint("\n Deleting parent node, reconnecting to network\n");
    	  //arrancar_red();
    	  borrar_padre();
    	  manda_request();
    	  manda_repair_global();
    	  cuentaDC = 0;
  }
#endif
}

// This callback is called when the application starts and can be used to
// perform any additional initialization required at system startup.
void emberAfMainInitCallback(void)
{
  emberAfCorePrintln("Powered UP");
  emberAfCorePrintln("\n%p>", EMBER_AF_DEVICE_NAME);

  emberNetworkInit();
}

// This callback is called in each iteration of the main application loop and
// can be used to perform periodic functions.
void emberAfMainTickCallback(void)
{
  #ifndef UNIX_HOST
  if (emberStackIsUp()) {
    halSetLed(NETWORK_UP_LED);
  } else {
    halClearLed(NETWORK_UP_LED);
  }
  #endif
}

void emberAfStackStatusCallback(EmberStatus status)
{
  switch ( status ) {
    case EMBER_NETWORK_UP:
      emberAfCorePrintln("Network up");
      break;
    case EMBER_NETWORK_DOWN:
      emberAfCorePrintln("Network down");
      break;
    default:
      emberAfCorePrintln("Stack status: 0x%x", status);
      break;
  }
}

void emberAfIncomingBeaconExtendedCallback(EmberPanId panId,
                                           EmberMacAddress *source,
                                           int8_t rssi,
                                           bool permitJoining,
                                           uint8_t beaconFieldsLength,
                                           uint8_t *beaconFields,
                                           uint8_t beaconPayloadLength,
                                           uint8_t *beaconPayload)
{
  emberAfCorePrint("BEACON: panId 0x%2X source ", panId);
  if (source->mode == EMBER_MAC_ADDRESS_MODE_SHORT) {
    emberAfCorePrint("0x%2X", source->addr.shortAddress);
  } else if (source->mode == EMBER_MAC_ADDRESS_MODE_LONG) {
    emberAfPrintBigEndianEui64(source->addr.longAddress);
  } else {
    emberAfCorePrint("none");
  }

  emberAfCorePrint(" payload {");
  emberAfCorePrintBuffer(beaconPayload, beaconPayloadLength, false);
  emberAfCorePrintln("}");
}

void emberAfEnergyScanCompleteCallback(int8_t mean,
                                       int8_t min,
                                       int8_t max,
                                       uint16_t variance)
{
  emberAfCorePrintln("Energy scan complete, mean=%d min=%d max=%d var=%d",
                     mean, min, max, variance);
}

void emberAfActiveScanCompleteCallback(void)
{
  emberAfCorePrintln("Active scan complete");
}

#if defined(EMBER_AF_PLUGIN_MICRIUM_RTOS) && defined(EMBER_AF_PLUGIN_MICRIUM_RTOS_APP_TASK1)

// Simple application task that prints something every second.

void emberAfPluginMicriumRtosAppTask1InitCallback(void)
{
  emberAfCorePrintln("app task init");
}

#include <kernel/include/os.h>
#define TICK_INTERVAL_MS 1000

void emberAfPluginMicriumRtosAppTask1MainLoopCallback(void *p_arg)
{
  RTOS_ERR err;
  OS_TICK yieldTimeTicks = (OSCfg_TickRate_Hz * TICK_INTERVAL_MS) / 1000;

  while (true) {
    emberAfCorePrintln("app task tick");

    OSTimeDly(yieldTimeTicks, OS_OPT_TIME_DLY, &err);
  }
}

#endif // EMBER_AF_PLUGIN_MICRIUM_RTOS && EMBER_AF_PLUGIN_MICRIUM_RTOS_APP_TASK1

// ------------------------ CLI commands ---------------------------------------
void setSecurityKeyCommand(void)
{
  EmberKeyData key;
  emberCopyKeyArgument(0, &key);

  if (emberSetSecurityKey(&key) == EMBER_SUCCESS) {
    uint8_t i;

    emberAfCorePrint("Security key set {");
    for (i = 0; i < EMBER_ENCRYPTION_KEY_SIZE; i++) {
      emberAfCorePrint("%x", key.contents[i]);
    }
    emberAfCorePrintln("}");
  } else {
    emberAfCorePrintln("Security key set failed");
  }
}

void setBeaconPayloadCommand(void)
{
  EmberStatus status;
  uint8_t length;
  uint8_t *contents = emberStringCommandArgument(0, &length);

  status = emberSetApplicationBeaconPayload(length, contents);

  emberAfCorePrint("Set beacon payload: {");
  emberAfCorePrintBuffer(contents, length, false);
  emberAfCorePrintln("}: status=0x%x", status);
}

void joinCommissionedCommand(void)
{
  EmberNetworkParameters parameters;
  EmberStatus status;
  uint8_t nodeType = emberUnsignedCommandArgument(0);
  EmberNodeId nodeId = emberUnsignedCommandArgument(1);
  parameters.panId = emberUnsignedCommandArgument(2);
  parameters.radioTxPower = emberSignedCommandArgument(3);
  parameters.radioChannel = emberUnsignedCommandArgument(4);

  status = emberJoinCommissioned(nodeType, nodeId, &parameters);

  if (status == EMBER_SUCCESS) {
    emberAfCorePrintln("Node parameters commissioned");
  } else {
    emberAfCorePrintln("Commissioning failed, status 0x%X", status);
  }
}

void joinNetworkCommand(void)
{
  EmberNetworkParameters parameters;
  EmberStatus status;
  uint8_t nodeType = emberUnsignedCommandArgument(0);
  parameters.panId = emberUnsignedCommandArgument(1);
  parameters.radioTxPower = emberSignedCommandArgument(2);
  parameters.radioChannel = emberUnsignedCommandArgument(3);

  status = emberJoinNetwork(nodeType, &parameters);

  if (status == EMBER_SUCCESS) {
    emberAfCorePrintln("Started the joining process");
  } else {
    emberAfCorePrintln("Join network failed, status 0x%X", status);
  }
}

void setPermitJoinCommand(void)
{
  uint8_t duration = emberUnsignedCommandArgument(0);
  EmberStatus status = emberPermitJoining(duration);

  if (status == EMBER_SUCCESS) {
    emberAfCorePrintln("Permit join set 0x%X", duration);
  } else {
    emberAfCorePrintln("Permit join failed");
  }
}

void setAllocateAddressFlagCommand(void)
{
  bool allocateAddress = (emberUnsignedCommandArgument(0) > 0);
  EmberStatus status = emberMacSetAllocateAddressFlag(allocateAddress);

  if (status == EMBER_SUCCESS) {
    emberAfCorePrintln("Allocate address flag set %d", allocateAddress);
  } else {
    emberAfCorePrintln("Allocate address flag set failed, 0x%x", status);
  }
}

void setOptionsCommand(void)
{
  txOptions = emberUnsignedCommandArgument(0);
  emberAfCorePrintln("Send options set: 0x%x", txOptions);
}

void pollCommand(void)
{
  EmberStatus status = emberPollForData();

  emberAfCorePrintln("Poll status 0x%x", status);
}

void setPollDestinationCommand(void)
{
  EmberStatus status;
  EmberMacAddress destAddr;

  destAddr.addr.shortAddress = emberUnsignedCommandArgument(0);
  destAddr.mode = EMBER_MAC_ADDRESS_MODE_SHORT;

  if (destAddr.addr.shortAddress == EMBER_NULL_NODE_ID) {
    emberCopyEui64Argument(1, destAddr.addr.longAddress);
    destAddr.mode = EMBER_MAC_ADDRESS_MODE_LONG;
  }

  status = emberSetPollDestinationAddress(&destAddr);

  if (status == EMBER_SUCCESS) {
    emberAfCorePrintln("Poll address set");
  } else {
    emberAfCorePrintln("Poll address set failed, 0x%x", status);
  }
}

/* Display command parameters:
 *
 * 0 - 1 - 2 - 3 -  4
 * H - T - A - G - GPS
 *
 * Enables or disables printing of data related to a given magnitude. Initially, all enabled by default
 *
 * Example: display 1 0 0 1 1      (T, A disabled, rest enabled)
 *
 */
void display_command(void){
	EnablePrint_H = emberUnsignedCommandArgument(0);
	EnablePrint_T = emberUnsignedCommandArgument(1);
	EnablePrint_A = emberUnsignedCommandArgument(2);
	EnablePrint_G = emberUnsignedCommandArgument(3);
	EnablePrint_GPS = emberUnsignedCommandArgument(4);
}
/* Parametros del envio_simple:
 *
 * Direccion de destino (una nodeID, 2bytes, ej 0xc230)
 *
 * Mensaje a enviar (string, ej "hola mundo")
 *
 */

void envio_simple(void){
	    uint8_t *envio = emberStringCommandArgument(1, &longitud_simple);
	    destino = emberUnsignedCommandArgument(0);
  		rangorigen = mi_rango;
	    memset(paquete, 0, sizeof(paquete));
	    memcpy(paquete, envio, longitud_simple);			// envio = el mensaje a enviar. longitud_simple = su longitud.

	    manda_simple(destino);//(nodo_padre.shortAddress);

}


// Params:
// 0: a "nibble mask" indicating which fields are specified, specifically:
//    0x000F - source ID mode (0x00 = none, 0x02 = short, 0x03 = long)
//    0x00F0 - destination ID mode (0x00 = none, 0x02 = short, 0x03 = long)
//    0x0F00 - the source pan ID is specified (0x01) or not (0x00).
//    0xF000 - the destination pan ID is specified (0x01) or not (0x00).
// 1: the source short ID (if specified)
// 2: the source long ID (if specified)
// 3: the destination short ID (if specified)
// 4: the destination long ID (if specified)
// 5: the source PAN ID (if specified)
// 6: the destination PAN ID (if specified)
// 7: MAC payload length
void sendCommand(void)
{
  EmberStatus status;
  EmberMacFrame macFrame;
  uint8_t length;
  uint8_t argLength;
  uint16_t macFrameInfo = emberUnsignedCommandArgument(0);
  EmberNodeId shortSrcId = emberUnsignedCommandArgument(1);
  EmberNodeId shortDestId = emberUnsignedCommandArgument(3);
  EmberPanId srcPanId = emberUnsignedCommandArgument(5);
  EmberPanId dstPanId = emberUnsignedCommandArgument(6);
  uint8_t *message = emberStringCommandArgument(7, &length);

  if ((macFrameInfo & 0x000F) == EMBER_MAC_ADDRESS_MODE_SHORT) {
    macFrame.srcAddress.addr.shortAddress = shortSrcId;
    macFrame.srcAddress.mode = EMBER_MAC_ADDRESS_MODE_SHORT;
  } else if ((macFrameInfo & 0x000F) == EMBER_MAC_ADDRESS_MODE_LONG) {
    emberStringCommandArgument(2, &argLength);
    assert(argLength == EUI64_SIZE);
    emberCopyEui64Argument(2, macFrame.srcAddress.addr.longAddress);
    macFrame.srcAddress.mode = EMBER_MAC_ADDRESS_MODE_LONG;
  } else {
    macFrame.srcAddress.mode = EMBER_MAC_ADDRESS_MODE_NONE;
  }

  if (((macFrameInfo & 0x00F0) >> 4) == EMBER_MAC_ADDRESS_MODE_SHORT) {
    macFrame.dstAddress.addr.shortAddress = shortDestId;
    macFrame.dstAddress.mode = EMBER_MAC_ADDRESS_MODE_SHORT;
  } else if (((macFrameInfo & 0x00F0) >> 4) == EMBER_MAC_ADDRESS_MODE_LONG) {
    emberStringCommandArgument(4, &argLength);
    assert(argLength == EUI64_SIZE);
    emberCopyEui64Argument(4, macFrame.dstAddress.addr.longAddress);
    macFrame.dstAddress.mode = EMBER_MAC_ADDRESS_MODE_LONG;
  } else {
    macFrame.dstAddress.mode = EMBER_MAC_ADDRESS_MODE_NONE;
  }

  if (macFrameInfo & 0x0F00) {
    macFrame.srcPanId = srcPanId;
    macFrame.srcPanIdSpecified = true;
  } else {
    macFrame.srcPanIdSpecified = false;
  }

  if (macFrameInfo & 0xF000) {
    macFrame.dstPanId = dstPanId;
    macFrame.dstPanIdSpecified = true;
  } else {
    macFrame.dstPanIdSpecified = false;
  }

  status = emberMacMessageSend(&macFrame,
                               0x00, // messageTag
                               length,
                               message,
                               txOptions);

  if (status == EMBER_SUCCESS) {
    emberAfCorePrintln("MAC frame submitted");
  } else {
    emberAfCorePrintln("MAC frame submission failed, status=0x%x",
                       status);
  }
}

void infoCommand(void)
{
  emberAfCorePrintln("%p:", EMBER_AF_DEVICE_NAME);
  emberAfCorePrintln("         Network state: 0x%x", emberNetworkState());
  // emberAfCorePrintln("      node type: 0x%x", emberGetNodeType());		// el tipo de nodo me lo puedo fumar
    emberAfCorePrint("         eui64 (complete MAC address):  ");
  emberAfPrintBigEndianEui64(emberGetEui64());
  emberAfCorePrintln("");
  emberAfCorePrintln("         Node id: 0x%2x", mi_nodeID);//emberGetNodeId());
  emberAfCorePrintln("         PAN id: 0x%2x", emberGetPanId());
  //emberAfCorePrintln("         La otra pan id: 0x%2x", mi_panID);
  emberAfCorePrintln("         Rank: %lu", mi_rango);
  emberAfCorePrintln("         Display status: H = %s, T = %s, A = %s, G = %s, GPS = %s",
		  (EnablePrint_H==0)?"Disabled":"Enabled",
		  (EnablePrint_T==0)?"Disabled":"Enabled",
		  (EnablePrint_A==0)?"Disabled":"Enabled",
		  (EnablePrint_G==0)?"Disabled":"Enabled",
		  (EnablePrint_GPS==0)?"Disabled":"Enabled"
				  );
  emberAfCorePrintln("         Parent node: %2x with rank %lu", nodo_padre.shortAddress, nodo_padre.rango);
  //emberAfCorePrintln("         rango padre: 0x%2x", nodo_padre.rango);
  emberAfCorePrintln("         Neighboring nodes: ");
  for(int i=0;i<20;i++){
	  if(listanodos[i].shortAddress != 0xFFFF){
	  emberAfCorePrintln("         Node number %d, id %2x with rank %lu, received %lu out of %lu: %lu.%lu %%, sequence number %lu", i, listanodos[i].shortAddress, listanodos[i].rango, listanodos[i].paq_recibidos, listanodos[i].recuento_ultimo,
			  (listanodos[i].recuento_ultimo == 0) ? 0 : (100*(listanodos[i].paq_recibidos))/listanodos[i].recuento_ultimo,
			  (listanodos[i].recuento_ultimo == 0) ? 0 : (100*(listanodos[i].paq_recibidos))%listanodos[i].recuento_ultimo,
			   listanodos[i].secuencia);
	  }
  }
  // emberAfCorePrintln("        channel: %d", (uint16_t)emberGetRadioChannel());		// el canal me da igual saberlo, es fijo
  emberAfCorePrintln("         Power: %d", (int16_t)emberGetRadioPower());
//  emberAfCorePrintln("     TX options: MAC acks %s, security %s, priority %s",		// las opciones son siempre fijas (solo ACK)
//                     ((txOptions & EMBER_OPTIONS_ACK_REQUESTED) ? "enabled" : "disabled"),
//                     ((txOptions & EMBER_OPTIONS_SECURITY_ENABLED) ? "enabled" : "disabled"),
//                     ((txOptions & EMBER_OPTIONS_HIGH_PRIORITY) ? "enabled" : "disabled"));
}

void windowCommand(void)
{
	sensorReportPeriodMs = emberUnsignedCommandArgument(0);
	emberAfCorePrintln("Window config: %u", sensorReportPeriodMs);
	flagConfig = 1;
	manda_config(mi_nodeID, 0xFFFF);
}

void autoposCommand(void)
{
	emberAfCorePrintln("Autopos request");
	flagConfig = 2;
	manda_config(mi_nodeID, 0xFFFF);
}

void forceDiscoveryCommand(void)
{
	emberAfCorePrintln("Forced discovery");
	manda_discovery(mi_nodeID, 0xFFFF);
}

void standbyCommand(void)
{
	flagStandby = emberUnsignedCommandArgument(0);
	emberAfCorePrintln("Standby mode");
	flagConfig = 4;
	manda_config(mi_nodeID, 0xFFFF);
}

void GNSSManualCommand(void)
{
	GNSS_mode = emberUnsignedCommandArgument(0);					//modo del GNSS

	if(emberUnsignedCommandArgument(1)<(n_modes+1)){				//valido si entra en el rango del numero de modos configurados en set_GNSS_mode()
		mode_selector_GNSS = emberUnsignedCommandArgument(1);		//en 0 no se busca. Si no empieza a buscar desde el modo seleccionado
	}else{mode_selector_GNSS=0;}

	cont_selector_cicles = emberUnsignedCommandArgument(2);			//Numero de ciclos que se espera para obtener el PDOP mientras se busca nuevo modo
	PDOP_lim = emberUnsignedCommandArgument(3);						//Umbral maximo de PDOP como malo para aumentar el contador
	cont_bad_PDOP_lim = emberUnsignedCommandArgument(4);			//Umbral maximo de contador de PDOP malos por el que se empezaria a buscar otra vez el mejor modo

	emberAfCorePrintln("GNSS mode changed");
	flagConfig = 5;
	manda_config(mi_nodeID, 0xFFFF);
}

void purgeIndirectCommand(void)
{
  EmberStatus status = emberPurgeIndirectMessages();

  if (status == EMBER_SUCCESS) {
    emberAfCorePrintln("Purge indirect success");
  } else {
    emberAfCorePrintln("Purge indirect failed, 0x%x", status);
  }
}

void setFrequencyCommand(void)
{
	sensorReportPeriodMs = emberUnsignedCommandArgument(0);
}

void activeScanCommand(void)
{
  EmberStatus status;
  uint8_t channelToScan = emberUnsignedCommandArgument(0);
  status = emberStartActiveScan(channelToScan);
  emberAfCorePrintln("Start active scanning: channel %d, status=0x%x", channelToScan, status);
}

void energyScanCommand(void)
{
  EmberStatus status;
  uint8_t channelToScan = emberUnsignedCommandArgument(0);
  uint8_t samples = emberUnsignedCommandArgument(1);
  status = emberStartEnergyScan(channelToScan, samples);
  emberAfCorePrintln("Start energy scanning: channel %d, samples %d, status=0x%x", channelToScan, samples, status);
}
