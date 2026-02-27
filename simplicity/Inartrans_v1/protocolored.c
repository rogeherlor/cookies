#include "protocolored.h"

EmberMessageOptions txOptions = EMBER_OPTIONS_ACK_REQUESTED;
//EmberNetworkParameters parameters;
/*
#if (RED_USADA!=RED_JAIME)
uint8_t paquete[(5*sizeof(char) + 1*sizeof(uint32_t) + 4*sizeof(int32_t))/sizeof(char) + 40*sizeof(char) + 8*sizeof(char) + 1*sizeof(int8_t)
#if (RED_USADA == RED_DEMO2)
+ 3*sizeof(char) + 3*sizeof(uint32_t) + 1*sizeof(int32_t) 			// de tres comas y la aceleracion angular (gyroX,Y,Z) para la DEMO2
#endif
] = {0};		// 73? //74
#else
uint8_t paquete[1024];
#endif
*/
uint8_t paquete[75];	//Feb23

extern uint16_t sensorReportPeriodMs;
extern uint8_t flagConfig;
extern uint8_t flagStandby;
extern uint8_t GNSS_mode;
extern uint8_t mode_selector_GNSS;
extern uint16_t cont_selector_cicles;
extern uint16_t PDOP_lim;
extern uint8_t cont_bad_PDOP_lim;

void arrancar_red(void)
{
//	  EmberStatus status;


	  parameters.radioTxPower = 3;//-15;
#if (RED_USADA == RED_SCOTT2)
	  parameters.radioChannel = 1;				// NOTA: para todas las redes (pruebas, scott1, demo2, redjaime usamos channel 0, para red scott2 usamos channel 1)
#elif (RED_USADA == RED_INSECTT1)
	  parameters.radioChannel = 11;
#elif (RED_USADA == RED_INSECTT2)
	  parameters.radioChannel = 12;
#elif (RED_USADA == RED_INSECTT3)
	  parameters.radioChannel = 13;
#elif (RED_USADA == RED_INSECTT4)
	  parameters.radioChannel = 14;
#elif (RED_USADA == RED_INSECTT5)
	  parameters.radioChannel = 15;
#else
	  parameters.radioChannel = 0;
#endif

emberResetNetworkState();			// reseteamos la red, volviendo al estado NO_NETWORK (por si acaso) //julio22 en flex-callbacks pone que este reset puede hacer que crashee cuando hay discoveries

#if NODO_QUE_ENVIA						// para el nodo sensor

	  numpaq = 1;				// inicialmente ponemos el numpaq a 1 (evito el 0 por si me la lia al dividir, improbable pero meh)
	  mi_numseq = 0;			// inicialmente el mi_numseq arranca a 0, se incrementara con cada request enviado (incluyendo el inicial)
	  parameters.panId = RED_USADA;


/*			ELIMINAMOS ESTO, VERSION 1
	  parameters.panId = 0xffff;		// no se usa para modo MAC, pero te lo piden igualmente. Creo que ni siquiera la llega a utilizar
*/
	  mi_panID = parameters.panId;		// de todos modos, la panID la gestionamos manualmente como parte del payload, mi_panID

	  mi_rango = 0xffff;
	  nodo_padre.rango = 0xffff;
  	  nodo_padre.rssi = -128;
	  nodo_padre.shortAddress = 0xfffe;					//jul22 antes = EMBER_NULL_NODE_ID;  comprobar que ninguna mac acaba en 0xfffe
	  emberAfCorePrint("Starting up network (sensor), sending request\n");
	  origen = 0xFFFE;			//jul22 0xffff por 0xfffe

	  memcpy(&mi_nodeID, emberGetEui64(), 2);				// fijo la nodeID como los 2 ultimos bytes de la MAC del nodo

	  emberJoinCommissioned(6, mi_nodeID, &parameters );		// nodetype, nodeID, &parameters //rearranque de la red

	  manda_request();

	  //emberPermitJoining(255);
	  //emberJoinNetworkExtended(6, mi_nodeID, &parameters);		//nodetype, nodeID, &parameters

#else									// para el nodo receptor (coordinador)

/*				ELIMINAMOS ESTO, VERSION 1
	  memcpy(&parameters.panId, emberGetEui64(), 2);
	  mi_panID = parameters.panId;
*/
	  parameters.panId = RED_USADA;

	  mi_panID = parameters.panId;
for(int i = 0; i<20; i++){
	listanodos[i].shortAddress = 0xFFFF;
	listanodos[i].paq_recibidos = 0;
	listanodos[i].recuento_ultimo = 0;
	listanodos[i].secuencia = 0;			//secuencia por defecto a 0x0000 o a 0xFFFF?? por si posterior comparativa para goteo
	listanodos[i].rango = 0xFFFF;
}
	  mi_rango = 0;
	  mi_nodeID = 0x0000;
	  origen = mi_nodeID;
	  emberJoinCommissioned(6, mi_nodeID, &parameters );		// nodetype, nodeID, &parameters
	  emberPermitJoining(255);								// 255 = permitir para siempre
	  emberAfCorePrint("Starting up network (coordinator), sending discovery\n");
//	  memcpy(&mi_nodeID, emberGetEui64(), 2);
//	  origen = mi_nodeID;

	  manda_discovery(mi_nodeID, 0xFFFF);
#endif
}

/*		 		FORMATO DE TRAMA PARA LOS ENVIOS:
 *
 * 		|__|__ __|__ __|__ __|__ __|__ __|__ __|__ __|________________(...)___________________|
 * 		  0  1 2   3 4   5 6   7 8   9 10 11 12 13 14    15+
 *
 *		0 : tipo de paquete
 *		1-2 : mi_rango, rango del nodo que lo envia
 *		3-4 : destino, NodeId del destinatario del paquete
 *		5-6 : mi_panID, la PAN ID de la red a la que pertenece el nodo que envia
 *		7-8 : origen, NodeId del nodo que envia
 *		9-10 : paqno, el numero de paquete (para el recuento de recibidos/enviados)
 *		11-12 : rango origen, el rango del emisor original
 *		13-14 : mi_numseq, el numero de secuencia del nodo
 *		15+ : el resto del payload, de haberlo (caso paquete tipo 3 - PQT_DATOS)
 *
 */

void enviar(uint8_t tipo, EmberNodeId nodo_origen, EmberNodeId nodo_destino)
{
/*
	uint8_t mensaje[1*sizeof(uint8_t) + 7*sizeof(uint16_t) // de las cabeceras
#if !SOLO_CABEZA
	  				+ (1*sizeof(uint32_t) + 4*sizeof(int32_t))/sizeof(char) + 53*sizeof(char) + 1*sizeof(int8_t) // del propio payload
#endif
#if (RED_USADA == RED_DEMO2)
	+ 3*sizeof(char) + 3*sizeof(uint32_t)			// de tres comas y la aceleracion angular (gyroX,Y,Z) para la DEMO2
#endif
	] = {0}; //89
*/
uint8_t mensaje[90]; 	//Feb23 15+75

	EmberMacFrame macFrame;
	uint8_t length;

	uint8_t cabeza_trama[15];

	macFrame.srcAddress.mode = EMBER_MAC_ADDRESS_MODE_SHORT;			// uso addr corta para el origen y para el destino
	macFrame.dstAddress.mode = EMBER_MAC_ADDRESS_MODE_SHORT;

	macFrame.srcPanIdSpecified = true;									// la panID de origen nos da igual
	macFrame.dstPanIdSpecified = true;									// NECESITAMOS indicar la pan ID siempre, o no funciona.
	macFrame.srcPanId = mi_panID;
	if(tipo == PQT_DISCOVERY){//((tipo == PQT_DISCOVERY)&&(!NODO_QUE_ENVIA)){
		/*
		macFrame.dstPanId = 0xFFFF;//mi_panID;			// Si es a un nodo NUEVO, debe ser a 0xFFFF, si es a un nodo existente, debe ser a mi_panID (la de la red)
		*/
		macFrame.dstPanId = mi_panID;
	} else {
		macFrame.dstPanId = mi_panID;
	}
	macFrame.srcAddress.addr.shortAddress = nodo_origen;						// me dan las dos direcciones desde fuera, al llamar a enviar()
	macFrame.dstAddress.addr.shortAddress = nodo_destino;

	//memset(mensaje, 0, sizeof(mensaje));
	memcpy(cabeza_trama, &tipo, 1);
	memcpy(cabeza_trama + 1, &mi_rango, 2);
	memcpy(cabeza_trama + 3, &destino, 2);		//destino no es lo mismo que nodo_destino, destino es siempre 0x0000
	memcpy(cabeza_trama + 5, &mi_panID, 2);
	memcpy(cabeza_trama + 7, &origen, 2);
	memcpy(cabeza_trama + 9, &numpaq, 2);
	memcpy(cabeza_trama + 11, &rangorigen, 2);
	memcpy(cabeza_trama + 13, &mi_numseq, 2);
	/*------ AQUI VOY A MODIFICAR EL CONTENIDO DE LA TRAMA, Y ENVIAR UN PAYLOAD COMPLETO U OTRA COSA ------*/
#if NODO_QUE_ENVIA
#if VERBOSEO
	emberAfCorePrint("\nCabeza de trama: ");
	for (int i=0;i<sizeof(cabeza_trama);i++){
		emberAfCorePrint("%x ", cabeza_trama[i]);
	}
	emberAfCorePrintln("\n");
#endif
#endif

switch (tipo){													//de if() a switch() julio 2022
	case PQT_DISCOVERY:
		memset(mensaje, 0, sizeof(mensaje));
		memcpy(mensaje, cabeza_trama, sizeof(cabeza_trama));	// aunque modifique el tama�o de la cabeza de trama, esto funcionara igual

		length = sizeof(cabeza_trama)
		#if !SOLO_CABEZA
			+ sizeof(sensorReportPeriodMs);		//se suma a length
			memcpy(mensaje + sizeof(cabeza_trama), &sensorReportPeriodMs, sizeof(sensorReportPeriodMs));
		#else
			;
		#endif
		break;
	case PQT_DATOS:												// solo los paquetes de DATOS necesitan llevar el payload completo
		memset(mensaje, 0, sizeof(mensaje));
		memcpy(mensaje, cabeza_trama, sizeof(cabeza_trama));	// aunque modifique el tama�o de la cabeza de trama, esto funcionara igual

		length = sizeof(cabeza_trama)
		#if !SOLO_CABEZA
					+ sizeof(paquete);	//85
			memcpy(mensaje + sizeof(cabeza_trama), paquete, sizeof(paquete));
		#else
			;
		#endif
			//emberAfCorePrint("\nTest mensaje: ");
			//for(int i=0; i < sizeof(mensaje); i++){emberAfCorePrint("%x", mensaje[i]);}	//lectura paquete
		break;
	case PQT_SIMPLE:											// el resto de tipos de envio con llevar la trama inicial con tipo, origen/destino, etc ya vale
		memcpy(mensaje, cabeza_trama, sizeof(cabeza_trama));
		memcpy(mensaje + sizeof(cabeza_trama), paquete, longitud_simple);
		length = sizeof(cabeza_trama) + longitud_simple;					// lo que sea que mida eso, es variable
		break;
	case PQT_CONFIG:
		memset(mensaje, 0, sizeof(mensaje));
		memcpy(mensaje, cabeza_trama, sizeof(cabeza_trama));	// aunque modifique el tama�o de la cabeza de trama, esto funcionara igual

		length = sizeof(cabeza_trama)
		#if !SOLO_CABEZA
			+ sizeof(flagConfig)+7;		//se suma a length. longitud max posible de los casos de configuracion

			if(flagConfig==1){
				memcpy(mensaje + sizeof(cabeza_trama), &flagConfig, sizeof(flagConfig));
				memcpy(mensaje + sizeof(cabeza_trama)+ sizeof(flagConfig), &sensorReportPeriodMs, sizeof(sensorReportPeriodMs));
				flagConfig = 0;
			}
			else if(flagConfig==2){	//se envia en broadcast. El padre lee o sigue propagando la info hasta el coord y los hijos que lo reciben lo toman como una peticion para mandar sus datos
				memcpy(mensaje + sizeof(cabeza_trama), &flagConfig, sizeof(flagConfig));
				memcpy(mensaje + sizeof(cabeza_trama)+ sizeof(flagConfig), &nodo_origen, 2);
				memcpy(mensaje + sizeof(cabeza_trama)+ sizeof(flagConfig)+ 2, &mi_rango, 2);
				memcpy(mensaje + sizeof(cabeza_trama)+ sizeof(flagConfig)+ 4, &(nodo_padre.shortAddress), 2);
				memcpy(mensaje + sizeof(cabeza_trama)+ sizeof(flagConfig)+ 6, &(nodo_padre.rssi), 1);	//longitud max config size(flag)+7
				flagConfig = 0;
			}
			else if(flagConfig==3){	//rely config
				memcpy(mensaje + sizeof(cabeza_trama), &paquete, 8);	//comprobar tama�o de paquete y mensaje por overflow de ceros
				memcpy(mensaje + sizeof(cabeza_trama), &flagConfig, sizeof(flagConfig));	//se cambia la cabecera
				flagConfig = 0;
			}
			else if(flagConfig==4){	//standby
				memcpy(mensaje + sizeof(cabeza_trama), &flagConfig, sizeof(flagConfig));
				memcpy(mensaje + sizeof(cabeza_trama)+ sizeof(flagConfig), &flagStandby, sizeof(flagStandby));
				flagConfig = 0;
			}
			else if(flagConfig==5){	//cambio modo GNSS, se envia en broadcast
				memcpy(mensaje + sizeof(cabeza_trama), &flagConfig, sizeof(flagConfig));
				memcpy(mensaje + sizeof(cabeza_trama)+ sizeof(flagConfig), &GNSS_mode, 1);
				memcpy(mensaje + sizeof(cabeza_trama)+ sizeof(flagConfig)+ 1, &mode_selector_GNSS, 1);
				memcpy(mensaje + sizeof(cabeza_trama)+ sizeof(flagConfig)+ 2, &cont_selector_cicles, 2);
				memcpy(mensaje + sizeof(cabeza_trama)+ sizeof(flagConfig)+ 4, &PDOP_lim, 2);
				memcpy(mensaje + sizeof(cabeza_trama)+ sizeof(flagConfig)+ 6, &cont_bad_PDOP_lim, 1);
				flagConfig = 0;
			}

		#else
			;															//se suma a length
		#endif
		break;
	default:
		memcpy(mensaje, cabeza_trama, sizeof(cabeza_trama));
		length = sizeof(cabeza_trama); // 15;
}


	emberMacMessageSend(&macFrame,
            0x00, // messageTag
            length,
            mensaje,
            txOptions);
}

/*------ AQUI ES DONDE FIJO EL ORIGEN/DESTINO DE LOS ENVIOS ------*/

void manda_discovery(EmberNodeId nodo_origen, EmberNodeId nodo_destino)
{
	//emberAfCorePrintln("\nTest sent disc");
	enviar(PQT_DISCOVERY, nodo_origen, nodo_destino);		// igual meter un delay y repetir el envio ??
}
void manda_confirmacion(EmberNodeId nodo_origen, EmberNodeId nodo_destino)
{
	//emberAfCorePrintln("\nTest sent conf");
	enviar(PQT_CONFIRMACION, nodo_origen, nodo_destino);
}
void manda_datos(EmberNodeId nodo_origen, EmberNodeId nodo_destino)
{
	//emberAfCorePrintln("\nTest sent data");
	enviar(PQT_DATOS, nodo_origen, nodo_destino);
}
void manda_repair_global(void)		// esto siempre va a ser un broadcast, ya se filtrara en recepcion
{
	//emberAfCorePrintln("\nTest sent repair glob");
	enviar(PQT_REPAIR_BROADCAST, mi_nodeID, 0xFFFF);
}
void manda_repair(EmberNodeId nodo_origen, EmberNodeId nodo_destino)
{
	//emberAfCorePrintln("\nTest sent repair");
	enviar(PQT_REPAIR_UNICAST, nodo_origen, nodo_destino);
}
void manda_request(void)			// esto siempre va a ser un broadcast, ya se filtrara en recepcion
{
	//emberAfCorePrintln("\nTest sent req");
	  mi_numseq++;
	enviar(PQT_REQUEST, mi_nodeID, 0xFFFF);
}
void manda_simple(EmberNodeId nodo_destino)
{
	enviar(PQT_SIMPLE, mi_nodeID, nodo_destino);
}
void manda_config(EmberNodeId nodo_origen, EmberNodeId nodo_destino)
{
	enviar(PQT_CONFIG, nodo_origen, nodo_destino);
}
void envio_pruebas(uint8_t *contenido, EmberNodeId nodo_destino){				//TODO: funcion de envio
		rangorigen = mi_rango;
		destino = nodo_destino;
    memset(paquete, 0, sizeof(paquete));
    longitud_simple = sizeof(contenido);
    memcpy(paquete, contenido, longitud_simple);			// envio = el mensaje a enviar. longitud_simple = su longitud.
    manda_simple(destino);//(nodo_padre.shortAddress);			// esto cambiaria a manda_simple(destino) si estamos en algun tipo de mesh y no en la red "normal" de arbol
}
void borrar_padre(void)						// "borro" mi rango, el de mi padre y su nodeID, lo unico que quiero conservar es la panID
{
	  mi_rango = 0xffff;
	  nodo_padre.rango = 0xffff;
	  nodo_padre.rssi = -128;
//	  for (int i=0;i<EUI64_SIZE;i++){
//		  nodo_padre.longAddress[i] = 0;
//	  }
	  nodo_padre.shortAddress = 0xfffe;		//jul22 antes = EMBER_NULL_NODE_ID;   comprobar que ninguna mac acaba en 0xfffe
}
