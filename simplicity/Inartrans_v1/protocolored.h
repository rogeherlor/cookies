
#ifndef PROTOCOLORED_H_
#define PROTOCOLORED_H_
#include <string.h>
#include <stdbool.h>
#ifndef PGM
#define PGM     const
#endif
#ifndef PGM_P
#define PGM_P   const char *
#endif
#include "stack/include/ember.h"
#include "em_device.h"
#include "em_chip.h"
#include "bsp.h"
//#include "command-interpreter/command-interpreter2.h"
#include "debug-print/debug-print.h"
//#include "hal/hal.h"
//#include "flex-bookkeeping.h"
//#include "flex-callbacks.h"
//#include "hal/plugin/serial/com.h"
/*************************************************************/
#define RED_PRUEBAS 0x1234		// para pruebas internas
#define RED_SCOTT1 0x1230		// para demos, etc
#define RED_SCOTT2 0x1233		// para demos, etc pero la 2 	//parameters.radioChannel = 1; el resto =0
#define RED_SCOTT3 0x1236

#define RED_JAIME 0x1231		// para testear autenticacion y encriptacion
#define RED_DEMO2 0x1232		// para demo no-SCOTT

#define RED_INSECTT1 0x1240
#define RED_INSECTT2 0x1241
#define RED_INSECTT3 0x1242
#define RED_INSECTT4 0x1243
#define RED_INSECTT5 0x1244

#ifndef NODO_QUE_ENVIA
#define NODO_QUE_ENVIA 	1		//0 Coord, 1 Nodo
#endif

#define SOLO_CABEZA 0

#define RED_USADA RED_INSECTT1					// aqui cambiamos si usamos la panID de pruebas, la de SCOTT1 o cualquier otra
#define VERBOSEO 0
#define SOLO_RELAY 0							// si queremos que el nodo solo rebote mensajes y no mande paquetes propios, poner a 1

#define SALTO_RANGO 1							// aqui fijo cuanto aumento el rango por cada salto. por defecto pongo 1

EmberEventControl reportControl;
EmberNodeId mi_nodeID;
EmberNetworkParameters parameters;

struct nodosvec{
	uint16_t rango;
	int8_t rssi;
	uint16_t paq_recibidos;
	uint16_t recuento_ultimo;
	uint16_t shortAddress;
	uint16_t secuencia;
};
typedef struct nodosvec nodos_vecinos;
nodos_vecinos nodo_padre;
nodos_vecinos listanodos[20];

/* v1.0 IoT Network Protocol		*/
enum {										// los 5 tipos de paquete (de momento)
	PQT_DISCOVERY = 1,
	PQT_CONFIRMACION = 2,					// en la v1.0, el tipo PQT_CONFIRMACION no vale para nada
	PQT_DATOS = 3,
	PQT_REPAIR_BROADCAST = 4,
	PQT_REPAIR_UNICAST = 5,
	PQT_REQUEST = 6,
	PQT_SIMPLE = 7,
	PQT_CONFIG = 8
};

uint16_t mi_rango;
uint16_t mi_panID;
uint16_t destino;
uint16_t origen;
uint16_t numpaq;
uint16_t mi_numseq;
uint16_t rangorigen;
uint16_t cuentaDC;
uint8_t longitud_simple;
/*************************************************************/

void arrancar_red(void);
void manda_discovery(EmberNodeId nodo_origen, EmberNodeId nodo_destino);
void manda_confirmacion(EmberNodeId nodo_origen, EmberNodeId nodo_destino);
void manda_datos(EmberNodeId nodo_origen, EmberNodeId nodo_destino);
void manda_repair_global(void);		// esto siempre va a ser un broadcast, ya filtrare en recepcion
void manda_repair(EmberNodeId nodo_origen, EmberNodeId nodo_destino);
void manda_request(void);			// esto siempre va a ser un broadcast, ya filtrare en recepcion
void manda_simple(EmberNodeId nodo_destino);
void manda_config(EmberNodeId nodo_origen, EmberNodeId nodo_destino);
void envio_pruebas(uint8_t *contenido, EmberNodeId nodo_destino);
void borrar_padre(void);




#endif /* PROTOCOLORED_H_ */
