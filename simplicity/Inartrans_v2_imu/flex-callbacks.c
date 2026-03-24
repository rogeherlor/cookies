// MAC Mode Device Sample Application
//
// Copyright 2017 Silicon Laboratories, Inc.                                *80*
#define TENGO_GPS 1				// Habilito o deshabilito directamente el GPS desde macro

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

typedef uint8_t EmberStatus;

#define NETWORK_UP_LED BOARDLED0

extern EmberMessageOptions txOptions;
uint8_t EnablePrint_H = 1;
uint8_t EnablePrint_T = 1;
uint8_t EnablePrint_A = 1;
uint8_t EnablePrint_G = 1;
uint8_t EnablePrint_GPS = 1;
uint8_t flagConfig = 0;

//EmberEventControl reportControl;
#if (RED_USADA!=RED_DEMO2)
uint16_t sensorReportPeriodMs =  125; 				// Periodo de "reportes" AKA envio de sensores, por defecto 250 ms inicialmente
#else
static uint16_t sensorReportPeriodMs =  1000;		// para RED_DEMO2 quiero envios cada 1 segundo
#endif
uint8_t flagStandby = 0;							// Oct22 0 manda datos, 1 no manda. No deshabilita la radio ni su consumo (pendiente)
//extern EmberStatus emberAfPluginWstkSensorsGetSample(uint32_t *rhData, int32_t *tData);

//////////////   GNSS   /////////////////			//Feb23
extern uint8_t PDOP[6];
extern uint8_t vel_GNSS[6];
extern uint8_t cog_GNSS[6];
uint16_t PDOP_u = 0;
uint8_t n_modes = 7;								//n_modes en set_GNSS_mode
uint8_t GNSS_mode = 7;								//modo del GNSS
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
extern uint8_t validez;
extern uint8_t tiempo[10];
extern uint8_t latitud[9];
extern uint8_t longitud[10];
extern uint8_t altitud[8];

uint8_t cont_valid_gps = 0;
uint16_t cont_invalid_gps = 0;
bool flag_first_gps = true;

// ---- IMU 200Hz timer (RTCDRV periodic, 5ms) ----
static RTCDRV_TimerID_t imu_timer_id;
static volatile bool imu_sample_pending = false;

static void imuTimerCallback(RTCDRV_TimerID_t id, void *user)
{
    (void)id; (void)user;
    imu_sample_pending = true;
}

static void IMU_initTimer(void)
{
    RTCDRV_AllocateTimer(&imu_timer_id);
    RTCDRV_StartTimer(imu_timer_id, rtcdrvTimerTypePeriodic, 5, imuTimerCallback, NULL);
}
// -------------------------------------------------

// ---- GNSS epoch detection ----
volatile uint8_t gnss_lines_received = 0;   // incremented by USART1 ISR on each '\n'
#define GNSS_LINES_PER_EPOCH  6             // RMC + GGA + GSA + others; all arrive within 1s at 9600 baud
// ------------------------------

// ---- GPS-synced timestamp ----
// After each valid GPS fix, gps_ms_ref stores GPS time (ms since midnight) and
// rtc_at_gps_fix stores the corresponding RTCDRV tick.  IMU and GNSS both print
// timestamps in this same domain.  Falls back to ms-since-boot until first fix.
static uint32_t gps_ms_ref     = 0;
static uint32_t rtc_at_gps_fix = 0;
static bool     gps_time_valid = false;
// ------------------------------

void reportHandler(void)
{
	if (flagStandby == 1) {
		flag_first_gps = true;
		cont_valid_gps = 0;
		emberEventControlSetDelayMS(reportControl, sensorReportPeriodMs);
		return;
	}

	// Advance GNSS constellation selector counter (~8Hz)
	if (mode_selector_GNSS > 0) {
		if (cont_selector_GNSS == 0) {
			set_GNSS_mode(mode_selector_GNSS);
			GNSS_mode = mode_selector_GNSS;
		}
		cont_selector_GNSS++;
		if (cont_selector_GNSS > cont_selector_cicles) {
			cont_selector_GNSS = 0;
			mode_selector_GNSS++;
			if (mode_selector_GNSS > n_modes) {
				mode_selector_GNSS = 0;
				set_GNSS_mode(best_mode_selector_GNSS);
				GNSS_mode = best_mode_selector_GNSS;
				emberAfCorePrint("\nBest Mode %u, Best PDOP %u\n", best_mode_selector_GNSS, PDOP_best);
				best_mode_selector_GNSS = 7;
				PDOP_best = 65535;
			}
		}
	}

	emberEventControlSetDelayMS(reportControl, sensorReportPeriodMs);
}


void emberAfChildJoinCallback(EmberNodeType nodeType,
                              EmberNodeId nodeId)
{
  emberAfCorePrintln("Node joined with short address 0x%2x", nodeId);
}


void emberAfIncomingMacMessageCallback(EmberIncomingMacMessage *message)
{
  (void)message;  // not used — sensor-only node
}

// MAC mode message sent handler
void emberAfMacMessageSentCallback(EmberStatus status,
                                   EmberOutgoingMacMessage *message)
{
  (void)status; (void)message;  // not used — sensor-only node
}

// This callback is called when the application starts and can be used to
// perform any additional initialization required at system startup.
void emberAfMainInitCallback(void)
{
  emberAfCorePrintln("Powered UP");
  emberAfCorePrintln("\n%p>", EMBER_AF_DEVICE_NAME);

  emberNetworkInit();
  IMU_initTimer();    // start 200Hz IMU sampling timer
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

#if NODO_QUE_ENVIA
  // --- IMU at 200Hz (triggered by RTCDRV periodic timer ISR) ---
  if (imu_sample_pending) {
    imu_sample_pending = false;
    float acelflo[3], gyroflo[3];
    IMU_getAccelerometerData(acelflo);
    IMU_getGyroData(gyroflo);
    uint32_t rtc_now = RTCDRV_GetWallClockTicks32();
    uint32_t ts = gps_time_valid
                  ? (gps_ms_ref + (rtc_now - rtc_at_gps_fix) / 4)
                  : (rtc_now / 4);  // ms since midnight (GPS-synced) or since boot
    emberAfCorePrint("\nIMU t=%lu; A=%ld,%ld,%ld; G=%ld,%ld,%ld",
        ts,
        (int32_t)(acelflo[0] * 1000), (int32_t)(acelflo[1] * 1000), (int32_t)(acelflo[2] * 1000),
        (int32_t)(gyroflo[0] * 100),  (int32_t)(gyroflo[1] * 100),  (int32_t)(gyroflo[2] * 100));
  }

  // --- GNSS: process immediately when a complete epoch has been received ---
  if (gnss_lines_received >= GNSS_LINES_PER_EPOCH) {
    gnss_lines_received = 0;  // byte write is atomic on Cortex-M

    // Detect talker ID ('P'=GPS-only, 'N'=multi-constellation).
    // Scans the buffer until 3 consecutive epochs agree (locked), then stops.
    // Re-triggers automatically when GNSS_mode changes (mode selector switched
    // constellation), since that is the only time the talker can legitimately flip.
    static char    gnss_talker[3]   = "GN";
    static uint8_t talker_confirm   = 0;          // epochs seen with current talker
    static uint8_t talker_last_mode = 0xFF;       // tracks GNSS_mode changes
    #define TALKER_LOCK_COUNT  3

    if (talker_confirm < TALKER_LOCK_COUNT || GNSS_mode != talker_last_mode) {
      if (GNSS_mode != talker_last_mode) {
        talker_confirm   = 0;                     // mode changed — re-detect
        talker_last_mode = GNSS_mode;
      }
      for (uint32_t k = 0; k + 2 < indice; k++) {
        if (mi_buffer[k] == '$' && mi_buffer[k + 1] == 'G') {
          gnss_talker[1] = mi_buffer[k + 2];      // 'P' or 'N'
          talker_confirm++;
          break;
        }
      }
    }

    if (parse_nmea_epoch()) {
      // Parse "HHMMSS.sss" → ms since midnight and anchor to RTCDRV
      uint32_t gps_h   = (uint32_t)((tiempo[0]-'0')*10 + (tiempo[1]-'0'));
      uint32_t gps_min = (uint32_t)((tiempo[2]-'0')*10 + (tiempo[3]-'0'));
      uint32_t gps_s   = (uint32_t)((tiempo[4]-'0')*10 + (tiempo[5]-'0'));
      uint32_t gps_ms  = (uint32_t)((tiempo[7]-'0')*100 + (tiempo[8]-'0')*10 + (tiempo[9]-'0'));
      gps_ms_ref     = gps_h*3600000UL + gps_min*60000UL + gps_s*1000UL + gps_ms;
      rtc_at_gps_fix = RTCDRV_GetWallClockTicks32();
      gps_time_valid  = true;

      // Parse "DDMMYY" → compact integer (e.g., 240326 for 24 March 2026)
      uint32_t gps_date = (uint32_t)(
          ((fecha[0]-'0')*10 + (fecha[1]-'0')) * 10000UL +
          ((fecha[2]-'0')*10 + (fecha[3]-'0')) * 100UL +
          ((fecha[4]-'0')*10 + (fecha[5]-'0')));

      float lat  = atof((char *)latitud);
      float lon  = atof((char *)longitud);
      float alt  = atof((char *)altitud);
      float vel  = atof((char *)vel_GNSS) / 1.94384f;  // knots → m/s
      float cog  = atof((char *)cog_GNSS);
      float pdop = atof((char *)PDOP);
      emberAfCorePrint("\n%s V=%c; T=%lu; D=%lu; Lat=%ld; Lon=%ld; Alt=%ld; Vel=%ld; COG=%ld; PDOP=%ld",
          gnss_talker, validez, gps_ms_ref, gps_date,
          (int32_t)(lat * 10000), (int32_t)(lon * 10000), (int32_t)(alt * 100),
          (int32_t)(vel * 100), (int32_t)(cog * 100), (int32_t)(pdop * 100));

      // PDOP-based GNSS mode selector (runs on each valid epoch)
      if (validez == 0x41) {
        PDOP_u = (uint16_t)(pdop * 100);
        if (PDOP_u > 0 && PDOP_u < PDOP_best && mode_selector_GNSS > 0) {
          PDOP_best = PDOP_u;
          best_mode_selector_GNSS = mode_selector_GNSS;
        }
        if (PDOP_u > 0 && PDOP_u < PDOP_lim) { cont_bad_PDOP = 0; }
        if ((PDOP_u == 0 || PDOP_u > PDOP_lim) && mode_selector_GNSS == 0) {
          cont_bad_PDOP++;
          if (cont_bad_PDOP > cont_bad_PDOP_lim) {
            cont_bad_PDOP = 0;
            mode_selector_GNSS = mode_selector_GNSS_min;
          }
        }
      } else {
        cont_valid_gps = 0;
        if (cont_invalid_gps < 200) { cont_invalid_gps++; }
        else { flag_first_gps = true; cont_invalid_gps = 0; mode_selector_GNSS = mode_selector_GNSS_min; }
      }
    }

    // Reset buffer only if >75% full — lets parse_nmea_epoch() accumulate across epochs
    // when GGA arrives before RMC (non-default NMEA order). parse_nmea_epoch() resets
    // internally on a successful parse, so no explicit reset is needed on success.
    if (indice > (MI_BUFFER_SIZE * 3 / 4)) {
      INTERRUPTS_OFF();
      for (int i = 0; i < MI_BUFFER_SIZE; i++) mi_buffer[i] = '0';
      indice = 0;
      gnss_lines_received = 0;
      INTERRUPTS_ON();
    }
  }
#endif  // NODO_QUE_ENVIA
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
