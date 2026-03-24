################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include wstk-sensors/subdir.mk
-include emlib/subdir.mk
-include drivers/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: Insectt_GPSaidedINS_step_cookie.axf

# Tool invocations
Insectt_GPSaidedINS_step_cookie.axf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GNU ARM C Linker'
	arm-none-eabi-gcc -g3 -gdwarf-2 -mcpu=cortex-m4 -mthumb -T "Insectt_GPSaidedINS_step_cookie.ld" -nostdlib -Xlinker --gc-sections -Xlinker -Map="Insectt_GPSaidedINS_step_cookie.map" -mfpu=fpv4-sp-d16 -mfloat-abi=softfp --specs=nano.specs -o Insectt_GPSaidedINS_step_cookie.axf "./GPSaidedINS_step.o" "./GPSaidedINS_step_data.o" "./GPSaidedINS_step_emxutil.o" "./GPSaidedINS_step_initialize.o" "./GPSaidedINS_step_terminate.o" "./INSFilter.o" "./board_cookie.o" "./flex-callbacks.o" "./gps-uart.o" "./icm20648.o" "./imu.o" "./imu_dcm.o" "./imu_fuse.o" "./imu_math.o" "./mrdivide_helper.o" "./norm.o" "./protocolored.o" "./q2dcm.o" "./read_serial_conf.o" "./rtGetInf.o" "./rtGetNaN.o" "./rt_nonfinite.o" "./util.o" "./xgeqp3.o" "./xnrm2.o" "./drivers/calibrate.o" "./drivers/caplesense.o" "./drivers/capsense.o" "./drivers/cdc.o" "./drivers/cpt112s_i2c.o" "./drivers/display.o" "./drivers/displayls013b7dh03.o" "./drivers/displayls013b7dh06.o" "./drivers/displaypalemlib.o" "./drivers/dmactrl.o" "./drivers/eeprom.o" "./drivers/hidkbd.o" "./drivers/i2ccaptouch.o" "./drivers/i2cspm.o" "./drivers/ksz8851snl.o" "./drivers/ksz8851snl_spi.o" "./drivers/microsd.o" "./drivers/msdbot.o" "./drivers/msdd.o" "./drivers/msdh.o" "./drivers/msdscsi.o" "./drivers/mx25flash_spi.o" "./drivers/nandflash.o" "./drivers/norflash.o" "./drivers/norflash_s29gl.o" "./drivers/retargetio.o" "./drivers/retargetserial.o" "./drivers/retargetswo.o" "./drivers/retargettextdisplay.o" "./drivers/retargettft.o" "./drivers/segmentlcd.o" "./drivers/si1147_i2c.o" "./drivers/si114x_algorithm.o" "./drivers/si114x_functions.o" "./drivers/si114x_sys_out.o" "./drivers/si114x_uv_index.o" "./drivers/si7013.o" "./drivers/si72xx.o" "./drivers/tempsens.o" "./drivers/textdisplay.o" "./drivers/tftamapped.o" "./drivers/tftdirect.o" "./drivers/tftspi.o" "./drivers/touch.o" "./drivers/udelay.o" "./drivers/usbkbdscancodes.o" "./drivers/vddcheck.o" "./emlib/em_adc.o" "./emlib/em_cmu.o" "./emlib/em_core.o" "./emlib/em_emu.o" "./emlib/em_gpio.o" "./emlib/em_i2c.o" "./emlib/em_ldma.o" "./emlib/em_leuart.o" "./emlib/em_mpu.o" "./emlib/em_msc.o" "./emlib/em_prs.o" "./emlib/em_rmu.o" "./emlib/em_rtcc.o" "./emlib/em_system.o" "./emlib/em_timer.o" "./emlib/em_usart.o" "./emlib/em_wdog.o" "./emlib/system_efr32mg12p.o" "./wstk-sensors/wstk-sensors.o" 
	@echo 'Finished building target: $@'
	@echo ' '

	@echo 'Building hex file: Insectt_GPSaidedINS_step_cookie.hex'
	arm-none-eabi-objcopy -O ihex "Insectt_GPSaidedINS_step_cookie.axf" "Insectt_GPSaidedINS_step_cookie.hex"
	@echo ' '

	@echo 'Building bin file: Insectt_GPSaidedINS_step_cookie.bin'
	arm-none-eabi-objcopy -O binary "Insectt_GPSaidedINS_step_cookie.axf" "Insectt_GPSaidedINS_step_cookie.bin"
	@echo ' '

	@echo 'Building s37 file: Insectt_GPSaidedINS_step_cookie.s37'
	arm-none-eabi-objcopy -O srec "Insectt_GPSaidedINS_step_cookie.axf" "Insectt_GPSaidedINS_step_cookie.s37"
	@echo ' '

	@echo 'Running size tool'
	arm-none-eabi-size "Insectt_GPSaidedINS_step_cookie.axf"
	@echo ' '

# Other Targets
clean:
	-$(RM) $(EXECUTABLES)$(OBJS)$(C_DEPS) Insectt_GPSaidedINS_step_cookie.axf
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
