################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include wstk-sensors/subdir.mk
-include stack/stack-common/subdir.mk
-include sim-eeprom1/subdir.mk
-include serial/EFR32/subdir.mk
-include main/EFR32/subdir.mk
-include heartbeat/subdir.mk
-include hal/EFR32_M_1_2_P/gcc/subdir.mk
-include hal/EFR32/subdir.mk
-include hal/subdir.mk
-include emlib/subdir.mk
-include emdrv/subdir.mk
-include ekf/subdir.mk
-include drivers/subdir.mk
-include diagnostic/subdir.mk
-include debug-print/subdir.mk
-include command-interpreter/subdir.mk
-include CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/subdir.mk
-include CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/subdir.mk
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
all: pre-build main-build

# Main-build Target
main-build: Inartrans_v1.axf

# Tool invocations
Inartrans_v1.axf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GNU ARM C Linker'
	arm-none-eabi-gcc -g3 -gdwarf-2 -mcpu=cortex-m4 -mthumb -T "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//platform/base/hal/micro/cortexm3/efm32/gcc-cfg.ld" -L"C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//platform/base/hal/micro/cortexm3/" -Xlinker --defsym=FLASH_SIZE=1048576 -Xlinker --defsym=RAM_SIZE=262144 -Xlinker --defsym=SIMEEPROM_SIZE=8192 -Xlinker --defsym=NULL_BTL=1 -Xlinker --gc-sections -Xlinker -Map="Inartrans_v1.map" -mfpu=fpv4-sp-d16 -mfloat-abi=softfp --specs=nano.specs -u _printf_float -o Inartrans_v1.axf -Wl,--start-group "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_abs_f16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_abs_f32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_abs_f64.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_abs_q15.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_abs_q31.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_abs_q7.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_add_f16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_add_f32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_add_f64.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_add_q15.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_add_q31.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_add_q7.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_and_u16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_and_u32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_and_u8.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_clip_f16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_clip_f32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_clip_q15.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_clip_q31.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_clip_q7.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_dot_prod_f16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_dot_prod_f32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_dot_prod_f64.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_dot_prod_q15.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_dot_prod_q31.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_dot_prod_q7.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_mult_f16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_mult_f32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_mult_f64.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_mult_q15.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_mult_q31.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_mult_q7.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_negate_f16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_negate_f32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_negate_f64.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_negate_q15.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_negate_q31.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_negate_q7.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_not_u16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_not_u32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_not_u8.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_offset_f16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_offset_f32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_offset_f64.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_offset_q15.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_offset_q31.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_offset_q7.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_or_u16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_or_u32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_or_u8.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_scale_f16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_scale_f32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_scale_f64.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_scale_q15.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_scale_q31.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_scale_q7.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_shift_q15.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_shift_q31.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_shift_q7.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_sub_f16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_sub_f32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_sub_f64.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_sub_q15.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_sub_q31.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_sub_q7.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_xor_u16.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_xor_u32.o" "./CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_xor_u8.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_add_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_add_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_add_q15.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_add_q31.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_cholesky_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_cholesky_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_cholesky_f64.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_cmplx_mult_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_cmplx_mult_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_cmplx_mult_q15.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_cmplx_mult_q31.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_cmplx_trans_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_cmplx_trans_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_cmplx_trans_q15.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_cmplx_trans_q31.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_init_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_init_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_init_q15.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_init_q31.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_inverse_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_inverse_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_inverse_f64.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_ldlt_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_ldlt_f64.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_mult_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_mult_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_mult_f64.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_mult_fast_q15.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_mult_fast_q31.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_mult_opt_q31.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_mult_q15.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_mult_q31.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_mult_q7.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_scale_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_scale_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_scale_q15.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_scale_q31.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_solve_lower_triangular_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_solve_lower_triangular_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_solve_lower_triangular_f64.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_solve_upper_triangular_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_solve_upper_triangular_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_solve_upper_triangular_f64.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_sub_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_sub_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_sub_f64.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_sub_q15.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_sub_q31.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_trans_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_trans_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_trans_f64.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_trans_q15.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_trans_q31.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_trans_q7.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_vec_mult_f16.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_vec_mult_f32.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_vec_mult_q15.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_vec_mult_q31.o" "./CMSIS_5/CMSIS/DSP/Source/MatrixFunctions/arm_mat_vec_mult_q7.o" "./board_cookie.o" "./flex-bookkeeping.o" "./flex-callback-stubs.o" "./flex-callbacks.o" "./flex-cli.o" "./flex-events.o" "./gps-uart.o" "./icm20648_r.o" "./imu.o" "./imu_dcm.o" "./imu_fuse.o" "./imu_math.o" "./protocolored.o" "./rail_config.o" "./read_serial_conf.o" "./util.o" "./command-interpreter/command-interpreter2-binary.o" "./command-interpreter/command-interpreter2-error.o" "./command-interpreter/command-interpreter2-util.o" "./command-interpreter/command-interpreter2.o" "./debug-print/debug-print.o" "./diagnostic/assert-crash-handlers.o" "./diagnostic/diagnostic.o" "./drivers/i2cspm.o" "./drivers/si7013.o" "./ekf/cookie_ekf.o" "./emdrv/dmadrv.o" "./emdrv/gpiointerrupt.o" "./emdrv/rtcdriver.o" "./emdrv/sleep.o" "./emdrv/tempdrv.o" "./emdrv/ustimer.o" "./emlib/em_adc.o" "./emlib/em_cmu.o" "./emlib/em_core.o" "./emlib/em_emu.o" "./emlib/em_gpio.o" "./emlib/em_i2c.o" "./emlib/em_ldma.o" "./emlib/em_leuart.o" "./emlib/em_mpu.o" "./emlib/em_msc.o" "./emlib/em_prs.o" "./emlib/em_rmu.o" "./emlib/em_rtcc.o" "./emlib/em_system.o" "./emlib/em_timer.o" "./emlib/em_usart.o" "./emlib/em_wdog.o" "./emlib/system_efr32mg12p.o" "./hal/EFR32/adc-efr32.o" "./hal/EFR32/bsp_stk.o" "./hal/EFR32/em_crypto.o" "./hal/EFR32/faults-v7m.o" "./hal/EFR32/mpu.o" "./hal/EFR32_M_1_2_P/gcc/startup_efr32mg12p.o" "./hal/bsp_stk_leds.o" "./hal/button.o" "./hal/buzzer-efr32.o" "./hal/com.o" "./hal/crc.o" "./hal/cstartup-common.o" "./hal/dmactrl.o" "./hal/hal-config-gpio.o" "./hal/hal-config.o" "./hal/isr-stubs.o" "./hal/led.o" "./hal/mem-util.o" "./hal/mfg-token.o" "./hal/micro-common.o" "./hal/micro.o" "./hal/random.o" "./hal/sleep-efm32.o" "./hal/symbol-timer.o" "./hal/token-def.o" "./hal/token.o" "./hal/uartdrv.o" "./hal/udelay.o" "./heartbeat/heartbeat.o" "./main/EFR32/main.o" "./main/EFR32/stack-handlers.o" "./serial/EFR32/ember-printf.o" "./serial/EFR32/serial.o" "./sim-eeprom1/sim-eeprom.o" "./stack/stack-common/ember-configuration.o" "./wstk-sensors/wstk-sensors.o" "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//protocol/flex/connect/plugins/libraries/hal-efr32mg12p-library-gcc.a" "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//protocol/flex/connect/plugins/libraries/libparent-support-library-gcc.a" "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//protocol/flex/connect/plugins/libraries/libstack-common-library-gcc.a" "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//platform/radio/rail_lib/autogen/librail_release/librail_efr32xg12_gcc_release.a" "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//protocol/flex/connect/plugins/libraries/phy-rail-add-efr32mg12p-library-gcc.a" "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//protocol/flex/connect/plugins/libraries/libsim-eeprom1-library-gcc.a" "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//protocol/flex/connect/plugins/libraries/libaes-security-library-gcc.a" "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//protocol/flex/connect/plugins/libraries/libpacket-queue-library-gcc.a" "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//protocol/flex/connect/plugins/libraries/libform-and-join-library-gcc.a" "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//protocol/flex/connect/plugins/libraries/libfrequency-hopping-stub-library-gcc.a" "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//protocol/flex/connect/plugins/libraries/libstack-counters-library-gcc.a" "C:/SiliconLabs/SimplicityStudio/v4/developer/sdks/gecko_sdk_suite/v2.4//protocol/flex/connect/plugins/libraries/libxxtea-security-stub-library-gcc.a" -lm -Wl,--end-group -Wl,--start-group -lgcc -lc -lnosys -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

	@echo 'Building hex file: Inartrans_v1.hex'
	arm-none-eabi-objcopy -O ihex "Inartrans_v1.axf" "Inartrans_v1.hex"
	@echo ' '

	@echo 'Building bin file: Inartrans_v1.bin'
	arm-none-eabi-objcopy -O binary "Inartrans_v1.axf" "Inartrans_v1.bin"
	@echo ' '

	@echo 'Building s37 file: Inartrans_v1.s37'
	arm-none-eabi-objcopy -O srec "Inartrans_v1.axf" "Inartrans_v1.s37"
	@echo ' '

	@echo 'Running size tool'
	arm-none-eabi-size "Inartrans_v1.axf"
	@echo ' '

	$(MAKE) --no-print-directory post-build

# Other Targets
clean:
	-$(RM) $(EXECUTABLES)$(OBJS)$(C_DEPS) Inartrans_v1.axf
	-@echo ' '

pre-build:
	-cmd //c 'C:\Users\PCNET2~1\SIMPLI~1\V4_WOR~1\NODOPA~2/nodo para mac 2-prebuild.bat' 'Inartrans_v1' 'C:/Users/PCNET22Win10/SimplicityStudio/v4_workspace/Inartrans_v1' '' 'C:\SiliconLabs\SimplicityStudio\v4'
	-@echo ' '

post-build:
	-cmd //c 'C:\Users\PCNET2~1\SIMPLI~1\V4_WOR~1\NODOPA~2/nodo para mac 2-postbuild.bat' 'Inartrans_v1' 'C:/Users/PCNET22Win10/SimplicityStudio/v4_workspace/Inartrans_v1' '' 'C:\SiliconLabs\SimplicityStudio\v4'
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY: main-build pre-build post-build

-include ../makefile.targets
