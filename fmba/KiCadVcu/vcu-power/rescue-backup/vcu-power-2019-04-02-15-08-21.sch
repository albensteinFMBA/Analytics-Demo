EESchema Schematic File Version 2
LIBS:power
LIBS:device
LIBS:transistors
LIBS:conn
LIBS:linear
LIBS:regul
LIBS:74xx
LIBS:cmos4000
LIBS:adc-dac
LIBS:memory
LIBS:xilinx
LIBS:microcontrollers
LIBS:dsp
LIBS:microchip
LIBS:analog_switches
LIBS:motorola
LIBS:texas
LIBS:intel
LIBS:audio
LIBS:interface
LIBS:digital-audio
LIBS:philips
LIBS:display
LIBS:cypress
LIBS:siliconi
LIBS:opto
LIBS:atmel
LIBS:contrib
LIBS:valves
LIBS:pcr
LIBS:vcu-power-cache
EELAYER 25 0
EELAYER END
$Descr A4 11693 8268
encoding utf-8
Sheet 1 1
Title ""
Date ""
Rev ""
Comp ""
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L Q_PMOS_GDS Q1
U 1 1 5C8F3D8E
P 3500 5600
F 0 "Q1" H 3800 5650 50  0000 R CNN
F 1 "Q_PMOS_GDS" H 4150 5550 50  0000 R CNN
F 2 "TO_SOT_Packages_THT:TO-247_Vertical_Neutral123_largePads" H 3700 5700 50  0001 C CNN
F 3 "" H 3500 5600 50  0000 C CNN
	1    3500 5600
	-1   0    0    1   
$EndComp
$Comp
L Q_PMOS_GDS Q2
U 1 1 5C8F3DCB
P 2500 5600
F 0 "Q2" H 2800 5650 50  0000 R CNN
F 1 "Q_PMOS_GDS" H 3150 5550 50  0000 R CNN
F 2 "TO_SOT_Packages_THT:TO-247_Vertical_Neutral123_largePads" H 2700 5700 50  0001 C CNN
F 3 "" H 2500 5600 50  0000 C CNN
	1    2500 5600
	-1   0    0    1   
$EndComp
$Comp
L Q_PMOS_GDS Q3
U 1 1 5C8F3E09
P 1600 5600
F 0 "Q3" H 1900 5650 50  0000 R CNN
F 1 "Q_PMOS_GDS" H 2250 5550 50  0000 R CNN
F 2 "TO_SOT_Packages_THT:TO-247_Vertical_Neutral123_largePads" H 1800 5700 50  0001 C CNN
F 3 "" H 1600 5600 50  0000 C CNN
	1    1600 5600
	-1   0    0    1   
$EndComp
$Comp
L Q_PMOS_GDS Q5
U 1 1 5C8F3E3C
P 5900 4550
F 0 "Q5" H 6200 4600 50  0000 R CNN
F 1 "Q_PMOS_GDS" H 6550 4500 50  0000 R CNN
F 2 "TO_SOT_Packages_THT:TO-220_Neutral123_Vertical" H 6100 4650 50  0001 C CNN
F 3 "" H 5900 4550 50  0000 C CNN
	1    5900 4550
	-1   0    0    1   
$EndComp
Wire Wire Line
	3400 5400 1500 5400
Connection ~ 2400 5400
Wire Wire Line
	3400 5800 1500 5800
Connection ~ 2400 5800
$Comp
L Q_NMOS_GDS Q6
U 1 1 5C8F42B7
P 7650 2250
F 0 "Q6" H 7950 2300 50  0000 R CNN
F 1 "Q_NMOS_GDS" H 8300 2200 50  0000 R CNN
F 2 "TO_SOT_Packages_THT:TO-220_Neutral123_Vertical" H 7850 2350 50  0001 C CNN
F 3 "" H 7650 2250 50  0000 C CNN
	1    7650 2250
	1    0    0    -1  
$EndComp
$Comp
L D D2
U 1 1 5C8F4400
P 2250 6100
F 0 "D2" H 2250 6200 50  0000 C CNN
F 1 "D" H 2250 6000 50  0000 C CNN
F 2 "Diodes_ThroughHole:Diode_TO-220_Dual_CommonCathode_Vertical" H 2250 6100 50  0001 C CNN
F 3 "" H 2250 6100 50  0000 C CNN
	1    2250 6100
	0    1    1    0   
$EndComp
$Comp
L D D4
U 1 1 5C8F4439
P 5800 4900
F 0 "D4" H 5800 5000 50  0000 C CNN
F 1 "D" H 5800 4800 50  0000 C CNN
F 2 "Diodes_ThroughHole:Diode_TO-220_Dual_CommonCathode_Vertical" H 5800 4900 50  0001 C CNN
F 3 "" H 5800 4900 50  0000 C CNN
	1    5800 4900
	0    1    1    0   
$EndComp
$Comp
L D D5
U 1 1 5C8F447C
P 7750 1900
F 0 "D5" H 7750 2000 50  0000 C CNN
F 1 "D" H 7750 1800 50  0000 C CNN
F 2 "Diodes_ThroughHole:Diode_TO-220_Dual_CommonCathode_Vertical" H 7750 1900 50  0001 C CNN
F 3 "" H 7750 1900 50  0000 C CNN
	1    7750 1900
	0    1    1    0   
$EndComp
$Comp
L pcr U1
U 1 1 5C8F4786
P 9450 4800
F 0 "U1" H 10150 4650 60  0000 C CNN
F 1 "pcr" H 9350 4650 60  0000 C CNN
F 2 "footprint:PCR" H 9350 4650 60  0001 C CNN
F 3 "" H 9350 4650 60  0000 C CNN
	1    9450 4800
	1    0    0    -1  
$EndComp
$Comp
L C C1
U 1 1 5C8F486D
P 3850 4150
F 0 "C1" H 3875 4250 50  0000 L CNN
F 1 "C" H 3875 4050 50  0000 L CNN
F 2 "Capacitors_ThroughHole:C_Radial_D12.5_L25_P5" H 3888 4000 50  0001 C CNN
F 3 "" H 3850 4150 50  0000 C CNN
	1    3850 4150
	1    0    0    -1  
$EndComp
$Comp
L R R5
U 1 1 5C8F4E2F
P 9200 4450
F 0 "R5" V 9280 4450 50  0000 C CNN
F 1 "R" V 9200 4450 50  0000 C CNN
F 2 "Resistors_ThroughHole:Resistor_TO-220_Vertical" V 9130 4450 50  0001 C CNN
F 3 "" H 9200 4450 50  0000 C CNN
	1    9200 4450
	0    1    1    0   
$EndComp
$Comp
L +12V #PWR01
U 1 1 5C8FDE1F
P 5800 4350
F 0 "#PWR01" H 5800 4200 50  0001 C CNN
F 1 "+12V" H 5800 4490 50  0000 C CNN
F 2 "" H 5800 4350 50  0000 C CNN
F 3 "" H 5800 4350 50  0000 C CNN
	1    5800 4350
	1    0    0    -1  
$EndComp
$Comp
L +12V #PWR02
U 1 1 5C8FDF33
P 2400 5400
F 0 "#PWR02" H 2400 5250 50  0001 C CNN
F 1 "+12V" H 2400 5540 50  0000 C CNN
F 2 "" H 2400 5400 50  0000 C CNN
F 3 "" H 2400 5400 50  0000 C CNN
	1    2400 5400
	1    0    0    -1  
$EndComp
$Comp
L GNDPWR #PWR03
U 1 1 5C8FE117
P 7750 2450
F 0 "#PWR03" H 7750 2250 50  0001 C CNN
F 1 "GNDPWR" H 7750 2320 50  0000 C CNN
F 2 "" H 7750 2400 50  0000 C CNN
F 3 "" H 7750 2400 50  0000 C CNN
	1    7750 2450
	1    0    0    -1  
$EndComp
Wire Wire Line
	9350 4450 9600 4450
Wire Wire Line
	9600 4450 9600 4500
$Comp
L CONN_01X01 P8
U 1 1 5C8FE801
P 9050 4250
F 0 "P8" H 9050 4350 50  0000 C CNN
F 1 "Bat_Pos" V 9150 4250 50  0000 C CNN
F 2 "footprint:0.210''" H 9050 4250 50  0001 C CNN
F 3 "" H 9050 4250 50  0000 C CNN
	1    9050 4250
	0    -1   -1   0   
$EndComp
$Comp
L +12V #PWR04
U 1 1 5C8FEAC7
P 9900 4500
F 0 "#PWR04" H 9900 4350 50  0001 C CNN
F 1 "+12V" H 9900 4640 50  0000 C CNN
F 2 "" H 9900 4500 50  0000 C CNN
F 3 "" H 9900 4500 50  0000 C CNN
	1    9900 4500
	1    0    0    -1  
$EndComp
$Comp
L GNDPWR #PWR05
U 1 1 5C908060
P 2250 6250
F 0 "#PWR05" H 2250 6050 50  0001 C CNN
F 1 "GNDPWR" H 2250 6120 50  0000 C CNN
F 2 "" H 2250 6200 50  0000 C CNN
F 3 "" H 2250 6200 50  0000 C CNN
	1    2250 6250
	1    0    0    -1  
$EndComp
Wire Wire Line
	2600 5950 2600 5800
Connection ~ 2600 5800
Wire Wire Line
	2250 5950 2250 5800
Connection ~ 2250 5800
$Comp
L CONN_01X01 P2
U 1 1 5C90839D
P 1300 5800
F 0 "P2" H 1300 5900 50  0000 C CNN
F 1 "Mot_Ctrl" H 1550 5800 50  0000 C CNN
F 2 "footprint:0.210''" H 1300 5800 50  0001 C CNN
F 3 "" H 1300 5800 50  0000 C CNN
	1    1300 5800
	-1   0    0    1   
$EndComp
$Comp
L GNDPWR #PWR06
U 1 1 5C908404
P 5800 5050
F 0 "#PWR06" H 5800 4850 50  0001 C CNN
F 1 "GNDPWR" H 5800 4920 50  0000 C CNN
F 2 "" H 5800 5000 50  0000 C CNN
F 3 "" H 5800 5000 50  0000 C CNN
	1    5800 5050
	1    0    0    -1  
$EndComp
$Comp
L CONN_01X01 P4
U 1 1 5C90843C
P 5600 4750
F 0 "P4" H 5600 4850 50  0000 C CNN
F 1 "Mot_Rly_Ctrl" H 5900 4750 50  0000 C CNN
F 2 "footprint:0.130''" H 5600 4750 50  0001 C CNN
F 3 "" H 5600 4750 50  0000 C CNN
	1    5600 4750
	-1   0    0    1   
$EndComp
$Comp
L +12V #PWR07
U 1 1 5C90862F
P 7750 1750
F 0 "#PWR07" H 7750 1600 50  0001 C CNN
F 1 "+12V" H 7750 1890 50  0000 C CNN
F 2 "" H 7750 1750 50  0000 C CNN
F 3 "" H 7750 1750 50  0000 C CNN
	1    7750 1750
	1    0    0    -1  
$EndComp
$Comp
L CONN_01X01 P6
U 1 1 5C908669
P 7950 2050
F 0 "P6" H 7950 2150 50  0000 C CNN
F 1 "S_DN" H 8100 2050 50  0000 C CNN
F 2 "footprint:0.130''" H 7950 2050 50  0001 C CNN
F 3 "" H 7950 2050 50  0000 C CNN
	1    7950 2050
	1    0    0    -1  
$EndComp
$Comp
L Q_NMOS_GDS Q4
U 1 1 5C908B22
P 5750 2100
F 0 "Q4" H 6050 2150 50  0000 R CNN
F 1 "Q_NMOS_GDS" H 6400 2050 50  0000 R CNN
F 2 "TO_SOT_Packages_THT:TO-220_Neutral123_Vertical" H 5950 2200 50  0001 C CNN
F 3 "" H 5750 2100 50  0000 C CNN
	1    5750 2100
	1    0    0    -1  
$EndComp
$Comp
L D D3
U 1 1 5C908B28
P 5850 1750
F 0 "D3" H 5850 1850 50  0000 C CNN
F 1 "D" H 5850 1650 50  0000 C CNN
F 2 "Diodes_ThroughHole:Diode_TO-220_Dual_CommonCathode_Vertical" H 5850 1750 50  0001 C CNN
F 3 "" H 5850 1750 50  0000 C CNN
	1    5850 1750
	0    1    1    0   
$EndComp
$Comp
L GNDPWR #PWR08
U 1 1 5C908B2E
P 5850 2300
F 0 "#PWR08" H 5850 2100 50  0001 C CNN
F 1 "GNDPWR" H 5850 2170 50  0000 C CNN
F 2 "" H 5850 2250 50  0000 C CNN
F 3 "" H 5850 2250 50  0000 C CNN
	1    5850 2300
	1    0    0    -1  
$EndComp
$Comp
L +12V #PWR09
U 1 1 5C908B34
P 5850 1600
F 0 "#PWR09" H 5850 1450 50  0001 C CNN
F 1 "+12V" H 5850 1740 50  0000 C CNN
F 2 "" H 5850 1600 50  0000 C CNN
F 3 "" H 5850 1600 50  0000 C CNN
	1    5850 1600
	1    0    0    -1  
$EndComp
$Comp
L CONN_01X01 P3
U 1 1 5C908B3A
P 6050 1900
F 0 "P3" H 6050 2000 50  0000 C CNN
F 1 "S_UP" H 6200 1900 50  0000 C CNN
F 2 "footprint:0.130''" H 6050 1900 50  0001 C CNN
F 3 "" H 6050 1900 50  0000 C CNN
	1    6050 1900
	1    0    0    -1  
$EndComp
$Comp
L Q_NMOS_GDS Q7
U 1 1 5C908BFF
P 9800 2150
F 0 "Q7" H 10100 2200 50  0000 R CNN
F 1 "Q_NMOS_GDS" H 10450 2100 50  0000 R CNN
F 2 "TO_SOT_Packages_THT:TO-220_Neutral123_Vertical" H 10000 2250 50  0001 C CNN
F 3 "" H 9800 2150 50  0000 C CNN
	1    9800 2150
	1    0    0    -1  
$EndComp
$Comp
L D D6
U 1 1 5C908C05
P 9900 1800
F 0 "D6" H 9900 1900 50  0000 C CNN
F 1 "D" H 9900 1700 50  0000 C CNN
F 2 "Diodes_ThroughHole:Diode_TO-220_Dual_CommonCathode_Vertical" H 9900 1800 50  0001 C CNN
F 3 "" H 9900 1800 50  0000 C CNN
	1    9900 1800
	0    1    1    0   
$EndComp
$Comp
L GNDPWR #PWR010
U 1 1 5C908C0B
P 9900 2350
F 0 "#PWR010" H 9900 2150 50  0001 C CNN
F 1 "GNDPWR" H 9900 2220 50  0000 C CNN
F 2 "" H 9900 2300 50  0000 C CNN
F 3 "" H 9900 2300 50  0000 C CNN
	1    9900 2350
	1    0    0    -1  
$EndComp
$Comp
L +12V #PWR011
U 1 1 5C908C11
P 9900 1650
F 0 "#PWR011" H 9900 1500 50  0001 C CNN
F 1 "+12V" H 9900 1790 50  0000 C CNN
F 2 "" H 9900 1650 50  0000 C CNN
F 3 "" H 9900 1650 50  0000 C CNN
	1    9900 1650
	1    0    0    -1  
$EndComp
$Comp
L CONN_01X01 P9
U 1 1 5C908C17
P 10100 1950
F 0 "P9" H 10100 2050 50  0000 C CNN
F 1 "PR_Coil" H 10300 1950 50  0000 C CNN
F 2 "footprint:0.05''" H 10100 1950 50  0001 C CNN
F 3 "" H 10100 1950 50  0000 C CNN
	1    10100 1950
	1    0    0    -1  
$EndComp
$Comp
L CONN_01X12 P1
U 1 1 5C90A1AD
P 3500 2100
F 0 "P1" H 3500 2750 50  0000 C CNN
F 1 "CONN_01X12" V 3600 2100 50  0000 C CNN
F 2 "footprint:connnector12-1" H 3500 2100 50  0001 C CNN
F 3 "" H 3500 2100 50  0000 C CNN
	1    3500 2100
	1    0    0    -1  
$EndComp
$Comp
L GNDPWR #PWR012
U 1 1 5C90A575
P 2450 1450
F 0 "#PWR012" H 2450 1250 50  0001 C CNN
F 1 "GNDPWR" H 2450 1320 50  0000 C CNN
F 2 "" H 2450 1400 50  0000 C CNN
F 3 "" H 2450 1400 50  0000 C CNN
	1    2450 1450
	1    0    0    -1  
$EndComp
Wire Wire Line
	2450 1450 3300 1450
Wire Wire Line
	3300 1450 3300 1550
$Comp
L GNDPWR #PWR013
U 1 1 5C90A5DE
P 3300 2650
F 0 "#PWR013" H 3300 2450 50  0001 C CNN
F 1 "GNDPWR" H 3300 2520 50  0000 C CNN
F 2 "" H 3300 2600 50  0000 C CNN
F 3 "" H 3300 2600 50  0000 C CNN
	1    3300 2650
	1    0    0    -1  
$EndComp
$Comp
L +12V #PWR014
U 1 1 5C90A620
P 2150 2050
F 0 "#PWR014" H 2150 1900 50  0001 C CNN
F 1 "+12V" H 2150 2190 50  0000 C CNN
F 2 "" H 2150 2050 50  0000 C CNN
F 3 "" H 2150 2050 50  0000 C CNN
	1    2150 2050
	1    0    0    -1  
$EndComp
Wire Wire Line
	2150 2050 3300 2050
Text GLabel 3300 1650 0    60   Input ~ 0
NMOS_S_UP
Text GLabel 3300 1750 0    60   Input ~ 0
NMOS_S_DN
Text GLabel 3300 1950 0    60   Input ~ 0
NMOS_PR
Text GLabel 3300 2350 0    60   Input ~ 0
1PMOS
Text GLabel 3000 2550 0    60   Input ~ 0
3PMOS
NoConn ~ 3300 1850
NoConn ~ 3300 2250
NoConn ~ 3300 2450
$Comp
L R R2
U 1 1 5C90AAB3
P 5400 2100
F 0 "R2" V 5480 2100 50  0000 C CNN
F 1 "22" V 5400 2100 50  0000 C CNN
F 2 "Resistors_ThroughHole:Resistor_Horizontal_RM7mm" V 5330 2100 50  0001 C CNN
F 3 "" H 5400 2100 50  0000 C CNN
	1    5400 2100
	0    1    1    0   
$EndComp
$Comp
L R R4
U 1 1 5C90AD98
P 7300 2250
F 0 "R4" V 7380 2250 50  0000 C CNN
F 1 "22" V 7300 2250 50  0000 C CNN
F 2 "Resistors_ThroughHole:Resistor_Horizontal_RM7mm" V 7230 2250 50  0001 C CNN
F 3 "" H 7300 2250 50  0000 C CNN
	1    7300 2250
	0    1    1    0   
$EndComp
$Comp
L R R6
U 1 1 5C90AEE3
P 9450 2150
F 0 "R6" V 9530 2150 50  0000 C CNN
F 1 "22" V 9450 2150 50  0000 C CNN
F 2 "Resistors_ThroughHole:Resistor_Horizontal_RM7mm" V 9380 2150 50  0001 C CNN
F 3 "" H 9450 2150 50  0000 C CNN
	1    9450 2150
	0    1    1    0   
$EndComp
$Comp
L R R3
U 1 1 5C90B1C2
P 6250 4550
F 0 "R3" V 6330 4550 50  0000 C CNN
F 1 "22" V 6250 4550 50  0000 C CNN
F 2 "Resistors_ThroughHole:Resistor_Horizontal_RM7mm" V 6180 4550 50  0001 C CNN
F 3 "" H 6250 4550 50  0000 C CNN
	1    6250 4550
	0    -1   -1   0   
$EndComp
Text GLabel 6400 4550 2    60   Input ~ 0
1PMOS
Text GLabel 5250 2100 0    60   Input ~ 0
NMOS_S_UP
Text GLabel 7150 2250 0    60   Input ~ 0
NMOS_S_DN
Text GLabel 9300 2150 0    60   Input ~ 0
NMOS_PR
$Comp
L GNDPWR #PWR015
U 1 1 5C90C48B
P 8700 5100
F 0 "#PWR015" H 8700 4900 50  0001 C CNN
F 1 "GNDPWR" H 8700 4970 50  0000 C CNN
F 2 "" H 8700 5050 50  0000 C CNN
F 3 "" H 8700 5050 50  0000 C CNN
	1    8700 5100
	1    0    0    -1  
$EndComp
$Comp
L CONN_01X01 P7
U 1 1 5C90C4D5
P 8700 4600
F 0 "P7" H 8700 4700 50  0000 C CNN
F 1 "Key_Sw" V 8800 4600 50  0000 C CNN
F 2 "footprint:0.130''" H 8700 4600 50  0001 C CNN
F 3 "" H 8700 4600 50  0000 C CNN
	1    8700 4600
	0    -1   -1   0   
$EndComp
$Comp
L CONN_01X01 P10
U 1 1 5C90C8A5
P 10100 4500
F 0 "P10" H 10100 4600 50  0000 C CNN
F 1 "Sw_Pos" H 10300 4500 50  0000 C CNN
F 2 "footprint:0.130''" H 10100 4500 50  0001 C CNN
F 3 "" H 10100 4500 50  0000 C CNN
	1    10100 4500
	1    0    0    -1  
$EndComp
$Comp
L +12V #PWR016
U 1 1 5C90CB83
P 3850 4000
F 0 "#PWR016" H 3850 3850 50  0001 C CNN
F 1 "+12V" H 3850 4140 50  0000 C CNN
F 2 "" H 3850 4000 50  0000 C CNN
F 3 "" H 3850 4000 50  0000 C CNN
	1    3850 4000
	1    0    0    -1  
$EndComp
$Comp
L GNDPWR #PWR017
U 1 1 5C90CBD1
P 3850 4300
F 0 "#PWR017" H 3850 4100 50  0001 C CNN
F 1 "GNDPWR" H 3850 4170 50  0000 C CNN
F 2 "" H 3850 4250 50  0000 C CNN
F 3 "" H 3850 4250 50  0000 C CNN
	1    3850 4300
	1    0    0    -1  
$EndComp
$Comp
L GNDPWR #PWR018
U 1 1 5C90CE2C
P 7650 3850
F 0 "#PWR018" H 7650 3650 50  0001 C CNN
F 1 "GNDPWR" H 7650 3720 50  0000 C CNN
F 2 "" H 7650 3800 50  0000 C CNN
F 3 "" H 7650 3800 50  0000 C CNN
	1    7650 3850
	1    0    0    -1  
$EndComp
$Comp
L CONN_01X01 P5
U 1 1 5C90CF28
P 7650 3650
F 0 "P5" H 7650 3750 50  0000 C CNN
F 1 "Bat_Neg" V 7750 3650 50  0000 C CNN
F 2 "footprint:0.130''" H 7650 3650 50  0001 C CNN
F 3 "" H 7650 3650 50  0000 C CNN
	1    7650 3650
	0    -1   -1   0   
$EndComp
$Comp
L R R1
U 1 1 5C90D8D1
P 3150 2550
F 0 "R1" V 3230 2550 50  0000 C CNN
F 1 "22" V 3150 2550 50  0000 C CNN
F 2 "Resistors_ThroughHole:Resistor_Horizontal_RM7mm" V 3080 2550 50  0001 C CNN
F 3 "" H 3150 2550 50  0000 C CNN
	1    3150 2550
	0    1    1    0   
$EndComp
Text GLabel 3700 5600 2    60   Input ~ 0
3PMOS
Text GLabel 2700 5600 2    60   Input ~ 0
3PMOS
Text GLabel 1800 5600 2    60   Input ~ 0
3PMOS
NoConn ~ 3300 2150
$Comp
L D D1
U 1 1 5C8F43B3
P 2600 6100
F 0 "D1" H 2600 6200 50  0000 C CNN
F 1 "D" H 2600 6000 50  0000 C CNN
F 2 "Diodes_ThroughHole:Diode_TO-220_Dual_CommonCathode_Vertical" H 2600 6100 50  0001 C CNN
F 3 "" H 2600 6100 50  0000 C CNN
	1    2600 6100
	0    1    1    0   
$EndComp
Wire Wire Line
	2600 6250 2250 6250
$EndSCHEMATC