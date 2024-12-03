
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import math

time_out_small = 18000
time_out_time = 18000

def expandTOs(arr, exp_cnt):
	assert len(arr) <= exp_cnt
	arr = arr + [time_out_time ]*(exp_cnt - len(arr))
	arr = np.array(arr)
	arr[arr > time_out_small] = time_out_small
	arr = list(arr)
	return arr

# Delay
dly_cnt = 16
dly_nuX = expandTOs([2.5, 7, 29, 121, 292, 529, 870, 1277, 1809, 2448, 3516, 4461], dly_cnt)
dly_nuR = expandTOs([44, 51 , 80, 92, 157 , 162 , 197 , 214 , 321 , 306 , 390 , 412 , 674 , 1500 , 3365 , 8684], dly_cnt)
dly_abc = expandTOs([398, 1759, 8666], dly_cnt)
dly_nnT = expandTOs([6, 10, 7, 7, 41, 24, 15, 36, 23, 15, 44, 17, 22, 26, 45, 124], dly_cnt)
dly_indX = expandTOs([442, 802, 801, 815, 788, 814, 809, 793, 809, 804, 789, 788, 808, 815, 802, 813], dly_cnt)
dly_sss = [(2**3)*750, (2**3)*1250, (2**3)*2500, (2**3)*5000, (2**3)*7500, (2**3)*10000, (2**3)*12500, (2**3)*15000, (2**3)*17500, (2**3)*20000, (2**3)*22500, (2**3)*25000, (2**3)*50000, (2**3)*100000, (2**3)*200000, (2**3)*400000]

dly_nuR_A1 = expandTOs([14, 11, 15, 21, 47, 90, 169, 182, 221, 255, 420, 383, 443, 686, 1239, 4526], dly_cnt)
dly_nuR_A2 = expandTOs([21, 23, 27, 37, 41, 52, 119, 109, 104, 132, 161, 177, 204, 618, 2217, 3272], dly_cnt)
dly_nuR_A4 = expandTOs([121, 172, 324, 330, 609, 410, 427, 595, 789, 654, 1093, 1200, 1446, 3463, 6525], dly_cnt)
dly_nuR_EL = expandTOs([29, 30, 64, 71, 81, 84, 84, 71, 115, 125, 192, 135], dly_cnt)
dly_nuR_M = expandTOs([44, 66, 48, 53, 57, time_out_time, 64, time_out_time, 160, time_out_time, time_out_time, time_out_time, 375, 643, time_out_time, 1597], dly_cnt)
bre

# GRAY
gry_cnt = 11
gry_nuX = expandTOs([0.3, 1.2, 5, 22, 96, 447, 2062, 12935], gry_cnt)
gry_nuR = expandTOs([18   , 30   , 25   , 37   , 62   , 102, 175, 299, 639, 1566, 5790], gry_cnt)
gry_abc = expandTOs([53, 78, 233, 6490, 6217], gry_cnt)
gry_nnT = expandTOs([3, 3, 6, 10, 4, 12, 17, 28, 39, 118, 218], gry_cnt)
gry_indX = expandTOs([81, 293, 784, 795, 789, 786, 801, 797, 811, 792, 787], gry_cnt)
gry_sss = [(2**9), 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19]

gry_nuR_A1 = expandTOs([5, 5, 9, 12, 14, 39, 113, 444, 648, 1014, 2207], gry_cnt)
gry_nuR_A2 = expandTOs([9, 12, 19, 27, 30, 40, 124, 163, 308, 849, 3318], gry_cnt)
gry_nuR_A4 = expandTOs([66, 91, 148, 346, 425, 309, 648, 1040, 2178, 4727, 10035], gry_cnt)
gry_nuR_EL = expandTOs([22, 14, 30, 47, 56, 73, 101], gry_cnt)
gry_nuR_M = expandTOs([8, 18, 25, 15, 46, 81, 170, 178, time_out_time, 1395, 1480], gry_cnt)


# i2c
i2c_cnt = 20
i2c_nuX = expandTOs([2.5, 10, 44, 196, 527, 1038, 1801, 2738, 9288, 15674, 21726], i2c_cnt)
i2c_nuR = expandTOs([49 , 74 , 102, 163, 170, 173, 173, 245, 289, 474, 354, 386, 358, 417, 661, 585, 615, 908], i2c_cnt)
i2c_abc = expandTOs([201, 1195, 6396], i2c_cnt)
i2c_nnT = expandTOs([6, 9, 12, 78, 42, 25, 29, 62, 49, 63, 62, 97, 134, 114, 342, 140, 332, 474], i2c_cnt)
i2c_indX = expandTOs([169, time_out_time, 793, 801, 795, 797, 794, 800, 797, time_out_time, 807, 805, 798, 818, 792, 812, 798, 824, 817, 811], i2c_cnt)
i2c_sss = [(2**3)*500, (2**3)*1000, (2**3)*2000, (2**3)*4000, (2**3)*6000, (2**3)*8000, (2**3)*10000, (2**3)*12000, (2**3)*14000, (2**3)*16000, (2**3)*18000, (2**3)*20000, (2**3)*22000, (2**3)*24000, (2**3)*26000, (2**3)*28000, (2**3)*40000, (2**3)*70000, (2**3)*140000, (2**3)*280000]

i2c_nuR_A1 = expandTOs([], i2c_cnt)
i2c_nuR_A2 = expandTOs([time_out_time, time_out_time, time_out_time, 921], i2c_cnt)
i2c_nuR_A4 = expandTOs([288, time_out_time, time_out_time, 596, time_out_time, 987, time_out_time, 815, 1637, 1595, 1016, time_out_time, 1116, time_out_time, 1499, time_out_time, 2372], i2c_cnt)
i2c_nuR_EL = expandTOs([time_out_time]*4 + [254] + [time_out_time]*7 + [754], i2c_cnt)
i2c_nuR_M = expandTOs([56, 73, 368], i2c_cnt)


# LCD 
lcd_cnt = 14
lcd_nuX = expandTOs([0.8, 1.9, 12, 12, 951, 4444, 3262, 11061, 1743, 20550], lcd_cnt)
lcd_nuR = expandTOs([42 , 63 , 53 , 83 , 245, 297, 360, 335, 355, 470, 360, 622, 3321, 7133], lcd_cnt)
lcd_abc = expandTOs([129, 1189, 1712], lcd_cnt)
lcd_nnT = expandTOs([4 , 5 , 5 , 8 , 46, 18, 22, 31, 38, 32, 23, 102, 79, 181], lcd_cnt)
lcd_indX = expandTOs([55, 215, 808, 799, 838, 815, 843, 828, 807, 837, 825, 805, 850, 817], lcd_cnt)
lcd_sss = [(2**12)*500, (2**12)*1000, (2**12)*1500, (2**12)*2500, (2**12)*5000, (2**12)*7500, (2**12)*10000, (2**12)*12500, (2**12)*15000, (2**12)*17500, (2**12)*20000, (2**12)*22500, (2**12)*90000, (2**12)*180000]

lcd_nuR_A1 = expandTOs([], lcd_cnt)
lcd_nuR_A2 = expandTOs([15, 23, 34, 32, 51, 116, 234, time_out_time, time_out_time, 297, 429, 224, 1905, 3041], lcd_cnt)
lcd_nuR_A4 = expandTOs([172, 349, 335, 575, 1043, 829, 1106, 1329, 1328, 2751, 1679, 2660, 7537], lcd_cnt)
lcd_nuR_EL = expandTOs([], lcd_cnt)
lcd_nuR_M = expandTOs([30, 29, time_out_time, time_out_time, 173, time_out_time, time_out_time, time_out_time, time_out_time, time_out_time, 179, time_out_time, 929], lcd_cnt)



# Load-Store 
ls_cnt = 16
ls_nuX = expandTOs([16, 56, 251, 1263, 2612, 6722, 9490, 12665], ls_cnt)
ls_nuR = expandTOs([51   , 53  , 78  , 126 , 185 , 218 , 403 , 300 , 473 , 486 , 558 , 695 , 1420, 4336, 14533, 39981], ls_cnt)
ls_abc = expandTOs([768, 10772], ls_cnt)
ls_nnT = expandTOs([6  , 5  , 3  , 19 , 21 , 26 , 22 , 24 , 24 , 19 , 22 , 75 , 22 , 125, 197, 88], ls_cnt)
ls_indX = expandTOs([626, 859, 859, 864, 874, 878, 871, 873, 870, 845, 867, 875, 877, 881, 860, 878], ls_cnt)
ls_sss = [(2**2)*750, (2**2)*1250, (2**2)*2500, (2**2)*5000, (2**2)*7500, (2**2)*10000, (2**2)*12500, (2**2)*15000, (2**2)*17500, (2**2)*20000, (2**2)*22500, (2**2)*25000, (2**2)*50000, (2**2)*100000, (2**2)*200000, (2**2)*400000]

ls_nuR_A1 = expandTOs([], ls_cnt)
ls_nuR_A2 = expandTOs([18, 20, 49, time_out_time, time_out_time, time_out_time, 1092, 706, time_out_time, 366, 299, 334, 838, 1668, 5387, 25467], ls_cnt)
ls_nuR_A4 = expandTOs([264, 279, time_out_time, 740, 997, 1214, 1264, 1591, 1336, 1605, 1756, 1869, 4168, 8826], ls_cnt)
ls_nuR_EL = expandTOs([], ls_cnt)
ls_nuR_M = expandTOs([time_out_time]*4 + [132, 206, time_out_time, 179, 361, 357, time_out_time, 449, 1345, time_out_time, 4782], ls_cnt)


# PWM
pwm_cnt = 12
pwm_nuX = expandTOs([1.5, 6.5, 30, 137, 638, 2563, 10667, 61118], pwm_cnt)
pwm_nuR = expandTOs([30  , 26  , 97  , 69  , 94  , 177 , 365 , 1028, 2611, 6527, 9353], pwm_cnt)
pwm_abc = expandTOs([926, 5087], pwm_cnt)
pwm_nnT = expandTOs([2  , 1  , 4  , 12 , 14 , 30 , 64 , 334, 17, 20, 33], pwm_cnt)
pwm_indX = expandTOs([792, 776, 785, 782, 788, 773, 787, 798, 781, 788, 787, 785], pwm_cnt)
pwm_sss = [2**11, (2**12), 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20, 2**21, 2**22]

pwm_nuR_A1 = expandTOs([65, 77, 72, 85, 601, 374, 695, 1364, 6030, 5667, 29992], pwm_cnt)
pwm_nuR_A2 = expandTOs([15, 12, 14, 20, 24, 68, 154], pwm_cnt)
pwm_nuR_A4 = expandTOs([83, 109, 255], pwm_cnt)
pwm_nuR_EL = expandTOs([18, 16, 44, 89, 102, 147, 241, 462, 1561, 1597, 8019], pwm_cnt)
pwm_nuR_M = expandTOs([18, 85, 25, 71, 41, 119, 248, 355, time_out_time, 1943, 8190], pwm_cnt)


# SevenSeg 
sg7_cnt = 15
sg7_nuX = expandTOs([2, 8, 70, 1405, 11605, 58181], sg7_cnt)
sg7_nuR = expandTOs([21  , 34  , 24  , 38  , 47  , 58  , 86  , 104 , 215 , 154 , 242 , 208 , 495 , 977 , 2077], sg7_cnt)
sg7_abc = expandTOs([39, 119, 614, 1469], sg7_cnt)
sg7_nnT = expandTOs([5, 5, 4, 5, 5, 4, 5, 6, 15, 10, 11, 20, 18, 26, 68], sg7_cnt)
sg7_indX = expandTOs([28, 192, 467, 680, 812, 806, 820, 815, 816, 816, 817, 817, 815, 819, 824], sg7_cnt)
sg7_sss = [2**7*250, 2**7*500, 2**7*750, 2**7*1000, 2**7*2500, 2**7*5000, 2**7*7500, 2**7*10000, 2**7*12500, 2**7*15000, 2**7*17500, 2**7*20000, 2**7*40000, 2**7*80000, 2**7*160000]

sg7_nuR_A1 = expandTOs([16, time_out_time, 484, 675, time_out_time, 210, time_out_time, 432, 611, 542, 574, 790, 981, 988, 1579], sg7_cnt)
sg7_nuR_A2 = expandTOs([time_out_time]*8+[64], sg7_cnt)
sg7_nuR_A4 = expandTOs([73, 130, 160, 189, 199, 349, 329, 479, 611, 641, 858, 835, 1109, 4137, 7811], sg7_cnt)
sg7_nuR_EL = expandTOs([time_out_time]*12+[836], sg7_cnt)
sg7_nuR_M = expandTOs([time_out_time]*3+[14, time_out_time, time_out_time, time_out_time, 118, time_out_time, time_out_time, 164, time_out_time, 399], sg7_cnt)



# Thermocouple
tmp_cnt = 17
tmp_nuX = expandTOs([0.6, 9, 361, 601, 306, 573, 1192, 1935, 4224, 6365, 9691, 15129, 30426], tmp_cnt)
tmp_nuR = expandTOs([22 , 67 , 95 , 98 , 164, 156, 182, 210, 326, 516, 389, 932, 949, 1552, 6020, 7331, 13042], tmp_cnt)
tmp_abc = expandTOs([2, 62, 234, 344, 1872, 1246, 2195], tmp_cnt)
tmp_nnT = expandTOs([7, 23, 11, 11, 16, 12, 12, 23, 17, 20, 44, 39, 39, 104, 79, 60, 118], tmp_cnt)
tmp_indX = expandTOs([1, 470, 41, 103, 414, 246, 308, 532, 798, 796, 797, 791, 807, 803, 805, 800, 786], tmp_cnt)
tmp_sss = [2**33*30, 2**33*300, 2**33*600, 2**33*900, 2**33*1200, 2**33*1800, 2**33*2400, 2**33*3000, 2**33*6000, 2**33*9000, 2**33*12000, 2**33*15000, 2**33*18000, 2**33*36000, 2**33*72000, 2**33*144000, 2**33*288000]

tmp_nuR_A1 = expandTOs([], tmp_cnt)
tmp_nuR_A2 = expandTOs([12, 30, 46, 45, 59, 76, 94, 85, 330, 179, 246, 522, 509, 730, 1613, 4896, 8295], tmp_cnt)
tmp_nuR_A4 = expandTOs([70, 221, 235, 364, 269, 316, 320, 377, 1088, 1131, 3382, 3846, 3931, 4121], tmp_cnt)
tmp_nuR_EL = expandTOs([time_out_time]*6 + [162] + [time_out_time]*9 +[8406], tmp_cnt)
tmp_nuR_M = expandTOs([16, time_out_time, 42, 54, 96, 57, time_out_time, time_out_time, 115, time_out_time, 263, 451, time_out_time, 1023, 1276, 10333], tmp_cnt)




# UART Transmit
utt_cnt = 10
utt_nuX = expandTOs([0.04, 0.06, 0.08, 0.4, 0.24, 0.09, 0.14, 0.1, 0.12, 3.28], utt_cnt)
utt_nuR = expandTOs([35 , 29, 148, 74, 188, 371, 222, 1804, 567, 112], utt_cnt)
utt_abc = expandTOs([0.48, 0.39, 0.45, 0.45, 0.42, 0.39, 0.49, 0.46, 0.72, 0.79], utt_cnt)
utt_nnT = expandTOs([7 , 11, 19, 53, 103, 295, 26, 514, 51, 46], utt_cnt)
utt_indX = expandTOs([1.34, 1.30, 1.94, 1.21, 1.22, 1.25, 1.32, 1.21, 1.29, 1.41], utt_cnt)
utt_sss = [2**10, 2**13, 2**15, 2**16, 2**17, 2**18, 2**19, 2**21, 2**22, 2**23]

utt_nuR_A1 = expandTOs([], utt_cnt)
utt_nuR_A2 = expandTOs([20, 16, 24, 25, 32, 34, 42, 81, 303, 422], utt_cnt)
utt_nuR_A4 = expandTOs([109, 90, 593, 97, 281, 178], utt_cnt)
utt_nuR_EL = expandTOs([62, 234, 391, 429, 760, 1792, 251], utt_cnt)
utt_nuR_M = expandTOs([time_out_time, 21, 105, 101, time_out_time, 206, 195, 4677], utt_cnt)


# VGA
vga_cnt = 10
vga_nuX = expandTOs([25, 781, 4448, 19836], vga_cnt)
vga_nuR = expandTOs([], vga_cnt)
vga_abc = expandTOs([26, 28764, 2870, 4748], vga_cnt)
vga_nnT = expandTOs([], vga_cnt)
vga_indX = expandTOs([90, 794, 796, 795, 792, 792, 793, 801, 805, 841], vga_cnt)
vga_sss = [(94*1)**2*2**5, (94*2)**2*2**5, (94*3)**2*2**6, (94*4)**2*2**6, (94*5)**2*2**7, (94*6)**2*2**7, (94*8)**2*2**8, (94*10)**2*2**8, (94*12)**2*2**9, (94*16)**2*2**9]

vga_nuR_A1 = expandTOs([], vga_cnt)
vga_nuR_A2 = expandTOs([], vga_cnt)
vga_nuR_A4 = expandTOs([272, time_out_time, time_out_time, 2292, 3927, 15612], vga_cnt)
vga_nuR_EL = expandTOs([], vga_cnt)
vga_nuR_M = expandTOs([82, 303, 273, 637, 948, 1135, 2129, 3247, 13628], vga_cnt)


# Delay 2
dly2_cnt = 16
dly2_nuX = expandTOs([3, 7, 30, 130, 308, 570, 917, 1349, 1912, 2605, 3597, 4439, 18666], dly2_cnt)
dly2_nuR = expandTOs([89, 85, 181, 204, 312, 471 , 711 , 532 , 662, 746 , 885 , 930 , 2654 , 3893, 5700, 12697], dly2_cnt)
dly2_abc = expandTOs([231, 379, 3396], dly2_cnt)
dly2_nnT = expandTOs([12, 10, 12, 14, 30, 145, 158, 170, 25, 214, 226, 200, 363, 728, 588, 797], dly2_cnt)
dly2_indX = expandTOs([993, 6568, time_out_time, time_out_time, 4931], dly2_cnt)
dly2_sss = [2**3*750, 2**3*1250, 2**3*2500, 2**3*5000, 2**3*7500, 2**3*10000, 2**3*12500, 2**3*15000, 2**3*17500, 2**3*20000, 2**3*22500, 2**3*25000, 2**3*50000, 2**3*100000, 2**3*200000, 2**3*400000]

dly2_nuR_A1 = expandTOs([], dly2_cnt)
dly2_nuR_A2 = expandTOs([27, 24, 50, 79, 187], dly2_cnt)
dly2_nuR_A4 = expandTOs([380, 1066, 1276, 2264, 1889, 1971, time_out_time, 2706, time_out_time, 2949, 3292, time_out_time, 6326], dly2_cnt)
dly2_nuR_EL = expandTOs([], dly2_cnt)
dly2_nuR_M = expandTOs([], dly2_cnt)


# GRAY 2
gry2_cnt = 11
gry2_nuX = expandTOs([0.3, 1, 5, 24, 110, 511, 2441, 14518], gry2_cnt)
gry2_nuR = expandTOs([41  , 97  , 160 , 207 , 302 , 292 , 862 , 2958, 3847, 4676, 8834], gry2_cnt)
gry2_abc = expandTOs([49, 196, 358, 1559, 27986], gry2_cnt)
gry2_nnT = expandTOs([3, 3, 3, 3, 5, 9, 7, 8, 10, 18, 36], gry2_cnt)
gry2_indX = expandTOs([824, 417, 3204, 3661, 13164, time_out_time, 3341, time_out_time, time_out_time, time_out_time, time_out_time], gry2_cnt)
gry2_sss = [2**9, 2**10, 2**11, (2**12), 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19]

gry2_nuR_A1 = expandTOs([], gry2_cnt)
gry2_nuR_A2 = expandTOs([41, 40, 37, 61, 82, 162], gry2_cnt)
gry2_nuR_A4 = expandTOs([118, 268, 332, time_out_time, 831, time_out_time, time_out_time, 3201, 6981, 17888], gry2_cnt)
gry2_nuR_EL = expandTOs([], gry2_cnt)
gry2_nuR_M = expandTOs([], gry2_cnt)


# SevenSeg 
sg7_2_cnt = 15
sg7_2_nuX = expandTOs([1.5, 5, 20, 1463, 13208], sg7_2_cnt)
sg7_2_nuR = expandTOs([30  , 60  , 59  , 75  , 102 , 125 , 181 , 207 , 438 , 238 , 439 , 343 , 578 , 2121, 2187], sg7_2_cnt)
sg7_2_abc = expandTOs([69, 350, 3606, 2663], sg7_2_cnt)
sg7_2_nnT = expandTOs([5, 6, 5, 5, 5, 6, 7, 12, 9, 14, 15, 14, 22, 65, 48], sg7_2_cnt)
sg7_2_indX = expandTOs([558, 5352, time_out_time, 2332], sg7_2_cnt)
sg7_2_sss = [2**7*250, 2**7*500, 2**7*750, 2**7*1000, 2**7*2500, 2**7*5000, 2**7*7500, 2**7*10000, 2**7*12500, 2**7*15000, 2**7*17500, 2**7*20000, 2**7*40000, 2**7*80000, 2**7*160000]

sg7_2_nuR_A1 = expandTOs([], sg7_2_cnt)
sg7_2_nuR_A2 = expandTOs([22, 46, 33, 39, 44, 60, 117, 143, 260, 198, 210, 220, 366, 2070], sg7_2_cnt)
sg7_2_nuR_A4 = expandTOs([121, 217, 280, 398, 752, 1124, 996, 1306, 3085, 1839, 2349, 1907, 3597, 5107], sg7_2_cnt)
sg7_2_nuR_EL = expandTOs([], sg7_2_cnt)
sg7_2_nuR_M = expandTOs([], sg7_2_cnt)



# GRAY 3
gry3_cnt = 11
gry3_nuX = expandTOs([0.3, 1.25, 5, 24, 105, 491, 2387, 14287], gry3_cnt)
gry3_nuR = expandTOs([41 , 52  , 100, 132, 260 , 256 , 1091, 947  , 2300, 5052, 9888], gry3_cnt)
gry3_abc = expandTOs([88, 139, 4428, 4373, 22629], gry3_cnt)
gry3_nnT = expandTOs([8, 12, 5, 6, 73, 17, 50, 176, 580, 1685, 169], gry3_cnt)
gry3_indX = expandTOs([94, 539, 3349, 3688, time_out_time, 3488], gry3_cnt)
gry3_sss = [2**9, 2**10, 2**11, (2**12), 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19]

gry3_nuR_A1 = expandTOs([], gry3_cnt)
gry3_nuR_A2 = expandTOs([23, 21, 49, 38, 54, 123, 228, 549, 1470, 2155, 6288], gry3_cnt)
gry3_nuR_A4 = expandTOs([147, 156, 288, 381, 1600, 1792, 2343, 2864, 6352], gry3_cnt)
gry3_nuR_EL = expandTOs([time_out_time]*7 + [757, 1879, 3266, 7596], gry3_cnt)
gry3_nuR_M = expandTOs([], gry3_cnt)


def a_print(arr):
	for ar in arr:
		if ar<time_out_small:
			print(ar)
		else:
			print("oot.")

def ar_min(arr1, arr2):
	return np.minimum(np.array(arr1), np.array(arr2))

# Generate random data
nuX 	= np.array( dly_nuX + gry_nuX + i2c_nuX + lcd_nuX + ls_nuX + pwm_nuX + sg7_nuX + tmp_nuX + utt_nuX + vga_nuX + dly2_nuX + gry2_nuX + sg7_2_nuX + gry3_nuX)
ABC		= np.array( dly_abc + gry_abc + i2c_abc + lcd_abc + ls_abc + pwm_abc + sg7_abc + tmp_abc + utt_abc + vga_abc + dly2_abc + gry2_abc + sg7_2_abc + gry3_abc)
nuR		= np.array( dly_nuR + gry_nuR + i2c_nuR + lcd_nuR + ls_nuR + pwm_nuR + sg7_nuR + tmp_nuR + utt_nuR + vga_nuR + dly2_nuR + gry2_nuR + sg7_2_nuR + gry3_nuR)
nnT     = np.array( dly_nnT + gry_nnT + i2c_nnT + lcd_nnT + ls_nnT + pwm_nnT + sg7_nnT + tmp_nnT + utt_nnT + vga_nnT + dly2_nnT + gry2_nnT + sg7_2_nnT + gry3_nnT)
indX     = np.array( dly_indX + gry_indX + i2c_indX + lcd_indX + ls_indX + pwm_indX + sg7_indX + tmp_indX + utt_indX + vga_indX + dly2_indX + gry2_indX + sg7_2_indX + gry3_indX)
sss     = np.array( dly_sss + gry_sss + i2c_sss + lcd_sss + ls_sss + pwm_sss + sg7_sss + tmp_sss + utt_sss + vga_sss + dly2_sss + gry2_sss + sg7_2_sss + gry3_sss)

nuR_A1	= np.array(dly_nuR_A1 + gry_nuR_A1 + i2c_nuR_A1 + lcd_nuR_A1 + ls_nuR_A1 + pwm_nuR_A1 + sg7_nuR_A1 + tmp_nuR_A1 + utt_nuR_A1 + vga_nuR_A1 + dly2_nuR_A1 + gry2_nuR_A1 + sg7_2_nuR_A1 + gry3_nuR_A1)
nuR_A2	= np.array(dly_nuR_A2 + gry_nuR_A2 + i2c_nuR_A2 + lcd_nuR_A2 + ls_nuR_A2 + pwm_nuR_A2 + sg7_nuR_A2 + tmp_nuR_A2 + utt_nuR_A2 + vga_nuR_A2 + dly2_nuR_A2 + gry2_nuR_A2 + sg7_2_nuR_A2 + gry3_nuR_A2)
nuR_A4	= np.array(dly_nuR_A4 + gry_nuR_A4 + i2c_nuR_A4 + lcd_nuR_A4 + ls_nuR_A4 + pwm_nuR_A4 + sg7_nuR_A4 + tmp_nuR_A4 + utt_nuR_A4 + vga_nuR_A4 + dly2_nuR_A4 + gry2_nuR_A4 + sg7_2_nuR_A4 + gry3_nuR_A4)
nuR_EL	= np.array(dly_nuR_EL + gry_nuR_EL + i2c_nuR_EL + lcd_nuR_EL + ls_nuR_EL + pwm_nuR_EL + sg7_nuR_EL + tmp_nuR_EL + utt_nuR_EL + vga_nuR_EL + dly2_nuR_EL + gry2_nuR_EL + sg7_2_nuR_EL + gry3_nuR_EL)
nuR_M	= np.array(dly_nuR_M  +  gry_nuR_M + i2c_nuR_M  + lcd_nuR_M  + ls_nuR_M  + pwm_nuR_M  + sg7_nuR_M  + tmp_nuR_M  + utt_nuR_M  + vga_nuR_M  + dly2_nuR_M  + gry2_nuR_M  + sg7_2_nuR_M  + gry3_nuR_M)


nuR_bst  = np.minimum(np.minimum(np.minimum(nuR, nuR_A1), nuR_A2), nuR_A4)

#----------------------------------------------------------------------------------
#							 STATISTICS
#----------------------------------------------------------------------------------

def stat1():
	print(f"{len(nuX[nuX<time_out_small])/len(nuX)*100}% tasks completed by nuXmv.")
	print(f"{len(ABC[ABC<time_out_small])/len(ABC)*100}% tasks completed by ABC.")
	print(f"{len(indX[indX<time_out_small])/len(indX)*100}% tasks completed by indX.")
	print(f"{len(nuR[nuR<time_out_small])/len(nuR)*100}% tasks completed by our tool (8, 5).")
	print(f"{len(nnT[nnT<time_out_small])/len(nuR)*100}% tasks trained successfully (verification might have timeout).")
	print(f"{len(nuR_bst[nuR_bst<time_out_small])/len(nuR_bst)*100}% tasks completed by our tool (best).\n")

	print(f"{len(nuX[nuX<time_out_small])} of {len(nuR)} tasks completed by nuXmv.")
	print(f"{len(ABC[ABC<time_out_small])} of {len(nuR)}tasks completed by ABC.")
	print(f"{len(indX[indX<time_out_small])} of {len(nuR)}tasks completed by indX.")
	print(f"{len(nuR[nuR<time_out_small])} of {len(nuR)} tasks completed by our tool (8, 5).")
	print(f"{len(nuR_bst[nuR_bst<time_out_small])} of {len(nuR)} tasks completed by our tool (best).\n")

def stat2():
	print(f"{len(nuR[nuR<time_out_small])/len(nuX)*100}% tasks completed by our (Default).")
	print(f"{len(nuR_A1[nuR_A1<time_out_small])/len(nuR_A1)*100}% tasks completed by our (3,2).")
	print(f"{len(nuR_A2[nuR_A2<time_out_small])/len(nuR_A2)*100}% tasks completed our (5,3).")
	print(f"{len(nuR_A4[nuR_A4<time_out_small])/len(nuR_A4)*100}% tasks completed by our (15,8).")
	print(f"{len(nuR_EL[nuR_EL<time_out_small])/len(nuR_EL)*100}% tasks completed by our (Ext.L).")
	print(f"{len(nuR_M[nuR_M<time_out_small])/len(nuR_M)*100}% tasks completed by our (Mono).")

	print(f"{len(nuR_bst[nuR_bst<time_out_small])/len(nuR_bst)*100}% tasks completed by our tool (best).\n")


def stat3():
	s_nuX = []
	s_abc = []
	s_indX = []
	for tim in range(0, 60*60*5+1):
		s_nuX.append(np.sum(nuR < tim) - np.sum(nuX < tim))
		s_abc.append(np.sum(nuR < tim) - np.sum(ABC < tim))
		s_indX.append(np.sum(nuR < tim) - np.sum(indX < tim))

	s_nuX = np.array(s_nuX)/len(nuX)
	s_abc = np.array(s_abc)/len(nuX)
	s_indX = np.array(s_indX)/len(nuX)
	print(f" At any given time we (8,5) do {np.mean(s_nuX)*100} % more task than nuXmv with a standard dev of +- {np.std(s_nuX)*100} %.")
	print(f" At any given time we (8,5) do {np.mean(s_abc)*100} % more task than ABC with a standard dev of +- {np.std(s_abc)*100} %.")
	print(f" At any given time we (8,5) do {np.mean(s_indX)*100} % more task than ABC with a standard dev of +- {np.std(s_indX)*100} %.\n")

def stat4():
	ABC_a = ABC.copy()
	nuX_a = nuX.copy()
	nuR_a = nuR.copy()
	nuR_bst_a = nuR_bst.copy()
	ABC_a.sort()
	nuX_a.sort()
	nuR_a.sort()
	nuR_bst_a.sort()
	print(f"60% tasks nuXmv {np.sum(nuX_a[: int(0.6031*(len(nuX_a))) ])/60/60} hrs and our (8,5) tool {np.sum(nuR_a[: int(0.6031*(len(nuX_a))) ])/60/60} hrs")
	print(f"29% tasks ABC {np.sum(ABC_a[: int(0.2938*(len(ABC_a))) ])/60/60} hrs and our tool {np.sum(nuR_a[: int(0.30*(len(ABC_a))) ])/60/60} hrs\n")

	print(f"60% tasks nuXmv {np.sum(nuX_a[: int(0.60*(len(nuX_a))) ])/60/60} hrs and our (best) tool {np.sum(nuR_bst_a[: int(0.6031*(len(nuX_a))) ])/60/60} hrs")
	print(f"29% tasks ABC {np.sum(ABC_a[: int(0.2938*(len(ABC_a))) ])/60/60} hrs and our (best) tool {np.sum(nuR_bst_a[: int(0.2938*(len(ABC_a))) ])/60/60} hrs")


def stat5(x, y):
	print(f"In {np.sum(y<=60)/len(x)*100} % tasks trained under 60 sec")
	print(f"In {np.sum(x>y)/np.sum(x+y<time_out_time)*100} % tasks training was faster than SMT-check")
	print(f"In {np.sum(x>y*10)/len(x+y<time_out_time)*100} % tasks training was 10x faster than SMT-check")
	print(f"In {np.sum(x>y*100)/len(x+y<time_out_time)*100} % tasks training was 100x faster than SMT-check")

def stat6():
	ABC_a = ABC.copy()
	nuX_a = nuX.copy()
	nuR_a = nuR.copy()
	nuR_bst_a = nuR_bst.copy()
	ABC_a.sort()
	nuX_a.sort()
	nuR_a.sort()
	nuR_bst_a.sort()
	print(f"60%-th completed tasks nuXmv {np.sum(nuX_a[int(0.6031*(len(nuX_a)))])/60/60} hrs and our tool (8,5) {nuR_a[int(0.6031*(len(nuR_a)))]} sec")
	print(f"29%-th completed tasks ABC {np.sum(ABC_a[int(0.30*(len(ABC_a))) ])/60/60}  hrs and our tool (8,5) {nuR_a[int(0.2938*(len(nuR_a)))]} sec\n")

	print(f"60%-th completed tasks nuXmv {np.sum(nuX_a[int(0.6031*(len(nuX_a)))])/60/60} hrs and our tool (best) {nuR_bst_a[int(0.6031*(len(nuR_bst_a)))]} sec")
	print(f"29%-th completed tasks ABC {np.sum(ABC_a[int(0.2939*(len(ABC_a))) ])/60/60}  hrs and our tool (best) {nuR_bst_a[int(0.2939*(len(nuR_bst_a)))]} sec")

def stat7(x, y, z):	
	print(f"In {np.sum(x>y)/len(x)*100} % tasks we (8,5) are faster than the best of nuxmv and ABC")
	print(f"In {np.sum(x>y*10)/len(x)*100} % tasks we (8,5) are 10X faster than the best of nuxmv and ABC")
	print(f"In {np.sum(x>y*100)/len(x)*100} % tasks we (8,5) are 100X faster than the best of nuxmv and ABC\n")

	print(f"In {np.sum(y>=time_out_time)/len(x)*100} % tasks we (8,5) Time out.")
	print(f"In {np.sum(x>=time_out_time)/len(y)*100} % tasks best of nuxmv and ABC Time out.")
	
def stat8(x, y, z):
	print(f"In {np.sum(x>y)/len(x)*100} % tasks we (8,5) are faster than industry tool X")
	print(f"In {np.sum(x>y*10)/len(x)*100} % tasks we (8,5) are 10X faster than industry tool X")
	print(f"In {np.sum(x>y*100)/len(x)*100} % tasks we (8,5) are 100X faster than industry tool X\n")

def stat9():
	print(f"In {np.sum((nuR_A1<nuR)*(nuR_A1<time_out_time))/np.sum(nuR_A1<time_out_time)*100} % of tasks that our (3, 2) completes it is FASTER than our default")
	print(f"In {np.sum((nuR_A2<nuR)*(nuR_A2<time_out_time))/np.sum(nuR_A2<time_out_time)*100} % of tasks that our (5, 3) completes it is FASTER than our default")
	print(f"In {np.sum(nuR_A2<nuR_A1)/len(nuR)*100} % of tasks our (5, 3) is faster than our (3, 2)")

	print(f"In {np.sum((nuR<nuR_A4)*(nuR_A4<time_out_time))/np.sum(nuR_A4<time_out_time)*100} % of tasks that our (15, 8) completes it is SLOWER than our default")
	print(f"In {np.sum((nuR_EL<nuR)*(nuR_EL<time_out_time))/np.sum(nuR_EL<time_out_time)*100} % of tasks that our (Extra Layer) completes it is FASTER than our default")
	print(f"In {np.sum((nuR_M<nuR)*(nuR_M<time_out_time))/np.sum(nuR_M<time_out_time)*100} % of tasks that our (Monolithic) completes it is FASTER than our default")
	print(f"In {np.sum((nuR>=time_out_time)*(nuR_A1>=time_out_time)*(nuR_A2>=time_out_time)*(nuR_A4>=time_out_time)*(nuR_EL>=time_out_time)*(nuR_M>=time_out_time))} tasks all our ablation study configurations failed.")

	nuR_allabs = np.minimum(np.minimum(nuR_bst, nuR_EL), nuR_M)

	print(f"{len(nuR_allabs[nuR_allabs<time_out_small])/len(nuR_allabs)*100}% tasks completed by our all our ablation study combined.\n")


#----------------------------------------------------------------------------------
#							 PLOTS-EXP
#----------------------------------------------------------------------------------

def matrixPlot(x_a, y_a, x_name, y_name, sizes):
	x_a[x_a >= time_out_small] = time_out_small * 1.35 # To make the time outs appear on the right of the 5hr line
	y_a[y_a >= time_out_small] = time_out_small * 1.35

	x = x_a.copy()
	y = y_a.copy()
	sorted_tuples = sorted(zip(sizes, x, y), reverse=True)
	sizes, x, y = zip(*sorted_tuples)
	
	# Create scatter plot
	plt.figure(figsize=(10, 8))
	x_cnt = np.arange(18001)
	y_1x = 1 * x_cnt
	plt.plot(x_cnt, y_1x, color='red', linewidth=1, linestyle='dashed')
	y_10x = 0.1 * x_cnt
	plt.plot(x_cnt, y_10x, color='red', linewidth=1, linestyle='dashed')
	y_100x = 0.01 * x_cnt
	plt.plot(x_cnt, y_100x, color='red', linewidth=1, linestyle='dashed')
	plt.scatter(x, y, s=np.log(sizes)*17, alpha=0.8, c=sizes, norm = "log", edgecolors='black', linewidths=1.5)

	formatter = ScalarFormatter()
	formatter.set_scientific(True)
	formatter.set_powerlimits((-1,1))
	
	# Add grid and adjust axis limits
	plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
	plt.xlim(1, time_out_small*1.757)
	plt.ylim(1, time_out_small*1.757)
	plt.yscale('log')
	plt.xscale('log')
	
	# Add colorbar for better understanding of point sizes
	cbar = plt.colorbar( orientation='vertical').ax.tick_params(labelsize=16)
	
	plt.gca().add_patch(plt.Rectangle((time_out_small*.97, 0), 100000, 900000, color='red', linestyle='dotted', alpha=0.1, linewidth=1))
	plt.gca().add_patch(plt.Rectangle((0, time_out_small*.97), 900000, 100000, color='red',linestyle='dotted', alpha=0.1, linewidth=1))
	plt.plot([18000, 18000], [18000,0], color='black', linewidth=1)
	plt.plot([0,18000], [18000, 18000],  color='black', linewidth=1)

	plt.text(1.2, time_out_small*1.1, 'Out of Time', fontsize=19, color='red')
	cst_value = [ 3600,  7200, 10800, 14400, 18000]  # 10 ticks spaced evenly between -1 and 1
	cst_label = ['1h', '2h', '3h', '4h', '5h']
	cst_value = [5, 15, 60,   300, 900,  3600, 18000]  # 10 ticks spaced evenly between -1 and 1
	cst_label = ['5s', '15s', '1m', '5m', '15m', '1h', '5h']
	plt.yticks(cst_value, cst_label)
	plt.xticks(cst_value, cst_label)
	plt.tick_params(axis='both', which='major', labelsize=19)
		
	plt.show()

def landslidePlot(nuX_a, ABC_a, nuR_a, nnT_a):
	nuX = nuX_a.copy()
	ABC = ABC_a.copy()
	nuR = nuR_a.copy()
	nnT = nnT_a.copy()

	n_values = np.arange(1, len(nuX) + 1)/(len(nuX))*100
	sum_tool = np.add(nuR,nnT) #np.add(np.add(nuX,ABC),nuR)
	sorted_tuples = sorted(zip(sum_tool,nuX, ABC, nuR, nnT))
	sum_tool, nuX, ABC, nuR, nnT = zip(*sorted_tuples)

	formatter = ScalarFormatter()
	formatter.set_scientific(True)
	formatter.set_powerlimits((-1,1))
	
	plt.plot(n_values, np.array(nnT)/np.array(nuR), color='#0E2E16')
	plt.plot(n_values, nuR, color='#0E2E16')
	plt.plot(n_values, nnT, color='#184E25')
	plt.xlim(1, round(np.sum(np.array(nnT)< time_out_small)/(len(nuX))*100))
	plt.ylim(1, 18000)
		
	plt.fill_between(n_values, nuR, nnT, color='#0E2E16', alpha=0.2, hatch='o')
	
	plt.fill_between(n_values[:-10], nnT[:-10], color='#184E25', alpha=0.3)
	plt.fill_between(n_values[-11:], nnT[-11:], color='#041809', alpha=0.3 )
	plt.tick_params(axis='both', which='major', labelsize=16)

	
	cst_value = [20, 40, 60, 80, round(np.sum(np.array(nuR)< time_out_small)/(len(nuR))*100)]  # 10 ticks spaced evenly between -1 and 1
	cst_label = [ str(cst)+"%" for cst in cst_value]
	plt.xticks(cst_value, cst_label)
	plt.gca().set_yscale('log')
	cst_value = [5, 15, 60,   300, 900,  3600, 18000]  # 10 ticks spaced evenly between -1 and 1
	cst_label = ['5s', '15s', '1m', '5m', '15m', '1h', '5h']
	plt.yticks(cst_value, cst_label)
	plt.grid(linestyle='--', alpha=1)
	plt.show()


def cactusPlot(nuX, ABC, nuR, nnT, indX, nuR_bst):
	method1_times = nuX.copy()
	method2_times = ABC.copy()
	method3_times = nuR.copy()
	method4_times = nnT.copy()
	method5_times = indX.copy()
	method6_times = nuR_bst.copy()
	
	
	method1_times.sort()
	method2_times.sort()
	method3_times.sort()
	method4_times.sort()
	method5_times.sort()
	method6_times.sort()

	num_tasks = len(method1_times)
	tasks_completed = np.arange(1, num_tasks + 1)/num_tasks*100
	formatter = ScalarFormatter()
	formatter.set_scientific(True)
	formatter.set_powerlimits((-1,1))
	plt.figure(figsize=(10, 6))

	plt.step(tasks_completed, method2_times, label='ABC', color='red', linewidth=3, where='post', alpha=0.7)
	plt.step(tasks_completed, method1_times, label='nuXmv', color='blue', linewidth=3, where='post', alpha=0.7)
	plt.step(tasks_completed, method5_times, label='ind-X', color='black', linewidth=3, where='post', alpha=0.7)
	plt.step(tasks_completed, method3_times, label='our (5,8)', color='green', linewidth=3, where='post', alpha=0.7)
	plt.step(tasks_completed, method6_times, label='our best', color='purple', linewidth=3, where='post', alpha=0.7)

	plt.ylim(1, 18000)
	plt.xlim(1, 100)
	#plt.yscale('function', functions = (lambda x : x**(1/3), lambda x : x**(3)))
	plt.yscale('log')

	ax = plt.gca()
	ax.yaxis.set_major_formatter(formatter)
	plt.tick_params(axis='both', which='major', labelsize=16)

	cst_value = [  3600,  7200, 10800, 14400, 18000]  # 10 ticks spaced evenly between -1 and 1
	cst_label = [ '1h', '2h', '3h', '4h', '5h']
	cst_value = [5, 15, 60,   300, 900,  3600, 18000]  # 10 ticks spaced evenly between -1 and 1
	cst_label = ['5s', '15s', '1m', '5m', '15m', '1h', '5h']
	plt.yticks(cst_value, cst_label)
	cst_value = [20, 40, 60, 80, 100]  # 10 ticks spaced evenly between -1 and 1
	cst_label = [ str(cst)+"%" for cst in cst_value]
	plt.xticks(cst_value, cst_label)
	plt.legend()
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.legend(fontsize=18)
	plt.show()



#----------------------------------------------------------------------------------
#							 PLOTS-BENCHMARKS
#----------------------------------------------------------------------------------


def whisker_plot(data, x_labs, if_lgc):
	fig, ax = plt.subplots()
	#breakpoint()
	for i,sv in enumerate(data):
		for j,w in enumerate(sv):
			#if j == 0:
			#	continue 
			clr = "purple" if j == 0 else ("red" if j==1 else ("blue" if j==2 else ("black" if j==3 else "green")))
			ax.plot([i+j/6-0.2, i+j/6-0.2], w, color=clr, linewidth=3, marker = 'o')  # marker = "o"
	ax.set_xticks(range(len(x_labs)))
	ax.set_xticklabels(x_labs)
	if if_lgc:
		plt.ylim(7*10**1, 10**4)
	plt.gca().set_yscale('log')
	colors = ['purple', 'red', 'blue', 'black', 'green']
	labels = ['All Task', 'ABC', 'nuXmv', 'indX', 'our']
	handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
	plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(colors), fontsize=19)
	plt.tick_params(axis='both', which='major', labelsize=19)
	plt.show()

# Range of LGC for four categories (AllTasks, ABC, nuXmv, indX, our)
Delay_lgc= (np.array([[312 , 484 ], [312 , 439 ], [312 , 384 ], [312 , 430 ], [312 , 484 ]]) - 0) / (1) * 1
Gray_lgc = (np.array([[142 , 282 ], [142 , 184 ], [142 , 240 ], [142 , 184 ], [142 , 282 ]]) - 0) / (1) * 1
i2cS_lgc = (np.array([[775 , 1543], [775 , 931 ], [775 , 1203], [775 , 1543], [775 , 1423]]) - 0) / (1) * 1
LCD_lgc =  (np.array([[3098, 5383], [3098, 3638], [3098, 4960], [3098, 5383], [3098, 5383]]) - 0) / (1) * 1
LS_lgc =   (np.array([[510,  873 ], [510,  539 ], [510,  674 ], [510,  873 ], [510 , 832 ]]) - 0) / (1) * 1
PWM_lgc =  (np.array([[100 , 155 ], [100 , 105 ], [100 , 130 ], [100 , 155 ], [100 , 150 ]]) - 0) / (1) * 1
sg7_lgc =  (np.array([[242 , 375 ], [242 , 262 ], [242 , 315 ], [242 , 262 ], [242 , 375 ]]) - 0) / (1) * 1
Tmcp_lgc = (np.array([[632 , 953 ], [632 , 778 ], [632 , 848 ], [632 , 953 ], [632 , 953 ]]) - 0) / (1) * 1
UARTt_lgc= (np.array([[289 , 661 ], [289 , 661 ], [289 , 661 ], [289 , 661 ], [289 , 661 ]]) - 0) / (1) * 1
VGA_lgc =  (np.array([[1020, 1345], [1020, 1156], [1020, 1141], [1020, 1345], [0  ,    0]]) - 0) / (1) * 1
data_lgc = [LS_lgc, LCD_lgc, Tmcp_lgc, i2cS_lgc, sg7_lgc, PWM_lgc, VGA_lgc, UARTt_lgc, Delay_lgc, Gray_lgc]


# Range of SSS for four categories (AllTasks, ABC, nuXmv, indX, our)
Delay_sss= (np.array([[2**3*750   , 2**3*400000 ], [2**3*750   , 2**3*2500  ], [2**3*750   , 2**3*25000 ], [2**3*750   , 2**3*7500   ], [2**3*750   , 2**3*400000 ]]) - 0) / (1) * 1
Gray_sss = (np.array([[2**9       , 2**19       ], [2**9       , 2**13      ], [2**9       , 2**16      ], [2**9       , 2**14       ], [2**9       , 2**19       ]]) - 0) / (1) * 1
i2cS_sss = (np.array([[2**3*500   , 2**3*280000 ], [2**3*500   , 2**3*2000  ], [2**3*500   , 2**3*16000 ], [2**3*500   , 2**3*280000 ], [2**3*500   , 2^3*70000   ]]) - 0) / (1) * 1
LCD_sss =  (np.array([[2**12*500  , 2**12*180000], [2**12*500  , 2**12*1500 ], [2**12*500  , 2**12*15000], [2**12*500  , 2**12*180000], [2**12*500  , 2**12*180000 ]]) - 0) / (1) * 1
LS_sss =   (np.array([[2**2*750   , 2**2*400000 ], [2**2*750   , 2**2*1250  ], [2**2*750   , 2**2*15000 ], [2**2*750   , 2**2*400000 ], [2**2*750   , 2**2*200000 ]]) - 0) / (1) * 1
PWM_sss =  (np.array([[2**11      , 2**22       ], [2**11      , 2**12      ], [2**11      , 2**17      ], [2**11      , 2**22       ], [2**11      , 2**22       ]]) - 0) / (1) * 1
sg7_sss =  (np.array([[2**7*250   , 2**7*160000 ], [2**7*250   , 2**7*1000  ], [2**7*250   , 2**7*2500  ], [2**7*250   , 2**7*1000   ], [2**7*250   , 2**7*160000 ]]) - 0) / (1) * 1
Tmcp_sss = (np.array([[2**33*30   , 2**33*288000], [2**33*30   , 2**33*2400 ], [2**33*30   , 2**33*15000], [2**33*30   , 2**33*288000], [2**33*30   , 2**33*288000 ]]) - 0) / (1) * 1
UARTt_sss= (np.array([[2**10      , 2**23       ], [2**10      , 2**23      ], [2**10      , 2**23      ], [2**10      , 2**23       ], [2**10      , 2**23       ]]) - 0) / (1) * 1
VGA_sss =  (np.array([[(94*1)**2*2**5 , (94*16)**2*2**9], [(94*1)**2*2**5, (94*4)**2*2**6], [(94*1)**2*2**5, (94*3)**2*2**6], [(94*1)**2*2**5 , (94*16)**2*2**9],      [0 , 0]]) - 0) / (1) * 1
data_sss = [LS_sss, LCD_sss, Tmcp_sss, i2cS_sss, sg7_sss, PWM_sss, VGA_sss, UARTt_sss, Delay_sss, Gray_sss]
x_labs = ['LS', 'LCD' , 'Tmcp', 'i2cS', '7-seg', 'PWM', 'VGA', 'UARTt', 'Delay', 'Gray']

#----------------------------------------------------------------------------------
#							 PLOT BIG TABLE
#----------------------------------------------------------------------------------


while(True):
	print("1)Fig.6a 2)Fig.6b 3)Fig.6c 4)Fig.5a 5)Fig.5b\n10)Stat1 11)Stat2 12)Stat3 13)Stat4 14)Stat5 15)Stat6 16)Stat7 17)Stat8 18)Stat9")
	ch = int(input("Enter your choice: "))
	if(ch == 1):
		cactusPlot(nuX, ABC, nuR, nnT, indX, nuR_bst)
	elif(ch == 2):
		matrixPlot(np.minimum(ABC, nuX), nuR, 'ABC-nuXmv', 'NuR', sss)
	elif(ch == 3):
		landslidePlot(nuX, ABC, nuR, nnT)
	elif(ch == 4):
		whisker_plot(data_sss, x_labs, if_lgc = False)
	elif(ch == 5):
		whisker_plot(data_lgc, x_labs, if_lgc = True)
	elif(ch == 10):
		stat1()
	elif(ch == 11):
		stat2()
	elif(ch == 12):
		stat3()
	elif(ch == 13):
		stat4()
	elif(ch == 14):
		stat5(nuR-nnT, nnT)
	elif(ch == 15):
		stat6()
	elif(ch == 16):
		stat7(np.minimum(ABC, nuX), nuR, nuR_bst)
	elif(ch == 17):
		stat8(indX, nuR, nuR_bst)
	elif(ch == 18):
		stat9()

	else:
		print("Please enter an number between 0 to 5 or 10 to 18")



