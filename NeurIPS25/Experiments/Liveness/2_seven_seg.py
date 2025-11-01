import sys
import os
import numpy as np
import itertools
sys.path.append('../')
import run_exp
sys.path.append('../../')
import Tools.neuralmc.cav_nuR as nuR

N_lims = [250, 500, 750, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 40000, 80000, 160000]
CBITSs = [0]*len(N_lims) #unused beside size

specTXT = "FG !rst -> (GF ds & GF !ds)"
module_name = "SEVEN"
file_name = "seven_seg"

LTLSpec = "LTLSPEC F G (Verilog.SEVEN.rst = FALSE) -> ( (G F (Verilog.SEVEN.digit_select = FALSE)) & (G F (Verilog.SEVEN.digit_select = TRUE)))"
SVSpec = "(@(posedge clk) s_eventually !rst implies ((s_eventually digit_select) and (s_eventually !digit_select)))"
range_vals_list = ['N', 'N', 'N', 'N', 'N', 'N']

scale = 1
size = [2] #[2]
gap = 1e-3 # this is the gap of the sign activation function 
F_prec = 5
bits = 50
start_ex = 0

Ps = [N_lim*2 for N_lim in N_lims[:7]] + [N_lim*4 for N_lim in N_lims[7:]]
Ms = [N_lim*2 for N_lim in N_lims]

is_acc = [0, 1, 1]
is_init = [1, 0, 0]
init_samp = [(0, np.array([float(0), float(1)]))]

def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    cases = []
    if q_cur == 0 and q_nex == 0:
        cases.append([])
    
    elif q_cur == 0 and q_nex == 1:
        cases.append([nuR.BUnSet(curr_vars, 'digit_select', 1, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])
    elif q_cur == 1 and q_nex == 1:
        cases.append([nuR.BUnSet(curr_vars, 'digit_select', 1, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])
    
    elif q_cur == 0 and q_nex == 2:
        cases.append([nuR.BUnSet(curr_vars, 'digit_select', 0, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])
    elif q_cur == 2 and q_nex == 2:
        cases.append([nuR.BUnSet(curr_vars, 'digit_select', 0, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])
    
    return cases

run_exp.exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms)


N_lims = [250, 500, 750, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 40000, 80000, 160000]
specTXT = "FG !rst -> (GF ds & GF !ds)"
module_name = "SEVEN"

scale = 1
size = [2, 1]
gap = 1e-3 # this is the gap of the sign activation function 
F_prec = 5
bits = 100

import math
import time
import itertools
import sys
import numpy as np


print("1)ABC 2)nuXmv 3)our")
ch = int(input("Enter your choice: "))
sys.path.append('../../')
if ch == 1:
	import Tools.abc_mc.abc_run as abc_run
elif ch == 2:
	import Tools.nuXmv.nuxmv_run as nuxmv_run
else:
	import Tools.neuralmc.cav_nuR as nuR

if ch in [1, 2]:
	for dut_i in range(len(N_lims)):
		name = f"seven_seg_{dut_i+1}"
		N_lim = N_lims[dut_i]
		idtxt = f"{name} ({specTXT}) {N_lim}"
		print(idtxt)
		LTLSpec = "LTLSPEC F G (Verilog.SEVEN.rst = FALSE) -> ( (G F (Verilog.SEVEN.digit_select = FALSE)) & (G F (Verilog.SEVEN.digit_select = TRUE)))"
		if ch == 1:
			abc_run.runABC(name, module_name, LTLSpec, idtxt)
		else: 
			nuxmv_run.runNuXmv(name, module_name, LTLSpec, idtxt)
		sys.stdout.flush()
		continue
	exit()

is_acc = [0, 1, 1]
is_init = [1, 0, 0]
init_samp = [(0, np.array([float(0), float(1)]))]

def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    cases = []
    if q_cur == 0 and q_nex == 0:
        cases.append([])#nuR.BLessThan(V0, V1, ctx)])
    
    elif q_cur == 0 and q_nex == 1:
        cases.append([nuR.BUnSet(curr_vars, 'digit_select', 1, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])#,  BLessThanE(V0, V1, s*.9, ctx)])
    elif q_cur == 1 and q_nex == 1:
        cases.append([nuR.BUnSet(curr_vars, 'digit_select', 1, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])#,  BLessThanE(V0, V1, s*.9, ctx)])
    
    elif q_cur == 0 and q_nex == 2:
        cases.append([nuR.BUnSet(curr_vars, 'digit_select', 0, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])#,  BLessThanE(V0, V1, s*.9, ctx)])
    elif q_cur == 2 and q_nex == 2:
        cases.append([nuR.BUnSet(curr_vars, 'digit_select', 0, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])#,  BLessThanE(V0, V1, s*.9, ctx)])
    
    return cases	

gu_times, bw_times, total_times = [], [], []
for dut_i in range(len(N_lims)):
	begin = time.time()
	N_lim = N_lims[dut_i]
	P = N_lim*4
	M = N_lim*2
	kappa = N_lim
	hyperparameters = (scale, P, size, gap, M, kappa)
	name = f"seven_seg_{dut_i+1}"
	module_name = "SEVEN"
	idtxt = f"{name} ({specTXT}) {N_lim}"
	print(f"\t\t\t\t {idtxt}\n\t\t\t\t")
	
	#nuR.getLGC_SSS(name, module_name, idtxt)
	#exit()
	
	range_vals = iter(['N', 'N', 'N', 'N', 'N', 'N'])
	state_vars, inp_out_vars = nuR.readForVars(name, module_name, range_vals)

	bw_obj, curr_vars, next_vars, non_state_vars = nuR.verilogSMT(name, module_name, state_vars, bits, inp_out_vars)
	ctx = state_vars, inp_out_vars, bw_obj, bits
	gu_time, bw_time = nuR.runExperiment(name, hyperparameters, bw_obj, curr_vars, next_vars, non_state_vars, spec_automata, ctx, F_prec, bits, is_acc, init_samp, is_init)
	end = time.time()
	print(f"BITS ---------->>>>>>>>> {bits} {idtxt}")
	print(f'Gurobi Time: {gu_time}; Bitwuzla Time: {bw_time}')
	print(f"Total Time: {end - begin}\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
	gu_times.append(gu_time)
	bw_times.append(bw_time)
	total_times.append(end-begin)
	with open(f"data.txt", "a") as file:
		file.write(f"BITS ---------->>>>>>>>> {bits} {idtxt}\n")
		file.write(f'Gurobi Time: {gu_time}; Bitwuzla Time: {bw_time}\n')
		file.write(f"Total Time: {end - begin}\nG:[{gu_times}]\nB:[{bw_times}]\nT:[{total_times}]\n\n\n\n")
	print("Start 2 sec Sleep")
	time.sleep(2)  # Pause for 2 seconds