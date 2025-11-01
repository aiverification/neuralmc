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

specTXT = "G !rst -> G (((!ds & X !ds) | (ds & X ds)) U X s)"
module_name = "SEVEN"
file_name = "seven_seg"

LTLSpec = "LTLSPEC G (Verilog.SEVEN.rst = FALSE) -> G  ( ((Verilog.SEVEN.digit_select = FALSE & X Verilog.SEVEN.digit_select = FALSE ) | (Verilog.SEVEN.digit_select = TRUE & X Verilog.SEVEN.digit_select = TRUE )) U X Verilog.SEVEN.sig = TRUE) "

SVSpec = "(@(posedge clk) (always !rst) implies always ((digit_select iff s_nexttime digit_select) s_until s_nexttime sig))"
range_vals_list = ['N', 'N', 'N', 'N', 'N', 'N']

scale = 1
size = [2] #[2, 1]
gap = 1e-3 # this is the gap of the sign activation function 
F_prec = 5
bits = 50
start_ex = 0

Ps = [N_lim*2 for N_lim in N_lims[:7]] + [N_lim*4 for N_lim in N_lims[7:]]
Ms = [N_lim*2 for N_lim in N_lims]

is_acc  = [0, 0, 1, 0, 1]
is_init = [1, 0, 0, 0, 0]
init_samp = [(0, np.array([float(0), float(1)]))]

def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    cases = []
    if q_cur == 0 and q_nex == 0:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx)])
        
    
    elif q_cur == 0 and q_nex == 2:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx)])
    elif q_cur == 0 and q_nex == 1:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.BUnSet(curr_vars, 'digit_select', 1, ctx)])
    elif q_cur == 0 and q_nex == 3:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.Bset(curr_vars, 'digit_select', 1, ctx)])
        
        
    elif q_cur == 2 and q_nex == 2:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.BUnSet(non_state, 'sig', 1, ctx)])
    elif q_cur == 2 and q_nex == 1:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.BUnSet(curr_vars, 'digit_select', 1, ctx), nuR.BUnSet(non_state, 'sig', 1, ctx)])
    elif q_cur == 2 and q_nex == 3:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.Bset(curr_vars, 'digit_select', 1, ctx), nuR.BUnSet(non_state, 'sig', 1, ctx)])
        
    
    elif q_cur == 3 and q_nex == 4:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.BUnSet(curr_vars, 'digit_select', 1, ctx), nuR.BUnSet(non_state, 'sig', 1, ctx)])
    elif q_cur == 1 and q_nex == 4:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.Bset(curr_vars, 'digit_select', 1, ctx), nuR.BUnSet(non_state, 'sig', 1, ctx)])
        
    if q_cur == 4 and q_nex == 4:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx)])
        
    
    return cases

run_exp.exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms)

