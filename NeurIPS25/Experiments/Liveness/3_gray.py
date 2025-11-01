import sys
import os
import numpy as np
import itertools
sys.path.append('../')
import run_exp
sys.path.append('../../')
import Tools.neuralmc.cav_nuR as nuR

N_lims = [2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18]
CBITSs = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
specTXT = "FG !rst -> (GF sig & GF !sig)"
module_name = "GRAY"
file_name = "gray"
LTLSpec = "LTLSPEC F G (Verilog.GRAY.rst = FALSE) -> G F (Verilog.GRAY.sig = TRUE & F Verilog.GRAY.sig = FALSE)"
SVSpec = "(@(posedge clk) s_eventually !rst implies ((s_eventually sig) and (s_eventually !sig)))"
range_vals_list = ['N',  'N', 'N', 'N', 'N']

start_ex = 0

scale = 1
size = [1] #[1]
gap = 1e-3 # this is the gap of the sign activation function 
F_prec = 5
bits = 100

Ps = [N_lim for N_lim in N_lims]
Ms = [N_lim*2 for N_lim in N_lims]

is_acc = [0, 1, 1]
is_init = [1, 0, 0]
init_samp = [(0, np.array([float(0)]))]

def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    cases = []
    if q_cur == 0 and q_nex == 0:
        cases.append([])
    
    elif q_cur == 0 and q_nex == 1:
        cases.append([nuR.BUnSet(non_state, 'sig', 1, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])
    elif q_cur == 1 and q_nex == 1:
        cases.append([nuR.BUnSet(non_state, 'sig', 1, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])
    
    elif q_cur == 0 and q_nex == 2:
        cases.append([nuR.BUnSet(non_state, 'sig', 0, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])
    elif q_cur == 2 and q_nex == 2:
        cases.append([nuR.BUnSet(non_state, 'sig', 0, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])
    
    return cases


run_exp.exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms)

