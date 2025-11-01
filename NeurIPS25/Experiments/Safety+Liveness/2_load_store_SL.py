import sys
import os
import numpy as np
import itertools
sys.path.append('../')
import run_exp
sys.path.append('../../')
import Tools.neuralmc.cav_nuR as nuR

N_lims = [750, 1250, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 50000, 100000, 200000, 400000]
CBITSs = [10, 11, 12, 13, 13, 14, 14, 14, 15, 15, 15, 15, 16, 17, 18, 19]
specTXT = "G !rst -> X G (m -> (!s U X !m))"
module_name = "Load_Store"
file_name = "load_store"
LTLSpec = "LTLSPEC G (Verilog.Load_Store.rst = FALSE) -> X G  (((Verilog.Load_Store.m = TRUE) -> (Verilog.Load_Store.sig = FALSE U (X Verilog.Load_Store.m = FALSE))))"
SVSpec = "(@(posedge clk) (always !rst) implies s_nexttime always (m implies (!sig s_until s_nexttime !m)))"
range_vals_list = ['N','N','N', 'N']

scale = 1
size = [2] #[2, 1]
gap = 1e-3 # this is the gap of the sign activation function 
F_prec = 5
bits = 50
start_ex = 0

Ps = [10 for N_lim in N_lims]
Ms = [N_lim*2 for N_lim in N_lims]

is_acc  = [0, 0, 1, 0, 1]
is_init = [1, 0, 0, 0, 0]
init_samp = [(0, np.array([float(0), float(1)]))]
def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    cases = []
    if q_cur == 0 and q_nex == 1:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx)])
    
    elif q_cur == 1 and q_nex == 1:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx)])
    elif q_cur == 1 and q_nex == 2:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.BUnSet(non_state, 'sig', 1, ctx), nuR.Bset(curr_vars, 'm', 1, ctx)])
    elif q_cur == 1 and q_nex == 3:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.Bset(non_state, 'sig', 1, ctx), nuR.Bset(curr_vars, 'm', 1, ctx)])
        
    elif q_cur == 2 and q_nex == 2:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.BUnSet(non_state, 'sig', 1, ctx), nuR.Bset(curr_vars, 'm', 1, ctx)])
    elif q_cur == 2 and q_nex == 3:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.Bset(non_state, 'sig', 1, ctx), nuR.Bset(curr_vars, 'm', 1, ctx)])
        
    elif q_cur == 3 and q_nex == 4:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.Bset(curr_vars, 'm', 1, ctx)])
        
    if q_cur == 4 and q_nex == 4:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx)])
    return cases

run_exp.exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms)

