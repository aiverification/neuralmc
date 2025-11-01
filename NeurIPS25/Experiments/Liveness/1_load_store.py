
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
specTXT = "FG !rst -> GF (vol = N)"
module_name = "Load_Store"
file_name = "load_store"
LTLSpec = "LTLSPEC F G (Verilog.Load_Store.rst = FALSE) -> G F (Verilog.Load_Store.sig = TRUE)"
SVSpec = "(@(posedge clk) s_eventually !rst -> sig)"
range_vals_list = ['N','N','N', 'N']

scale = 1
size = [2] #[2, 2]
gap = 1e-3 # this is the gap of the sign activation function 
F_prec = 5
bits = 50
start_ex = 0

Ps = [N_lim for N_lim in N_lims]
Ms = [N_lim*2 for N_lim in N_lims]

is_acc = [0, 1]
is_init = [1, 0]
init_samp = [(0, np.array([float(0), float(1)]))]
def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    cases = []
    if q_cur == 0 and q_nex == 0:
        cases.append([])
    elif q_cur == 0 and q_nex == 1:
        cases.append([nuR.BUnSet(non_state, 'sig', 1, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])
    elif q_cur == 1 and q_nex == 1:
        cases.append([nuR.BUnSet(non_state, 'sig', 1, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])
    return cases

run_exp.exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms)
