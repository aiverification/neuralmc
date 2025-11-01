import math
import sys
import os
import numpy as np
import itertools
sys.path.append('../')
import run_exp
sys.path.append('../../')
import Tools.neuralmc.cav_nuR as nuR

N_lims = [12   , 1000  , 2000  , 4000 , 6000 , 8000 , 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 40000, 70000, 140000, 280000]
CBITSs = [math.ceil(math.log2(N_lim)) for N_lim in N_lims]
specTXT = "((G !rst -> G (!stretch -> !stretch U switch_range)))"
module_name = "i2cStrech"
file_name = "i2c"

LTLSpec = "LTLSPEC G (Verilog.i2cStrech.rst = FALSE) -> G  ((Verilog.i2cStrech.stretch = FALSE) -> (Verilog.i2cStrech.stretch = FALSE U Verilog.i2cStrech.switch_range = TRUE))"

SVSpec = "(@(posedge clk) (always !rst) implies always (!stretch implies (!stretch s_until switch_range)))"
range_vals_list = ['N', 'N', 'N', 'N', 'N', 'N', 'N']

start_ex = 0

scale = 1
size = [3] #[3, 1]
gap = 1e-3 # this is the gap of the sign activation function 
F_prec = 5
bits = 50


Ps = [N_lim+1 for N_lim in N_lims]
Ms = [N_lim*2 for N_lim in N_lims]

is_acc = [0, 1, 1]
is_init = [1, 0, 0]
init_samp = [(0, np.array([float(0), float(0), float(0)]))]
def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    cases = []
    if q_cur == 0 and q_nex == 0:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx)])
        
    elif q_cur == 0 and q_nex == 1:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.BUnSet(curr_vars, 'stretch', 1, ctx), nuR.BUnSet(non_state, 'switch_range', 1, ctx)])
    elif q_cur == 1 and q_nex == 1:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.BUnSet(curr_vars, 'stretch', 1, ctx), nuR.BUnSet(non_state, 'switch_range', 1, ctx)])
    elif q_cur == 1 and q_nex == 2:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.Bset(curr_vars, 'stretch', 1, ctx), nuR.BUnSet(non_state, 'switch_range', 1, ctx)])
        
        
    if q_cur == 2 and q_nex == 2:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx)])
        
    return cases
run_exp.exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms)
