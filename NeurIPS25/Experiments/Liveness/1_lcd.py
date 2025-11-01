import sys
import os
import numpy as np
import itertools
sys.path.append('../')
import run_exp
sys.path.append('../../')
import Tools.neuralmc.cav_nuR as nuR

N_lims = [500   , 1000  , 1500  , 2500  , 5000, 7500 , 10000, 12500, 15000, 17500, 20000, 22500, 90000 , 180000]
CBITSs = [9, 10, 11, 12, 13, 13, 14, 14, 14, 15, 15, 15, 17, 18]
specTXT = "FG enable -> (GF state = 2)"
module_name = "LCD"
file_name = "lcd"
LTLSpec = "LTLSPEC F G (Verilog.LCD.lcd_enable = TRUE) -> G F (Verilog.LCD.state[1] = TRUE & Verilog.LCD.state[0] = FALSE)"
SVSpec = "(@(posedge clk) s_eventually lcd_enable -> state == 2)"
range_vals_list = ['N', 'N', 'N','N','N','N','N','N', 'N', 'N']


start_ex = 0
scale = 1
size = [2] # [2, 1]
gap = 1e-4 # this is the gap of the sign activation function 
F_prec = 5
bits = 50
Ps = [N_lim for N_lim in N_lims]
Ms = [N_lim*2 for N_lim in N_lims]

is_acc = [0, 1]
is_init = [1, 0]
init_samp = [(0, np.array([float(0), float(0)]))]
def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    cases = []
    if q_cur == 0 and q_nex == 0:
        cases.append([])
    elif q_cur == 0 and q_nex == 1:
        cases.append([nuR.BUnSet(curr_vars, 'state', 2, ctx), nuR.Bset(non_state, 'lcd_enable', 1, ctx), nuR.BUnSet(next_vars, 'state', 2, ctx)])
    elif q_cur == 1 and q_nex == 1:
        cases.append([nuR.BUnSet(curr_vars, 'state', 2, ctx), nuR.Bset(non_state, 'lcd_enable', 1, ctx), nuR.BUnSet(next_vars, 'state', 2, ctx)])
    return cases

run_exp.exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms)