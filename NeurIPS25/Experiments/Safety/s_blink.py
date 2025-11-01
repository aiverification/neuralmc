import sys
import os
import numpy as np
import itertools
sys.path.append('../')
import run_exp
sys.path.append('../../')
import Tools.neuralmc.cav_nuR as nuR

N_lims = [2**pr for pr in range(8, 33)]
CBITSs = [pr for pr in range(8, 33)]
specTXT = "G ((mode1 & X mode1) -> X !flg)"
module_name = "BLINK"
file_name = "blink"
LTLSpec = "LTLSPEC G ((Verilog.BLINK.mode = TRUE & X Verilog.BLINK.mode = TRUE ) -> X Verilog.BLINK.flg = FALSE)"
#LTLSpec = "LTLSPEC X G ((Verilog.BLINK.flg = FALSE & X Verilog.BLINK.flg = FALSE & Verilog.BLINK.rst = FALSE) -> ( (Verilog.BLINK.led = TRUE & X Verilog.BLINK.led= TRUE) | (Verilog.BLINK.led = FALSE & X Verilog.BLINK.led = FALSE) ) )"
SVSpec = "(@(posedge clk) (mode ##1 mode |-> !flg))"
range_vals_list = ['N',  'N', 'N', 'N', 'N', 'N']

start_ex = 0

scale = 1
size = [2] #[2, 1]
gap = 1e-3 # this is the gap of the sign activation function 
F_prec = 5
bits = 100

Ps = [5 for N_lim in N_lims]
Ms = [N_lim*2 for N_lim in N_lims]
is_acc =  [0, 0, 1]
is_init = [1, 0, 0]
init_samp = [(0, np.array([float(0), float(0)]))]

def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    _, _, bw_obj, _ = ctx
    cases = []

    if q_cur == 0 and q_nex == 0:
        cases.append([nuR.BUnSet(curr_vars, 'mode', 1, ctx)]) 
        
    elif q_cur == 0 and q_nex == 1:
        cases.append([nuR.Bset(curr_vars, 'mode', 1, ctx)]) 
        
    elif q_cur == 1 and q_nex == 0:
        cases.append([nuR.BUnSet(curr_vars, 'mode', 1, ctx)]) 

    elif q_cur == 1 and q_nex == 1:
        #cases.append([nuR.Bset(non_state, 'rst', 1, ctx),
        #					   nuR.Bset(curr_vars, 'mode', 1, ctx)])
        cases.append([nuR.BUnSet(non_state, 'flg', 1, ctx),
        					   nuR.Bset(curr_vars, 'mode', 1, ctx)])
        
    elif q_cur == 1 and q_nex == 2:
        cases.append([nuR.Bset(non_state, 'flg', 1, ctx), 
                      nuR.Bset(curr_vars, 'mode', 1, ctx)])

    elif q_cur == 2 and q_nex == 2:
        cases.append([])  
        
    return cases


run_exp.exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms)
