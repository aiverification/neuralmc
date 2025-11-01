import sys
import os
import numpy as np
import itertools
sys.path.append('../')
import run_exp
sys.path.append('../../')
import Tools.neuralmc.cav_nuR as nuR

N_lims = [30    , 300   , 600   , 900   , 1200  , 1800  , 2400  , 3000  , 6000  , 9000  , 12000 , 15000 , 18000 , 36000 , 72000 , 144000, 288000]
CBITSs = [0]*len(N_lims)
specTXT = "X G ((spi_not_busy & state1 & !rst) -> X state1)"
module_name = "Thermocouple"
file_name = "thermocouple"
LTLSpec = "LTLSPEC X G ((Verilog.Thermocouple.spi_not_busy = TRUE & Verilog.Thermocouple.state[1] = FALSE & Verilog.Thermocouple.state[0] = TRUE & Verilog.Thermocouple.rst = FALSE) -> X (Verilog.Thermocouple.state[1] = FALSE & Verilog.Thermocouple.state[0] = TRUE))"
SVSpec = "(@(posedge clk) spi_not_busy && (state == 1) && !rst |=> (state == 1))"
range_vals_list = ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']

start_ex = 0
scale = 1
size = [3] #[3, 1]
gap = 1e-3 # this is the gap of the sign activation function 
F_prec = 5
bits = 50

Ps = [5 for N_lim in N_lims]
Ms = [N_lim*2 for N_lim in N_lims]

is_acc  = [0, 0, 0, 1]
is_init = [1, 0, 0, 0]
init_samp = [(0, np.array([float(0), float(0), float(0)]))]
def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    _, _, bw_obj, _ = ctx
    cases = []
    
    if q_cur == 0 and q_nex == 1:
        cases.append([])
    
    elif q_cur == 1 and q_nex == 1:
        cases.append([nuR.bOr([nuR.BUnSet(non_state, 'rst', 1, ctx), nuR.BUnSet(non_state, 'spi_not_busy', 1, ctx), nuR.BUnSet(curr_vars, 'state', 1, ctx)], bw_obj)])
    elif q_cur == 1 and q_nex == 2:
        cases.append([nuR.BUnSet(non_state, 'rst', 1, ctx),  nuR.Bset(non_state, 'spi_not_busy', 1, ctx), nuR.Bset(curr_vars, 'state', 1, ctx)])
    
    elif q_cur == 2 and q_nex == 1:
        cases.append([
        	nuR.bOr([nuR.bAnd([nuR.Bset(non_state, 'rst', 1, ctx), nuR.Bset(curr_vars, 'state', 1, ctx)], bw_obj),
			nuR.bAnd([nuR.BUnSet(non_state, 'spi_not_busy', 1, ctx), nuR.Bset(curr_vars, 'state', 1, ctx)], bw_obj)
			], bw_obj)])
    elif q_cur == 2 and q_nex == 2:
        cases.append([nuR.BUnSet(non_state, 'rst', 1, ctx),  nuR.Bset(non_state, 'spi_not_busy', 1, ctx), nuR.Bset(curr_vars, 'state', 1, ctx)])
    elif q_cur == 2 and q_nex == 3:
        cases.append([nuR.BUnSet(curr_vars, 'state', 1, ctx)])
    
    elif q_cur == 3 and q_nex == 3:
        cases.append([])

    return cases

run_exp.exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms)
