import sys
import os
import numpy as np
import itertools
sys.path.append('../')
import run_exp
sys.path.append('../../')
import Tools.neuralmc.cav_nuR as nuR


N_lims = [2**4, 2**6, 2**8, 2**9, 2**10, 2**11, 2**12, 2**14, 2**15, 2**16]
CBITSs = [4, 6, 8, 9, 10, 11, 12, 14, 15, 16]
specTXT = "X G ((tx_state -> tx_busy) & (!tx_state -> !tx_busy))"
module_name = "UART_T"
file_name = "uart_transmit"
LTLSpec = "LTLSPEC X G ((Verilog.UART_T.tx_state = TRUE -> Verilog.UART_T.tx_busy = TRUE) & (Verilog.UART_T.tx_state = FALSE -> Verilog.UART_T.tx_busy = FALSE))"
SVSpec = "(@(posedge clk) ##1 (tx_state <-> tx_busy))"
range_vals_list = ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']

start_ex = 0
scale = 1
size = [3] #[3, 1]
gap = 1e-4 # this is the gap of the sign activation function 
F_prec = 5
bits = 50

Ps = [10 for N_lim in N_lims[:8]] + [N_lim/5 for N_lim in N_lims[8:]]
Ms = [N_lim*2 for N_lim in N_lims]

is_acc  = [0, 0, 1]
is_init = [1, 0, 0]
init_samp = [(0, np.array([float(0), float(0), float(0)]))]
def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    cases = []
    if q_cur == 0 and q_nex == 1:
        cases.append([])
    
    elif q_cur == 1 and q_nex == 1:
        cases.append([nuR.BUnSet(curr_vars, 'tx_state', 1, ctx), nuR.BUnSet(non_state, 'tx_busy', 1, ctx)])
        cases.append([nuR.Bset(curr_vars, 'tx_state', 1, ctx), nuR.Bset(  non_state, 'tx_busy', 1, ctx)])

    elif q_cur == 1 and q_nex == 2:
        cases.append([nuR.Bset(  curr_vars, 'tx_state', 1, ctx), nuR.BUnSet(non_state, 'tx_busy', 1, ctx)])
        cases.append([nuR.BUnSet(curr_vars, 'tx_state', 1, ctx), nuR.Bset(  non_state, 'tx_busy', 1, ctx)])
    
    if q_cur == 2 and q_nex == 2:
        cases.append([])

    return cases


run_exp.exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms)

