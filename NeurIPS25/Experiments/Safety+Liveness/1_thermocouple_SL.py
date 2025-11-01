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
specTXT = "((G !rst -> G (s2 -> s2 U s1)))"
#Note this is different from "G (!rst -> (s2 -> s2 U s1) ))"
module_name = "Thermocouple"
file_name = "thermocouple"


LTLSpec = "LTLSPEC  (G (Verilog.Thermocouple.rst = FALSE) -> G ( ((Verilog.Thermocouple.state[1] = TRUE & Verilog.Thermocouple.state[0] = TRUE) -> ((Verilog.Thermocouple.state[1] = TRUE & Verilog.Thermocouple.state[0] = TRUE) U (Verilog.Thermocouple.state[1] = FALSE & Verilog.Thermocouple.state[0] = TRUE))) ))"
SVSpec = "(@(posedge clk) (always !rst) implies always (state==2 implies (state==2 s_until state==1)))"
range_vals_list = ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']

start_ex = 0
scale = 1
size = [3] #[3, 1]
gap = 1e-3 # this is the gap of the sign activation function 
F_prec = 5
bits = 50

Ps = [1 for N_lim in N_lims]
Ms = [N_lim*2 for N_lim in N_lims]

is_acc = [0, 1, 1]
is_init = [1, 0, 0]
init_samp = [(0, np.array([float(0), float(0), float(0)]))]
def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    cases = []
    if q_cur == 0 and q_nex == 0:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx)])
        
    elif q_cur == 0 and q_nex == 1:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.Bset(curr_vars, 'state', 2, ctx), nuR.BUnSet(next_vars, 'state', 1, ctx)])
    elif q_cur == 1 and q_nex == 1:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx), nuR.Bset(curr_vars, 'state', 2, ctx), nuR.BUnSet(next_vars, 'state', 1, ctx)])
    elif q_cur == 1 and q_nex == 2:
        cases.append([nuR.BUnSet(curr_vars, 'state', 1, ctx), nuR.Bset(non_state, 'rst', 0, ctx), nuR.BUnSet(curr_vars, 'state', 2, ctx)])
        
        
    if q_cur == 2 and q_nex == 2:
        cases.append([nuR.Bset(non_state, 'rst', 0, ctx)])
        
    return cases



run_exp.exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms)
