import sys
import os
import numpy as np
import itertools
sys.path.append('../')
import run_exp
sys.path.append('../../')
import Tools.neuralmc.cav_nuR as nuR

Mults = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
specTXT = "F G ! rst-> G F disp_ena"
module_name = "VGA"

N_lims = [0]*len(Mults) # Only size of array needed
CBITSs = [0]*len(Mults) # Only size of array needed

file_name = "vga"
LTLSpec = "LTLSPEC F G (Verilog.VGA.rst = FALSE) -> G F (Verilog.VGA.disp_ena = TRUE)"
SVSpec = "(@(posedge clk) s_eventually !rst -> disp_ena)"
range_vals_list = ['N']*100

start_ex = 0

scale = 1
size = [4] # [4, 1]
gap = 1e-4 # this is the gap of the sign activation function 
F_prec = 5
bits = 50

Ps = [1 for Mult in Mults]
Ms = [Mult*66*2 for Mult in Mults]
is_acc = [0, 1]
is_init = [1, 0]
init_samp = [(0, np.array([float(0), float(0), float(0), float(0)]))]

def spec_automata(ctx, q_cur, curr_vars, V0, q_nex, next_vars, V1, non_state, s):
    cases = []
    if q_cur == 0 and q_nex == 0:
        cases.append([])
    elif q_cur == 0 and q_nex == 1:
        cases.append([nuR.BUnSet(non_state, 'disp_ena', 0, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])
    elif q_cur == 1 and q_nex == 1:
        cases.append([nuR.BUnSet(non_state, 'disp_ena', 0, ctx), nuR.Bset(non_state, 'rst', 0, ctx)])
    return cases

run_exp.exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms)

