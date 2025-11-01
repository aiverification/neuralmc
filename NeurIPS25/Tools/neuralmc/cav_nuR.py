"""
CAV-NUR -> Computer Aided Verification using Neural Reasoning.
===============================

This module provides ulilities for neural model checking of Verilog designs.

Global Constants:
  - colors: ANSI palettes for console output (Fore.*).

Primary Functions:
 1. readForVars(name, module_name, shell_inputs)
    • Extract state/input/output metadata by invoking EBMC’s symbol table.
    • Returns OrderedDicts: state_var, inp_out_vars with keys {lb, ub, size, dist, type}.

 2. set_lhs_state(var_list, value_list, bw_obj)
    • Build a balanced conjunction of BV equalities to assign values to state bits [for random sample ablation study].

 3. rangeBitwuzla(var, bw_obj, lb, ub)
    • Constrain a BitVec term within [lb, ub] using BV_UGE and BV_ULE.

 4. bAnd(term_list, bw_obj) / bOr(term_list, bw_obj)
    • Recursively assemble balanced AND/OR trees over lists of terms.

 5. bOrOfAnd(list_of_term_lists, bw_obj)
    • Create a disjunction of conjunctions for 2D arrays of terms.

 6. bv2int(bitvec_list, bw_obj, bits) / todecimal(x, bw_obj, bits) / todecimal2(...)
    • Decode Bitwuzla BitVec results into Python integers (two’s complement).

 7. bPrint(term_list, bw_obj) / bPrintFormula(term)
    • Print term symbols and values or formula strings for debugging.

 8. Bset(arr, var_name, val, context) / BUnSet(...)
    • Generate (or negate) equality constraints for named state/input variables.

 9. verilogSMT(name, module_name, state_vars, bits, inp_out_vars)
    • Invoke EBMC to create an SMT2 model, clean and extend with NuR-prefixed
      declarations, parse into Bitwuzla, assert range constraints.
    • Returns (bw_obj, curr_vars, next_vars, non_state_vars, state_names).

10. sanityCheckRange(bw_obj, rangesC, rangesN, curr_vars, next_vars)
    • Verify that current-state ranges imply next-state ranges (range invariant).

11. random_lhs_set(state_vars, state_names, curr_vars, bw_obj)
    • Sample random assignments for state variables and return assignment + term.

12. get_first_samples(...)
    • Collect initial SAT transitions without random inputs as sample tuples.

13. get_random_samples(...)
    • Collect random SAT/UNSAT transitions under random LHS sampling.

14. fake_cex_check(samples, cex_list)
    • Ensure no duplicate counterexamples between existing samples and new cex.

15. train_an_nrf(bw_obj, hyperparams, samples, init_samp, ..., engine, isAuto)
    • Train neural ranking/invariant functions with Gurobi or SMT-based learner.
    • Alternate training (nnparam) and verification (bwEBV.check), collect timings.
    • Return (training_time, checking_time, iterations) or (None,None,iter) on failure.

16. runExperiment(name, hyperparams, bw_obj, curr_vars, next_vars, non_state_vars,
    spec_automata, ctx, F_prec, bits, is_acc, init_samp, is_init,
    state_vars, state_names, rnd_smpC, engine, isAuto)
    • Orchestrate the full workflow: initial+random sampling, training loop,
      optional auto-architecture growth, return final metrics and model size.

Logging:
  • All timing data, counterexamples, and status messages are appended to `data.txt`.
  • Progress and debug prints appear on stdout with colorized output.

"""

import os
import time
import math
import re
import random
import numpy as np
import multiprocessing
from collections import OrderedDict
from itertools import chain, product
from colorama import init, Fore, Back, Style

import bitwuzla as bw
import Tools.neuralmc.check_for_gurobi as bwEBV
import Tools.neuralmc.train_by_gurobi as nnGurobi
import Tools.neuralmc.train_by_SMT as nnSMT

colours = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTCYAN_EX]
delta = 1e-2
col_num = len(colours)
norm_range = 100

"""
# =====================================================================
# 					Variable Name and Type using EBMC
# =====================================================================
"""


def readForVars(name, module_name, shell_inputs):
	os.system(f"ebmc ../../../Benchmarks/{name}.sv --show-symbol-table --bound 0 --top {module_name} > ../../Tools/neuralmc/Model-SMT/{name}_smb_tab.txt")
	data = ""
	print(os.getcwd())
	with open(f"../../Tools/neuralmc/Model-SMT/{name}_smb_tab.txt", 'r') as file:
	    data = file.read()
	data = data[data.find('Symbols:') + 9:]
	blocks = [block.strip() for block in data.strip().split('\n\n')]
	result = {}
	for block in blocks:
	    lines = block.split('\n')
	    var_name = lines[0].strip()
	    var_info = {}
	    for line in lines[1:]:
	        if len(line.split(':', 1)) == 1:
	          continue
	        key, value = map(str.strip, line.split(':', 1))
	        var_info[key] = value
	    result[var_name] = var_info
	
	state_var, inp_out_vars = OrderedDict(), OrderedDict()
	for var in result.keys():
	  if len(var.split(".", 1)) != 2:
	    continue
	  if var.split(".", 1)[1] == 'clk':
	  	continue
	  if(result[var]['flags'].split()[0] in ['state_var', 'input', 'output']):
	    
	    lb, ub, size, dist = 0, 1, 1, None
	    if(result[var]['type'] == 'bool'):
	      ap = input(f"Special Distribution for {var}? [Y/N]: ") if shell_inputs is None else next(shell_inputs)
	      if(ap in ['y','Y']):
	        p = int(input(f"Enter p value for {var}? [Int or 1/Int]: ")) if shell_inputs is None else int(next(shell_inputs))
	        if(int(p) < 1):
	          print("NOT SUPPORTED YET")
	        else:
	          dist = int(p)
	      elif(ap in ['n','N']):
	          dist = None
	      else:
	        print("INVALID INPUT")
	        breakpoint()
	        exit()
	    if(result[var]['type'] == 'unsignedbv'):
	      size = int(result[var]['* width'])
	      ap = input(f"Specify Lower and Upper bound for {var}? [Y/N]: ") if shell_inputs is None else next(shell_inputs)
	      if(ap in ['y','Y']):
	        lb = int(input(f"Lower Bound for {var}? [Unsigned Int]: ")) if shell_inputs is None else int(next(shell_inputs))
	        ub = int(input(f"Upper Bound for {var}? [Unsigned Int]: ")) if shell_inputs is None else int(next(shell_inputs))
	      elif(ap in ['n','N']):
	        lb = 0
	        ub = 2**size-1
	      else:
	        print("INVALID INPUT")
	        breakpoint()
	        exit()
	    if(result[var]['flags'].split()[0] in ['input', 'output']):
	      inp_out_vars[var.split(".", 1)[1]] = {'lb' : lb, 'ub' : ub, 'size' : size, 'dist' : dist, 'type' : result[var]['flags'].split()[0]}
	    else:
	      state_var[var.split(".", 1)[1]] = {'lb' : lb, 'ub' : ub, 'size' : size, 'dist' : dist, 'type' : 'state'}
	return state_var, inp_out_vars
	

"""
# =====================================================================
# 					    Bitwuzla Utility Functions
# =====================================================================

"""
def set_lhs_state(var, val, bw_obj):
    tm, opt, parser, bvsizeB = bw_obj
    eq_this = tm.mk_term(bw.Kind.EQUAL, [var[0], tm.mk_bv_value(bvsizeB, val[0])])
    if len(var) == 1:
        return eq_this
    return tm.mk_term(bw.Kind.AND, [eq_this, set_lhs_state(var[1:], val[1:], bw_obj)])

def rangeBitwuzla(var, bw_obj, lb, ub):
	tm, opt, parser, bvsizeB = bw_obj
	l1 = tm.mk_term(bw.Kind.BV_UGE, [var, tm.mk_bv_value(bvsizeB, lb)])
	u1 = tm.mk_term(bw.Kind.BV_ULE, [var, tm.mk_bv_value(bvsizeB, ub)])
	return tm.mk_term(bw.Kind.AND, [l1, u1])

def bAnd(arr, bw_obj):
    tm, opt, parser, bvsizeB = bw_obj
    if(len(arr) == 1):
        return arr[0]
    if(len(arr) == 2):								# REMOVE THIS CASE ITS REDUNDANT
        return tm.mk_term(bw.Kind.AND, [arr[0], arr[1]])
    part = len(arr) // 2
    return tm.mk_term(bw.Kind.AND, [bAnd(arr[:part], bw_obj), bAnd(arr[part:], bw_obj)])

def bOr(arr, bw_obj):
    tm, opt, parser, bvsizeB = bw_obj
    if(len(arr) == 1):
        return arr[0]
    if(len(arr) == 2):								# REMOVE THIS CASE ITS REDUNDANT
        return tm.mk_term(bw.Kind.OR, [arr[0], arr[1]])
    part = len(arr) // 2
    return tm.mk_term(bw.Kind.OR, [bOr(arr[:part], bw_obj), bOr(arr[part:], bw_obj)])

def bOrOfAnd(arr2D, bw_obj):
	arr1D = []
	for arr in arr2D:
		arr1D.append(bAnd(arr, bw_obj))
	return bOr(arr1D, bw_obj)

def bv2int(arr, bw_obj, bits):
	tm, opt, parser, bvsizeB = bw_obj
	arr2 = []
	for i in range(len(arr)):
		arr2.append(todecimal2(arr[i], bw_obj, bits))
	return arr2

def todecimal(x, bw_obj, bits):
    tm, opt, parser, bvsizeB = bw_obj
    val = int(parser.bitwuzla().get_value(x).value(10))
    s = 1 << (bits - 1)
    return (val & s - 1) - (val & s)

def todecimal2(x, bw_obj, bits):
    tm, opt, parser, bvsizeB = bw_obj
    val = int(parser.bitwuzla().get_value(x).value(10))
    s = 1 << (bits - 1)
    return (val & s - 1) - (val & s)


def bPrint(arr, bw_obj):
	tm, opt, parser, bvsizeB = bw_obj
	for ar in arr:
		print(f" {ar.symbol()} --> {parser.bitwuzla().get_value(ar).value(10)} ")

def bPrintFormula(trm):
	print(trm.str())

def Bset(arr, var, val, context):
	state_vars, inp_out_vars, bw_obj, bits = context
	tm, opt, parser, bvsizeB = bw_obj
	assert (var in state_vars.keys()) or (var in inp_out_vars.keys())
	var_keys = state_vars.keys() if (var in state_vars.keys()) else inp_out_vars.keys()
	return tm.mk_term(bw.Kind.EQUAL, [arr[[svk for svk in var_keys].index(var)], tm.mk_bv_value(bvsizeB, val)])

def BUnSet(arr, var, val, context):
	state_vars, inp_out_vars, bw_obj, bits = context
	tm, opt, parser, bvsizeB = bw_obj
	return tm.mk_term(bw.Kind.NOT, [Bset(arr, var, val, context)])

"""
# =====================================================================
# 					    Verilog to SMT using EBMC
# =====================================================================
"""


def verilogSMT(name, module_name, state_vars, bits, inp_out_vars):
	os.system(f"ebmc ../../../Benchmarks/{name}.sv --smt2 --bound 1 --top {module_name} --outfile ../../Tools/neuralmc/Model-SMT/{name}.smt2")
	smt2_model = ""
	with open(f"../../Tools/neuralmc/Model-SMT/{name}.smt2", 'r') as file:
	    smt2_model = file.read()
	# Remove the exit command
	smt2_model = '\n'.join([line for line in smt2_model.split("\n") if not any(rem_str in line for rem_str in ['(exit)', '; end of SMT2 file'])])
	
	state_changed = []
	for key, value in chain(state_vars.items(), inp_out_vars.items()):
		pre_    = f"|Verilog::{module_name}.{key}@0|"
		curr_   = f"|Verilog::{module_name}.{key}@1|" 
		next_   = f"|Verilog::{module_name}.{key}@2|"
		pre_bv  = pre_  if value['size'] > 1 else f"(ite {pre_} (_ bv1 1) (_ bv0 1))"
		curr_bv = curr_ if value['size'] > 1 else f"(ite {curr_} (_ bv1 1) (_ bv0 1))"
		next_bv = next_ if value['size'] > 1 else f"(ite {next_} (_ bv1 1) (_ bv0 1))"
		pre_B   = f"|NuR::{module_name}.{key}0|"
		curr_B  = f"|NuR::{module_name}.{key}1|"
		next_B  = f"|NuR::{module_name}.{key}2|"

		if((value['type'] == 'state')):
			assert (pre_ in smt2_model) and (curr_ in smt2_model) and (next_ in smt2_model)
			smt2_0 = f"(declare-const {pre_B } (_ BitVec {bits}))\n(assert (= ((_ zero_extend {bits - value['size']}) {pre_bv }) {pre_B }))"
			smt2_1 = f"(declare-const {curr_B} (_ BitVec {bits}))\n(assert (= ((_ zero_extend {bits - value['size']}) {curr_bv}) {curr_B}))"
			smt2_2 = f"(declare-const {next_B} (_ BitVec {bits}))\n(assert (= ((_ zero_extend {bits - value['size']}) {next_bv}) {next_B}))"
			smt2_model += "\n" + smt2_0 + "\n" + smt2_1 + "\n" + smt2_2

		elif((value['type'] == 'input')):
			assert (pre_ in smt2_model) and (curr_ in smt2_model)
			smt2_0 = f"(declare-const {pre_B } (_ BitVec {bits}))\n(assert (= ((_ zero_extend {bits - value['size']}) {pre_bv }) {pre_B }))"
			smt2_1 = f"(declare-const {curr_B} (_ BitVec {bits}))\n(assert (= ((_ zero_extend {bits - value['size']}) {curr_bv}) {curr_B}))"
			smt2_model += "\n" + smt2_0 + "\n" + smt2_1 

		elif((value['type'] == 'output')):
			assert (curr_ in smt2_model)
			if (next_ in smt2_model):
				smt2_1 = f"(declare-const {curr_B} (_ BitVec {bits}))\n(assert (= ((_ zero_extend {bits - value['size']}) {curr_bv}) {curr_B}))"
				smt2_2 = f"(declare-const {next_B} (_ BitVec {bits}))\n(assert (= ((_ zero_extend {bits - value['size']}) {next_bv}) {next_B}))"
				smt2_model += "\n" + smt2_1 + "\n" + smt2_2
			else:
				smt2_1 = f"(declare-const {curr_B} (_ BitVec {bits}))\n(assert (= ((_ zero_extend {bits - value['size']}) {curr_bv}) {curr_B}))"
				smt2_model += "\n" + smt2_1
		else:
			print("[INVALID Value Type]")
			breakpoint()
            
	smt2_model = smt2_model.replace("(assert false)", "").replace("(check-sat)", "")
	with open(f"../../Tools/neuralmc/Model-SMT/{name}.smt2", "w") as file:
	    file.write(smt2_model)
	#print(smt2_model)
	curr_vars, next_vars, non_state_vars, state_names = [] , [] , [], []
	tm = bw.TermManager()
	opt = bw.Options()
	parser = bw.Parser(tm, opt)
	res = parser.parse(f"../../Tools/neuralmc/Model-SMT/{name}.smt2")
	bvsizeB = tm.mk_bv_sort(bits)
	rangesC = []
	rangesN = []
	bw_obj = (tm, opt, parser, bvsizeB)
	for key, value in chain(state_vars.items(), inp_out_vars.items()):
		if((value['type'] == 'state')):
			state_names.append(key)
			cv = parser.parse_term(f"|NuR::{module_name}.{key}1|")
			nv = parser.parse_term(f"|NuR::{module_name}.{key}2|")
			curr_vars.append(cv)
			next_vars.append(nv)
			rangesN.append(rangeBitwuzla(next_vars[-1], bw_obj, value["lb"], value["ub"]))
			rangesC.append(rangeBitwuzla(curr_vars[-1], bw_obj, value["lb"], value["ub"]))
		elif((value['type'] == 'input')):
			cv = parser.parse_term(f"|NuR::{module_name}.{key}1|")
			non_state_vars.append(cv)
			rangesC.append(rangeBitwuzla(non_state_vars[-1], bw_obj, value["lb"], value["ub"]))
		elif((value['type'] == 'output')):
			cv = parser.parse_term(f"|NuR::{module_name}.{key}1|")
			non_state_vars.append(cv)
			rangesC.append(rangeBitwuzla(non_state_vars[-1], bw_obj, value["lb"], value["ub"]))
	# 0101, 1212 -> 1_delay; 1211 -> s2_lcd
	sanityCheckRange(bw_obj, rangesC, rangesN, curr_vars, next_vars)
	parser.bitwuzla().assert_formula(bAnd(rangesC, bw_obj))
	parser.bitwuzla().assert_formula(bAnd(rangesN, bw_obj))
	bw_obj = (tm, opt, parser, bvsizeB)
	os.system(f"rm ../../Tools/neuralmc/Model-SMT/{name}.smt2")
	return bw_obj, curr_vars, next_vars, non_state_vars, state_names

def sanityCheckRange(bw_obj, rangesC, rangesN, curr_vars, next_vars):
    # The variable range needs to be an invariant, which means if current state satisfies range so should next state.
    tm, opt, parser, bvsizeB = bw_obj
    beginV = time.time()
    parser.bitwuzla().push()
    parser.bitwuzla().assert_formula(bAnd(rangesC, bw_obj))
    parser.bitwuzla().assert_formula(tm.mk_term(bw.Kind.NOT, [bAnd(rangesN, bw_obj)]))
    res = parser.bitwuzla().check_sat()
    endV1 = time.time()
    print(f"Time Range Invar: {endV1 - beginV}")
    if(res == bw.Result.UNSAT):
        print(f"Range is an Invar [PASS]")
    elif(res == bw.Result.SAT):
        print(f"Range is an Invar [FAIL]")
        bPrint(curr_vars, bw_obj)
        bPrint(next_vars, bw_obj)
        breakpoint()
    parser.bitwuzla().pop()


"""
# =====================================================================
# 			 Run Neural Model Checking (main-functions)
# =====================================================================
"""
def random_lhs_set(state_vars, state_names, curr_vars, bw_obj):
    val = []
    for state in state_names:
        lb, ub = state_vars[state]['lb'], state_vars[state]['ub']
        val.append(random.randint(lb, ub))
    return val, set_lhs_state(curr_vars, val, bw_obj)

def get_random_samples(samples, bw_obj, curr_vars, next_vars, non_state_vars, spec_automata, ctx, q_set, bits, state_vars, state_names, rnd_smpC): 
    tm, opt, parser, bvsizeB = bw_obj
    print(20*'=')
    print('Random Samples')
    print(20*'=')
    for q_cur, q_nex in product(q_set, repeat=2):
        print(f"{Fore.CYAN}q = {q_cur} to q = {q_nex}{Style.RESET_ALL}")
        for smp_cnt in range(0, rnd_smpC):
            for trans_ in spec_automata(ctx, q_cur, curr_vars, None, q_nex, next_vars, None, non_state_vars, 1):
                parser.bitwuzla().push()
                val_lhs, set_lhs = random_lhs_set(state_vars, state_names, curr_vars, bw_obj)
                parser.bitwuzla().assert_formula(set_lhs)
                if len(trans_) > 0:
                    parser.bitwuzla().assert_formula(bAnd(trans_, bw_obj))
                res = parser.bitwuzla().check_sat()
                if (res == bw.Result.SAT):
                    #bPrint(curr_vars, bw_obj)
                    c_cur = np.array(bv2int(curr_vars, bw_obj, bits)) 
                    c_nex = np.array(bv2int(next_vars, bw_obj, bits))
                    samples.append((q_cur, c_cur, q_nex, c_nex))
                    print(f"{Fore.LIGHTGREEN_EX}Valid sample: {(q_cur, c_cur, q_nex, c_nex)}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Invalid LHS Sample: ({q_cur}, {val_lhs}, {q_nex},  X) {Style.RESET_ALL}")
                parser.bitwuzla().pop()    

def get_first_samples(samples, bw_obj, curr_vars, next_vars, non_state_vars, spec_automata, ctx, q_set, bits, state_vars, state_names): 
    tm, opt, parser, bvsizeB = bw_obj
    print(20*'=')
    print('Initial Samples')
    print(20*'=')
    for q_cur, q_nex in product(q_set, repeat=2):
        #print(f"q = {q_cur} to q = {q_nex}")
        for trans_ in spec_automata(ctx, q_cur, curr_vars, None, q_nex, next_vars, None, non_state_vars, 1):
            parser.bitwuzla().push()
            if len(trans_) > 0:
            	parser.bitwuzla().assert_formula(bAnd(trans_, bw_obj))
            res = parser.bitwuzla().check_sat()
            if (res == bw.Result.SAT):
                #bPrint(curr_vars, bw_obj)
                c_cur = np.array(bv2int(curr_vars, bw_obj, bits)) 
                c_nex = np.array(bv2int(next_vars, bw_obj, bits))
                samples.append((q_cur, c_cur, q_nex, c_nex))
                print(f"{Fore.CYAN}q = {q_cur} to q = {q_nex} is SAT {(q_cur, c_cur, q_nex, c_nex)}{Style.RESET_ALL}")
            else:
                print(f"{Fore.CYAN}q = {q_cur} to q = {q_nex} is UNSAT{Style.RESET_ALL}")
            parser.bitwuzla().pop()    

def fake_cex_check(samples, cex):
    for ce in cex:
        for sample in samples:
            cnd = (
                ce[0] == sample[0] and
                np.array_equal(ce[1], sample[1]) and
                ce[2] == sample[2] and
                np.array_equal(ce[3], sample[3])
            )
            if cnd:
                breakpoint()
                assert False

def train_an_nrf(bw_obj, hyperparameters, samples, init_samp, curr_vars, next_vars, non_state_vars, spec_automata, ctx, F_prec, bits, is_acc, q_set, is_init, engine="gurobi", isAuto=True):
    scale, P, size, gap, M, kappa = hyperparameters
    success = False
    bw_time = 0
    gu_time = 0
    cex = samples
    cex_init = init_samp
    if engine == "gurobi":
    	mipL = nnGurobi.MIPLearn(is_acc, size = size, P = P, M = M)
    else:
    	smtL = nnSMT.SMTLearn(is_acc, size = size, P = P, M = M)
    for try_i in range(5000):
    	begin = time.time()      
    	if engine == "gurobi":
    		nnparam, linparam, kappa, best_F = nnGurobi.gurobiNNtrain(try_i, mipL, cex, cex_init, samples, init_samp, scale, P, is_acc, size, gap, kappa)
    	else:
    		nnparam, linparam, kappa, best_F = nnSMT.smtNNtrain(try_i, smtL, cex, cex_init, samples, init_samp, scale, P, is_acc, size, gap, kappa, engine)
    	if nnparam == None:
    		return None, None, None
    	F_prec = best_F#max(best_F, F_prec)
    	print(f"PRECISION: {F_prec}")
    	end = time.time()
    	gu_time += (end - begin)

    	begin = time.time()
    	cex = bwEBV.check(nnparam, linparam, curr_vars, next_vars, non_state_vars, scale, spec_automata, ctx, q_set, is_acc, F_prec, bw_obj, bits, gap, kappa)
    	cex_init = bwEBV.check_init(nnparam, linparam, curr_vars, is_init,  F_prec, bw_obj, scale, bits, gap, q_set, ctx, kappa)
    	end = time.time()
    	bw_time += (end - begin)
    	if (len(cex) + len(cex_init)) == 0:
    		success = True
    		print(f'{Fore.GREEN}\t*********  Yay! We\'ve got a ranking function.  ************{Style.RESET_ALL}')
    		break
    	fake_cex_check(samples, cex)
    	#new_samples = cex
    	samples += cex
    	init_samp += cex_init
    if not success:
    	print(f'{Fore.RED}\t*********  Ranking Function training Failed!  ************{Style.RESET_ALL}')
    return gu_time, bw_time, try_i



def runExperiment(name, hyperparameters, bw_obj, curr_vars, next_vars, non_state_vars, spec_automata, ctx, F_prec, bits, is_acc, init_samp, is_init, state_vars, state_names, rnd_smpC, engine, isAuto):
    scale, P, size, gap, M, kappa = hyperparameters
    seed = 2
    random.seed(seed)
    q_set = list(range(len(is_acc)))
    samples = []
    get_first_samples(samples, bw_obj, curr_vars, next_vars, non_state_vars, spec_automata, ctx, q_set, bits, state_vars, state_names)
    get_random_samples(samples, bw_obj, curr_vars, next_vars, non_state_vars, spec_automata, ctx, q_set, bits, state_vars, state_names, rnd_smpC)
    while(True):
    	gu_time, bw_time, guess_cnt = train_an_nrf(bw_obj, hyperparameters, samples, init_samp, curr_vars, next_vars, non_state_vars, spec_automata, ctx, F_prec, bits, is_acc, q_set, is_init, engine)
    	if gu_time == None and isAuto == True:
    		size = [size[0], 1 if len(size) == 1 else (size[1] + 1)] 
    		hyperparameters = scale, P, size, gap, M, kappa
    		if size[1] == 5:
    			exit()
    	else:
    		break
    return gu_time, bw_time, guess_cnt, size
