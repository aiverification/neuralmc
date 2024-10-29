"""
Author:
--------
Abhinandan Pal  
University of Birmingham

Copyright:
-----------
© 2024 University of Birmingham. All rights reserved.

For Theoratical Details refer to [1].

[1] Mirco Giacobbe, Daniel Kroening, Abhinandan Pal, and Michael Tautschnig (alphabetical). “Neural Model Checking”.
Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS’24), December 9-15, 2024, Vancouver, Canada.
"""

import os
import time
import math
import re
import random
import multiprocessing
from collections import OrderedDict
from itertools import chain
from colorama import init, Fore, Back, Style

import torch
import pandas as pd
import torch.optim as optim

import bitwuzla as bw
import Tools.neuralmc.bitwuzlaEncoderBV as bwEBV
import Tools.neuralmc.nrf as neuralrf

colours = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTCYAN_EX]
delta = 1e-2
col_num = len(colours)
norm_range = 100

"""
# =====================================================================
# 					Variable Name and Type using EBMC
# =====================================================================

Overview:
----------
The `readForVars` function extracts variable information from the symbol table of a SystemVerilog file obtaned using EBMC.
It processes the symbol table to identify state variables, inputs, and outputs, allowing the user to specify sampling distributions, bounds, and sizes for certain variables.

Parameters:
------------
- `name` (str):  
  The name of the hardware design (SystemVerilog file without the extension).

- `module_name` (str):  
  The top-level module name within the SystemVerilog design.

- `shell_inputs` (iterator or None):  
  An optional iterator for shell inputs. If provided, user inputs are read from this iterator, otherwise, they are prompted during execution.

Process:
---------
1. Runs EBMC on the SystemVerilog file to generate a symbol table and stores it in a text file.

2. Reads and parses the symbol table to extract variable information, filtering by flags (e.g., `state_var`, `input`, `output`).

3. Prompts the user to specify special sampling distributions, lower/upper bounds, and sizes for the variables, or reads from `shell_inputs`.

4. Classifies variables into:
   - `state_var`: State variables for the design.
   - `inp_out_vars`: Input and output variables, with metadata including bounds, size, and distribution.

Returns:
---------
- **state_var** (OrderedDict):  
  Contains state variables and their respective properties (`lb`, `ub`, `size`, `dist`, `type`).

- **inp_out_vars** (OrderedDict):  
  Contains input and output variables with their properties (`lb`, `ub`, `size`, `dist`, `type`).

Notes:
-------
- If `shell_inputs` is provided, user prompts are bypassed in favor of pre-defined inputs.
- Handles only certain variable types (`bool`, `unsignedbv`).
- Includes special handling for clock signals, which are skipped.

"""


def readForVars(name, module_name, shell_inputs):
	os.system(f"../../Tools/ebmc/ebmc_v5_x ../../Benchmarks/{name}.sv --show-symbol-table --bound 0 --top {module_name} > ../../Tools/neuralmc/Model-SMT/{name}_smb_tab.txt")
	data = ""
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
	        ub = 2**size
	      else:
	        print("INVALID INPUT")
	        breakpoint()
	        exit()
	    if(result[var]['flags'].split()[0] in ['input', 'output']):
	      inp_out_vars[var.split(".", 1)[1]] = {'lb' : lb, 'ub' : ub, 'size' : size, 'dist' : dist, 'type' : result[var]['flags'].split()[0]}
	    else:
	      state_var[var.split(".", 1)[1]] = {'lb' : lb, 'ub' : ub, 'size' : size, 'dist' : dist, 'type' : 'state'}
	state_var['q'] = {'lb' : 0, 'ub' : 1, 'size' : 1, 'dist' : None}

	return state_var, inp_out_vars
	

"""
# =====================================================================
# 					  Sampling Using Verilator
# =====================================================================

Overview:
----------
This module provides several utility functions for generating Verilog-based data samplers, running Verilator for training data generation, and processing data files.

Functions:
-----------

1. **order_strings_by_appearance(larger_string, string_list)**:
    Orders strings from `string_list` based on their position in `larger_string`. 
    This ensures that the input and output variables appear in the list in the same order as they are defined in the function signature of the top-level module being verified.

2. **makeVerilogSampler(name, module_name, smp, trace_len, spec_str, state_vars, inp_out_vars, stt_acc=[1], q_bits=1, q_max=1)**:
    Generates a Verilog sampler based on the provided parameters and writes it to a file. It defines state variables, input-output variables, and automates sampling for neural training data.

    - `name` (str): Name of the hardware design.
    - `module_name` (str): Top-level module name.
    - `smp` (int): Number of samples to generate.
    - `trace_len` (int): Length of the traces.
 	- `spec_str` (list of strings): The A¬Φ automaton, written in SystemVerilog. It uses the clock (`clk`) of the `top_module` to synchronize with the model under verification, forming the synchronous composition M ∥∥ A¬Φ.    - `state_vars` (dict): Dictionary of state variables with metadata (size, bounds, etc.).
    - `inp_out_vars` (dict): Dictionary of input-output variable data obtained from function `readForVars`.
    - `stt_acc` (list of int): List of accepted states (default is `[1]`).
    - `q_bits` (int): Number of bits in the automata state `q`.
    - `q_max` (int): Maximum value of `q`.

3. **trace_file_as_tensor(name, stt_acc, q_max)**:
    Loads the generated trace file, converts it into tensors, and categorizes transitions by automata state.
    - Returns: Dictionary of tensors representing transitions, and a tensor of all unique states.

4. **run_verilator(name)**:
    Compiles and runs the Verilog sampler using Verilator to generate training data.

Process:
---------
1. **Verilog Sampler Creation (makeVerilogSampler)**:  
   - Reads the SystemVerilog file and matches the top-level module to order input-output variables.
   - Generates the Verilog code for sampling based on state variables, inputs, and specifications.
   - Writes the sampler code to a file.

2. **Trace File Processing (trace_file_as_tensor)**:  
   - Loads the CSV trace file, removes duplicates, and converts it into tensors.
   - Groups transitions by automata state and returns tensors for training purposes.

3. **Training Data Generation (run_verilator)**:  
   - Compiles the generated Verilog sampler using Verilator.
   - Runs the compiled binary to produce training data for neural model checking.

"""


def order_strings_by_appearance(larger_string, string_list):
    string_positions = {string: larger_string.find(string) for string in string_list}
    sorted_strings = sorted(string_list, key=lambda x: string_positions[x])
    return sorted_strings

def makeVerilogSampler(name, module_name, smp, trace_len, spec_str, state_vars, inp_out_vars, stt_acc = [1], q_bits = 1, q_max = 1):
	state_vars['q']['size'] = q_bits
	state_vars['q']['ub'] = q_max
	verilog_code = ""
	with open(f"../../Benchmarks/{name}.sv", 'r') as file:
	    verilog_code = file.read()
	
	# Define the regex pattern to match the module declaration
	pattern = r"(module)\s+(\w+)\s*(#\((.*?)\))?\s*\((.*?)\);"
	match = re.search(pattern, verilog_code)
	inp_out_vars_ordered = order_strings_by_appearance(match.group(0), [v for v in inp_out_vars.keys()] + ["clk"])
	#assert match.group(2) == module_name
	#inp_out_vars = match.group(3) if match else None
	#inp_out_vars = [part.strip().split()[1] for part in inp_out_vars.split(',')]

	mstr = "\n//=====================================================\n"
	mstr += "\n//\t\tAppend begin\n"
	mstr += "\n//=====================================================\n\n"
	mstr += "module main();\n"
	mstr += '\n'.join(['reg ' + ('' if value['size'] == 1 else f" [{value['size'] - 1}:0] ") + key + ";" for key, value in inp_out_vars.items()]) + "\n"
	q_str = "reg q" if q_bits == 1 else f"reg [{q_bits-1}:0] q"
	mstr += f"reg clk;\ninteger f1;\ninteger f2;\nreg [31:0] rnd;\nreg isValid;\nstring res;\n{q_str};\nalways #(1) clk <= ~clk;\n"
	call = ', '.join([f".{v}({v})" for v in inp_out_vars_ordered])
	mstr += f"{match.group(2)} obj({call});\n"
	mstr += f"initial begin\n\tf1 = $fopen(\"../../Tools/neuralmc/Traces/{name}_all_trans.csv\", \"w\");\n"
	
	mstr += f"\tfor (int i = 0; i < {smp}; i = i + 1) begin\n"
	fmt1, fmt2 = "", "" 
	for key, value in state_vars.items():
		rng = "0" if(value['size'] == 1) else f"{str(value['size'] - 1)}:0"
		svc = f"{key}" if ( key == "q") else f"obj.{key}"
		assert value['dist'] is None
		mstr += f"\t\trnd = {str(value['lb'])} + $urandom % {str(value['ub']+1)};\n\t\t{svc} = rnd[{rng}];\n"
		fmt2 += f"{svc}, "
		fmt1 += "%d, "
	mstr += f"\t\tfor (int j = 0; j < {trace_len}; j = j + 1) begin\n"
	for key, value in inp_out_vars.items():
		if value['type'] == 'output':
			continue
		rng = "0" if(value['size'] == 1) else f"{str(value['size'] - 1)}:0"
		if value['dist'] is not None:
			mstr += f"\t\t\t{key} = ($urandom % {value['dist']} == 0);\n"
		else:
			mstr += f"\t\t\trnd = {str(value['lb'])} + $urandom % {str(value['ub']+1)};\n\t\t\t{key} = rnd[{rng}];\n"	
	mstr += f"\t\t\tres = $sformatf(\"{fmt1[:-2]}\", {fmt2[:-2]});\n"
	mstr += "\t\t\t#2;\n"
	for sp in spec_str:
		mstr += "\t\t\t"+ sp + "\n"
	pstr1 =  f"$fwrite(f1, \"%s, {fmt1[:-2]} \\n \", res, {fmt2[:-2]});"
	pstr2 =  f"$fwrite(f2, \"%s, {fmt1[:-2]} \\n \", res, {fmt2[:-2]});"
	acc_cnd = (" | ").join([f"q == {num}" for num in stt_acc])
	mstr += f"\t\t\tif (isValid == 0)\n\t\t\t\tbreak;\n"  #\n\t\t\telse\n\t\t\t\tbreak;

	mstr += f"\t\t\t$fwrite(f1, \"%s, {fmt1[:-2]} \\n \", res, {fmt2[:-2]});"
	mstr += "\n\t\tend\n\tend\n"

	mstr += "\t$finish;\nend\nendmodule"
	verilog_code += mstr 
	#print(verilog_code)

	with open(f"../../Tools/neuralmc/Verilog-Sampler/{name}_smp.sv", "w") as file:
	    file.write(verilog_code)
	

def trace_file_as_tensor(name, stt_acc, q_max):
    all_t = torch.tensor(pd.read_csv(f"../../Tools/neuralmc/Traces/{name}_all_trans.csv").drop_duplicates().values)
    spec_states = torch.unique(all_t[:, -1])
    all_states = torch.unique(torch.cat((all_t[:,:all_t.size(1)//2], all_t[:,all_t.size(1)//2:])), dim = 0)
    #tnsr.veiw splits the current and next state, rest of it splits on the automata state
    all_trans_tensors = {q.item(): all_t[all_t[:, -1] == q].view(-1, 2, all_t.size(1)//2) for q in spec_states}
    for q in range(q_max+1):
    	if q not in spec_states:
    		all_trans_tensors[q] = torch.Tensor([])
    all_trans_sizes = {value: all_trans_tensors[value].size(0) for value in all_trans_tensors}
    os.system(f"rm ../../Tools/neuralmc/Traces/{name}_all_trans.csv")
    print(f"\t------------DATASET SIZE = (ALL: {all_trans_sizes})------------")
    return all_trans_tensors, all_states

def run_verilator(name):
	os.system(f"verilator --binary -j 0 ../../Tools/neuralmc/Verilog-Sampler/{name}_smp.sv")
	os.system(f"obj_dir/V{name}_smp")
	print("\t\t----------Training Data Generation Completed------------")


"""
# =====================================================================
# 					    Bitwuzla Utility Functions
# =====================================================================

Overview:
----------
This module provides utility functions for performing bit-vector operations using the Bitwuzla SMT solver.
These functions cover logical operations (`AND`, `OR`), range checks, bit-vector to integer conversions, and printing.

Functions:
-----------

1. **rangeBitwuzla(var, bw_obj, lb, ub)**:
    Creates a term that checks if the bit-vector `var` is within the range `[lb, ub]`.
    - Returns: SMT term representing the range check.

2. **bAnd(arr, bw_obj)**:
    Recursively applies the `AND` operation to all elements in the array `arr`.

3. **bOr(arr, bw_obj)**:
    Recursively applies the `OR` operation to all elements in the array `arr`.

4. **bOrOfAnd(arr2D, bw_obj)**:
    Combines a 2D array of terms, applying `AND` on inner arrays and `OR` on the resulting terms.

5. **Bset(arr, var, val, context)**:
    Sets the value of a variable `var` in the state/input-output array `arr` to `val`.

6. **BUnSet(arr, var, val, context)**:
    Sets the value of a variable `var` in the state/input-output array `arr` to NOT `val`.

	[The above six return the SMT term representing the specified operation]

7. **bv2int(arr, bw_obj, bits)**:
    Converts an array of bit-vector terms into their integer equivalents.

8. **todecimal(x, bw_obj, bits)**:
    Converts a single bit-vector term to its decimal integer representation.

9. **bPrint(arr, bw_obj)**:
    Prints the values of the bit-vector terms in `arr`.

10. **bPrintFormula(trm)**:
    Prints the string representation of an SMT formula.

"""


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

Overview:
----------
This module contains functions that convert SystemVerilog hardware designs into SMT-LIB 2 format for formal validity check using Bitwuzla. 
The functions also handle the sanity checks to ensure that variable ranges are valid.

Functions:
-----------

1. **verilogSMT(name, module_name, state_vars, bits, inp_out_vars, spec_vars)**:
    Converts a Verilog design into SMT2 format, extracts state and non-state variables, and constructs Bitwuzla terms. It ensures that state variables and their next states are properly declared and adds range constraints to the generated SMT2 file.

    - `name` (str): Name of the hardware design.
    - `module_name` (str): Top-level module name in the Verilog design.
    - `state_vars` (dict): Dictionary of state variables with bounds and sizes.
    - `bits` (int): Bit-width for variables.
    - `inp_out_vars` (dict): Dictionary of input/output variables with metadata.
    - `spec_vars` (list): List of specification variables not involved in state changes.
    - Returns: A tuple `(bw_obj, curr_vars, next_vars, non_state_vars)` representing Bitwuzla objects and the current/next state variables and other variables.

2. **sanityCheckRange(bw_obj, rangesC, rangesN, curr_vars, next_vars)**:
    Ensures that the variable range is preserved across state transitions. It checks if the current state's range implies that the next state also satisfies the range constraints.

    - `bw_obj` (tuple): Tuple containing Bitwuzla-related objects (`tm`, `opt`, `parser`, `bvsizeB`).
    - `rangesC` (list): List of range assertions for the current state variables.
    - `rangesN` (list): List of range assertions for the next state variables.
    - `curr_vars` (list): List of current state variable terms.
    - `next_vars` (list): List of next state variable terms.
    - Returns: None. Prints whether the range is invariant or fails.

Process:
---------
1. **Verilog to SMT Conversion** (`verilogSMT`):
   - Generates the SMT2 file from the Verilog design using EBMC.
   - Modifies the SMT2 file to extend variables to have a uniform number of `bits` as they serve as inputs to the neural ranking function.
   - Creates Bitwuzla terms for state variables and applies range constraints on both current and next states.
   - Reads the SMT2 file and parses it to construct Bitwuzla terms for state variables and their next state equivalents.
   - Applies range constraints on input/output variables and state variables.

2. **Sanity Check** (`sanityCheckRange`):
   - Asserts that the current state satisfies the range constraints.
   - Asserts that if the current state satisfies the range, the next state must also satisfy the range.
   - Performs a satisfiability check to verify whether the range is preserved.

Returns:
---------
- **verilogSMT**: 
  - `bw_obj`: Tuple containing Bitwuzla objects for term creation, parsing, and optimization.
  - `curr_vars`: List of terms representing current state variables.
  - `next_vars`: List of terms representing next state variables.
  - `non_state_vars`: List of input/output variables not involved in state transitions.

- **sanityCheckRange**: 
  - Returns None, but prints whether the range is invariant across state transitions.

"""


def verilogSMT(name, module_name, state_vars, bits, inp_out_vars, spec_vars):
	os.system(f"../../Tools/ebmc/ebmc_v5_x ../../Benchmarks/{name}.sv --smt2 --bound 0 --top {module_name} --outfile ../../Tools/neuralmc/Model-SMT/{name}.smt2")
	smt2_model = ""
	with open(f"../../Tools/neuralmc/Model-SMT/{name}.smt2", 'r') as file:
	    smt2_model = file.read()
	# Remove the exit command
	smt2_model = '\n'.join([line for line in smt2_model.split("\n") if not any(rem_str in line for rem_str in ['(exit)', '; end of SMT2 file'])])
	
	state_changed = []
	for key, value in chain(state_vars.items(), inp_out_vars.items()):
		if key in spec_vars:
			continue
		curr_ = f"|Verilog::{module_name}.{key}@0|" 
		next_ = f"|Verilog::{module_name}.{key}@1|"
		curr_bv = curr_ if value['size'] > 1 else f"(ite {curr_} (_ bv1 1) (_ bv0 1))"
		next_bv = next_ if value['size'] > 1 else f"(ite {next_} (_ bv1 1) (_ bv0 1))"
		curr_B = f"|Verilog::{module_name}.{key}0|"
		next_B = f"|Verilog::{module_name}.{key}1|"
		if not (curr_ in smt2_model):
			if(value['type'] != 'output'):
				breakpoint()
			assert value['type'] == 'output'
		if not (next_ in smt2_model):
			if(value['type'] != 'input'):
				breakpoint()
			assert value['type'] == 'input'
		
		if(value['type'] == 'state'):
			assert curr_ in smt2_model and next_ in smt2_model
			smt2_1 = f"(declare-const {curr_B} (_ BitVec {bits}))\n(assert (= ((_ zero_extend {bits - value['size']}) {curr_bv}) {curr_B}))"
			smt2_2 = f"(declare-const {next_B} (_ BitVec {bits}))\n(assert (= ((_ zero_extend {bits - value['size']}) {next_bv}) {next_B}))"
			smt2_model += "\n" + smt2_1 + "\n" + smt2_2
			state_vars[key]['state_changed'] = True
		elif(value['type'] == 'input'):
			assert curr_ in smt2_model
			smt2_1 = f"(declare-const {curr_B} (_ BitVec {bits}))\n(assert (= ((_ zero_extend {bits - value['size']}) {curr_bv}) {curr_B}))"
			smt2_model += "\n" + smt2_1
			inp_out_vars[key]['state_changed'] = False
		elif(value['type'] == 'output'):
			assert next_ in smt2_model
			smt2_1 = f"(declare-const {curr_B} (_ BitVec {bits}))\n(assert (= ((_ zero_extend {bits - value['size']}) {next_bv}) {curr_B}))"
			smt2_1 = smt2_1.replace("Verilog::ReadWrite.RW@1","Verilog::ReadWrite.RW@0").replace("Verilog::ReadWrite.emp@1","Verilog::ReadWrite.emp@0").replace("Verilog::ReadWrite.ful@1","Verilog::ReadWrite.ful@0")
			smt2_model += "\n" + smt2_1
			inp_out_vars[key]['state_changed'] = False

	with open(f"../../Tools/neuralmc/Model-SMT/{name}.smt2", "w") as file:
	    file.write(smt2_model)
	curr_vars, next_vars, non_state_vars = [] , [] , []
	tm = bw.TermManager()
	opt = bw.Options()
	parser = bw.Parser(tm, opt)
	res = parser.parse(f"../../Tools/neuralmc/Model-SMT/{name}.smt2")
	bvsizeB = tm.mk_bv_sort(bits)
	rangesC = []
	rangesN = []
	bw_obj = (tm, opt, parser, bvsizeB)
	for key, value in chain(state_vars.items(), inp_out_vars.items()):
		if key in spec_vars:
			continue
		cv = parser.parse_term(f"|Verilog::{module_name}.{key}0|")
		if(value['state_changed']):
			nv = parser.parse_term(f"|Verilog::{module_name}.{key}1|")
		else:
			nv = tm.mk_var(bvsizeB, f"|Verilog::{module_name}.{key}1|")
		if key in state_vars.keys():
			curr_vars.append(cv)
			next_vars.append(nv)
			rangesN.append(rangeBitwuzla(next_vars[-1], bw_obj, value["lb"], value["ub"]))
			rangesC.append(rangeBitwuzla(curr_vars[-1], bw_obj, value["lb"], value["ub"]))
		else:
			non_state_vars.append(cv)
			rangesC.append(rangeBitwuzla(non_state_vars[-1], bw_obj, value["lb"], value["ub"]))

	sanityCheckRange(bw_obj, rangesC, rangesN, curr_vars, next_vars)
	parser.bitwuzla().assert_formula(bAnd(rangesC, bw_obj))
	parser.bitwuzla().assert_formula(bAnd(rangesN, bw_obj))
	for var in spec_vars:
		curr_vars.append(tm.mk_const(bvsizeB, f"{var}"))
		next_vars.append(tm.mk_const(bvsizeB, f"{var}1"))
	bw_obj = (tm, opt, parser, bvsizeB)
	os.system(f"rm ../../Tools/neuralmc/Model-SMT/{name}.smt2")
	return bw_obj, curr_vars, next_vars, non_state_vars

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
# 					SMT Check NRF using Bitwuzla
# =====================================================================

Overview:
----------
This module provides functions to verify neural ranking functions (NRFs) using the Bitwuzla SMT solver.
The main goal is to check two conditions (as specified in the paper) using SMT encodings of the neural ranking function and 
to formally check if the candidate ranking function is actually a ranking function, and if not use the satifying assignment
from the solver as a counterexample, which we use to further train the NRF.

Functions:
-----------

1. **acc_state_select(stt_acc, q_max, bw_obj, next_vars, rank_before)**:
    Generates a condition term that checks if the next state belongs to the set of accepted states (`stt_acc`).

2. **nacc_state_select(stt_acc, q_max, bw_obj, next_vars, rank_before)**:
    Generates a condition term that checks if the next state does NOT belong to the set of accepted states.

3. **cond1Check(...)**:
    Verifies Condition 1: The next state must belong to an accepted state, and the ranking value must decrease.

    - Performs SMT-based verification using Bitwuzla.
    - Prints the verification result.
    - Incase there is a counterexample, returns it.

4. **cond2Check(...)**:
    Verifies Condition 2: The next state must belong to a non-accepted state, and the ranking value must not increase.

    - Performs SMT-based verification using Bitwuzla.
    - Prints the verification result.
    - Incase there is a counterexample, returns it.

5. **bwPickPiece(bwpnrfs, Svars, bw_obj, q_c, q_max)**:
    Combines multiple Bitwuzla-encoded neural networks (dict `bwpnrfs`) using an `if-else` chain based on A¬Φ states.

    - `bwpnrfs` (dict): Dictionary of Bitwuzla-encoded NRFs.
    - `Svars` (list): List of state variables.
    - `q_c` (int): Current automata state.
    - `q_max` (int): Maximum automata state value.
    - Returns: Bitwuzla term representing the selected NRF for the current state.

6. **bwPNRF_encode(pnrf, Svars, F_prec, bw_obj, bits)**:
    Encodes the neural ranking functions (NRFs) as Bitwuzla terms using `bwPickPiece`

7. **nrfSMT_verify(...)**:
    Verifies the neural ranking function using both Condition 1 and Condition 2 in parallel using Bitwuzla.

    - Creates parallel processes to verify both conditions using `cond1Check(...)` and `cond2Check(...)`.
    - Returns a counterexample (`cex`) if any condition fails.

8. **print_nrf(nrf, curr_vars, F_prec, bw_obj, bits)**:
    Prints a single neural network as system verilog code.

"""


def acc_state_select(stt_acc, q_max, bw_obj, next_vars, rank_before):
	print(f"ACCEPT -> {stt_acc}")
	tm, opt, parser, bvsizeB = bw_obj
	cnds = [tm.mk_term(bw.Kind.EQUAL, [next_vars[-1], tm.mk_bv_value(bvsizeB, state)]) for state in stt_acc]
	return bOr(cnds, bw_obj)

def nacc_state_select(stt_acc, q_max, bw_obj, next_vars, rank_before):
	stt_nacc = [x for x in range(0,q_max+1) if x not in stt_acc]
	print(f"NOT ACCEPT -> {stt_nacc}")
	tm, opt, parser, bvsizeB = bw_obj
	cnds = [tm.mk_term(bw.Kind.EQUAL, [next_vars[-1], tm.mk_bv_value(bvsizeB, state)]) for state in stt_nacc]
	return bOr(cnds, bw_obj)

def cond1Check(pnrf, bw_obj, curr_vars, next_vars, non_state_vars, F_prec, bits, pnum, rank_before, rank_after, cex1, idtext, stt_acc, q_max):
    tm, opt, parser, bvsizeB = bw_obj
    beginV = time.time()
    parser.bitwuzla().push()
    assert int(math.floor(delta*(2**(F_prec-1)))) > 0
    dt = tm.mk_bv_value(bvsizeB, int(math.floor(delta/5*(2**(F_prec-1)))))
    cnd1_pre = acc_state_select(stt_acc, q_max, bw_obj, next_vars, rank_before) #tm.mk_term(bw.Kind.EQUAL, [curr_vars[-1], tm.mk_bv_value(bvsizeB, 1)])
    cnd1_post = tm.mk_term(bw.Kind.BV_SLE, [tm.mk_term(bw.Kind.BV_SUB, [rank_before, dt]), rank_after])
    parser.bitwuzla().assert_formula(bAnd([cnd1_pre, cnd1_post], bw_obj))
    res = parser.bitwuzla().check_sat()
    endV = time.time()
    print(f"{colours[pnum % col_num]} [{pnum}] Time VERIFY 1: {endV - beginV}{Style.RESET_ALL}")
    if(res == bw.Result.UNSAT):
        print(f"{colours[pnum % col_num]} [{pnum}] Condition 1 [PASS]{Style.RESET_ALL}")
        cex1.put(None)
    elif(res == bw.Result.SAT):
        print(f"{colours[pnum % col_num]} [{pnum}] Condition 1 [FAIL]{Style.RESET_ALL}")
        bPrint(curr_vars, bw_obj)
        bPrint(next_vars, bw_obj)
        bPrint(non_state_vars, bw_obj)
        cntexm = (bv2int(curr_vars, bw_obj, bits), bv2int(next_vars, bw_obj, bits))
        print(f"Torch Rank [{pnrf[cntexm[0][-1]](torch.tensor(cntexm[0]))} -> {pnrf[cntexm[1][-1]](torch.tensor(cntexm[1]))}];\nBitwuzla Rank [{todecimal(rank_before, bw_obj, bits)/2**F_prec} -> {todecimal(rank_after, bw_obj, bits)/2**F_prec}]")
        cex1.put(cntexm)
        #print(todecimal(rank_before, bw_obj, bits)/2**F_prec)
        #print(parser.bitwuzla().get_value(cnd1_pre))
    else:
        print("Bitwuzla output: UNKNOWN")
        breakpoint()
    parser.bitwuzla().pop()



def cond2Check(pnrf, bw_obj, curr_vars, next_vars, non_state_vars, F_prec, bits, pnum, rank_before, rank_after, cex2, idtext, stt_acc, q_max):
   
    tm, opt, parser, bvsizeB = bw_obj
    beginV = time.time()
    parser.bitwuzla().push()
    cnd2_pre = nacc_state_select(stt_acc, q_max, bw_obj, next_vars, rank_before) #tm.mk_term(bw.Kind.EQUAL, [curr_vars[-1], tm.mk_bv_value(bvsizeB, 0)])
    cnd2_post = tm.mk_term(bw.Kind.BV_SLT, [rank_before, rank_after])
    parser.bitwuzla().assert_formula(bAnd([cnd2_pre, cnd2_post], bw_obj))
    res = parser.bitwuzla().check_sat()
    endV = time.time()
    print(f"{colours[pnum % col_num]} [{pnum}] Time VERIFY 2: {endV - beginV}{Style.RESET_ALL}")
    if(res == bw.Result.UNSAT):
        print(f"{colours[pnum % col_num]} [{pnum}] Condition 2 [PASS]{Style.RESET_ALL}")
        cex2.put(None)
    elif(res == bw.Result.SAT):
        print(f"{colours[pnum % col_num]} [{pnum}] Condition 2 [FAIL]{Style.RESET_ALL}")
        bPrint(curr_vars, bw_obj)
        bPrint(next_vars, bw_obj)
        bPrint(non_state_vars, bw_obj)
        cntexm = (bv2int(curr_vars, bw_obj, bits), bv2int(next_vars, bw_obj, bits))
        print(f"Torch Rank [{pnrf[cntexm[0][-1]](torch.tensor(cntexm[0]))} -> {pnrf[cntexm[1][-1]](torch.tensor(cntexm[1]))}];\nBitwuzla Rank [{todecimal(rank_before, bw_obj, bits)/2**F_prec} -> {todecimal(rank_after, bw_obj, bits)/2**F_prec}]")
        cex2.put(cntexm)
    else:
        print("Bitwuzla output: UNKNOWN")
        breakpoint()
    parser.bitwuzla().pop()

def bwPickPiece(bwpnrfs, Svars, bw_obj, q_c, q_max):
	tm, opt, parser, bvsizeB = bw_obj
	if q_c == q_max:
		return bwpnrfs[q_c]
	return tm.mk_term(bw.Kind.ITE, [tm.mk_term(bw.Kind.EQUAL, [Svars[-1], tm.mk_bv_value(bvsizeB, q_c)]), bwpnrfs[q_c], bwPickPiece(bwpnrfs, Svars, bw_obj, q_c + 1, q_max)])


def bwPNRF_encode(pnrf, Svars, F_prec, bw_obj, bits, SMTencode):
	if SMTencode == "new":
		bwpnrfs = {q: bwEBV.encode(pnrf[q], Svars, F_prec, bw_obj, bits)[0] for q in pnrf}
	elif SMTencode == "old":
		bwpnrfs = {q: bwEBV.old_encode(pnrf[q], Svars, F_prec, bw_obj, bits)[0] for q in pnrf}
	else:
		breakpoint()
	q_max = max(pnrf.keys())
	return bwPickPiece(bwpnrfs, Svars, bw_obj, 0, q_max)


def nrfSMT_verify(pnrf, bw_obj, curr_vars, next_vars, non_state_vars, F_prec, bits, pnum, idtext, stt_acc, q_bits, q_max, SMTencode):
    tm, opt, parser, bvsizeB = bw_obj
    rank_before = bwPNRF_encode(pnrf, curr_vars, F_prec, bw_obj, bits, SMTencode)
    rank_after = bwPNRF_encode(pnrf, next_vars, F_prec, bw_obj, bits, SMTencode)

    cex1 = multiprocessing.Queue()
    cex2 = multiprocessing.Queue()
    
    process1 = multiprocessing.Process(target=cond1Check, args=(pnrf, bw_obj, curr_vars, next_vars, non_state_vars, F_prec, bits, pnum, rank_before, rank_after, cex1, idtext, stt_acc, q_max))
    process2 = multiprocessing.Process(target=cond2Check, args=(pnrf, bw_obj, curr_vars, next_vars, non_state_vars, F_prec, bits, pnum, rank_before, rank_after, cex2, idtext, stt_acc, q_max))
    process1.start()
    process2.start()
    process1.join()
    process2.join()
    cex = [cex1.get(), cex2.get()]
    return cex

def print_nrf(nrf, curr_vars, F_prec, bw_obj, bits):
	rnk_exp = bwEBV.encodeAsSTR(nrf, curr_vars, F_prec, bw_obj, bits)
	print(rnk_exp[0])
	print(rnk_exp[1])
    


"""

# =====================================================================
# 						TEST Bitwuzla Encoding
# =====================================================================

Overview:
----------
This module provides utility functions to test the consistency between the PyTorch and Bitwuzla
encodings of the neural network. Additionally, we verify that all transitions in the dataset
are valid assignments to the SystemVerilog model represented in SMT. This establishes the consistency
between the Verilator (simulation) and the EBMC (SMT) interpretation of the SystemVerilog model.

Functions:
-----------

1. **bwSetListEqual(l1, l2, bw_obj)**:
    Creates SMT terms that assert SMT equality between two lists of bit-vectors `l1` and `l2`.

    - `l1` (list): List of Bitwuzla terms (bit-vectors).
    - `l2` (list): List of integers to compare against.
    - `bw_obj` (tuple): Tuple containing Bitwuzla-related objects (`tm`, `opt`, `parser`, `bvsizeB`).
    - Returns: A list of SMT terms.

2. **TestEncoder(pnrf, bw_obj, curr_vars, next_vars, F_prec, bits, all_trans_tensors)**:
    Tests the consistency of the Bitwuzla NRF encoding across state transitions in the dataset by comparing the Bitwuzla results with the PyTorch-based NRF.

Process:
---------
1. **bwSetListEqual**:
    - Constructs equality constraints for each element of `l1` and `l2`, ensuring they match in the Bitwuzla model.

2. **TestEncoder**:
    - Iterates over all state transitions in `all_trans_tensors`.
    - For each transition, it:
      - Asserts equality of `curr_vars` to the "before" state and `next_vars` to the "after" state.
      - Check if it is SAT, meaning the dataset point is a valid transition for the SystemVerilog model encoded in SMT.
      - Verifies the consistency between Bitwuzla's ranking result and the PyTorch-based NRF.
      - Breaks execution if inconsistencies are detected.
Notes:
-------
- The `TestEncoder` function uses a tolerance for floating-point comparisons (`rel_tol`), allowing small deviations due to fixed-point precision.
"""

def bwSetListEqual(l1, l2, bw_obj):
	tm, opt, parser, bvsizeB = bw_obj
	res = []
	for i in range(len(l1)):
		res.append(tm.mk_term(bw.Kind.EQUAL, [l1[i], tm.mk_bv_value(bvsizeB, int(l2[i]))]))
	return res

def TestEncoder(pnrf, bw_obj, curr_vars, next_vars, F_prec, bits, all_trans_tensors, SMTencode):
    tm, opt, parser, bvsizeB = bw_obj
    rank_before = bwPNRF_encode(pnrf, curr_vars, F_prec, bw_obj, bits, SMTencode)
    rank_after = bwPNRF_encode(pnrf, next_vars, F_prec, bw_obj, bits, SMTencode) 
    for q, trans in all_trans_tensors.items():
        for bef, aft in zip(trans[:,0], trans[:,1]):
        	#bef = [18656 , 1]
        	#aft = [18657, 1]
        	print(f"===========\n{bef}\n{aft}\n")
        	parser.bitwuzla().push()
        	parser.bitwuzla().assert_formula(bAnd(bwSetListEqual(curr_vars, bef, bw_obj), bw_obj))
        	parser.bitwuzla().assert_formula(bAnd(bwSetListEqual(next_vars, aft, bw_obj), bw_obj))
        	res = parser.bitwuzla().check_sat()
        	if(res == bw.Result.UNSAT):
        		print("ERROR!!!")
        		breakpoint()
        	#breakpoint()
        	print(f"{pnrf[bef[-1].item()](bef)} -> {pnrf[aft[-1].item()](aft)}")
        	print(f"{todecimal(rank_before, bw_obj, bits)/2**F_prec} -> {todecimal(rank_after, bw_obj, bits)/2**F_prec}")
        	
        	if(not math.isclose(todecimal(rank_before, bw_obj, bits)/2**F_prec, pnrf[bef[-1].item()](bef), rel_tol = (1000*2**-F_prec))):
        	    breakpoint()
        	if(not math.isclose(todecimal(rank_after, bw_obj, bits)/2**F_prec, pnrf[aft[-1].item()](aft), rel_tol = (1000*2**-F_prec))):
        	    breakpoint()       
        	
        	parser.bitwuzla().pop()




"""
# =====================================================================
# 			 Run Neural Model Checking (main-functions)
# =====================================================================

[Training multiple networks in parallel using multiprocessing is deprecated, as it was deemed unnecessary. However, the code remains intact to facilitate future extensions and experimentation.]

Overview:
----------
The `train_an_nrf` function trains a Neural Ranking Function (NRF) using provided state transitions and verifies the trained model using the Bitwuzla SMT solver. It runs in a loop, iteratively updating the NRF parameters and checking for counterexamples until a valid ranking function is found or the training fails.

Parameters:
------------
- `bw_obj` (tuple):  
  Contains Bitwuzla-related objects (`tm`, `opt`, `parser`, `bvsizeB`).

- `curr_vars` (list):  
  List of current state variables in Bitwuzla format.

- `next_vars` (list):  
  List of next state variables in Bitwuzla format.

- `non_state_vars` (list):  
  List of non-state variables in Bitwuzla format.

- `all_trans_tensors` (dict):  
  The dataset consists of transition pairs. The elements are organized into a dictionary
  keyed by the next A¬Φ state for debugging convenience.

- `all_states` (torch.Tensor):  
  Tensor of all states used during training.

- `result_queue` (multiprocessing.Queue):  
  Queue used to communicate the result (success/failure) from the process back to the main process.

- `exit_event` (multiprocessing.Event):  
  Event to signal that the process has finished.

- `pnum` (int):  
  Process number (used for logging and process identification).

- `F_prec` (int):  
  Fixed-point precision used for encoding and ranking.

- `bits` (int):  
  Bit-width for the Bitwuzla terms.

- `clamp_bits` (int):  
  Number of bits used for clamping in the neural network.

- `nnP` (str):  
  Neural network architecture parameter defining layer sizes.

- `idtext` (str):  
  Identifier text for logging purposes.

- `lr` (float):  
  Learning rate for the optimizer.

- `stt_acc` (list):  
  List of accepted states.

- `q_bits` (int):  
  Number of bits in the automata state `q`.

- `q_max` (int):  
  Maximum value for the automata state `q`.

Process:
---------
1. **Initialization**:  
   - Sets the random seed for reproducibility.
   - Initializes the NRF models (`pnrf`) and corresponding optimizers (`optimiserPNRF`).

2. **Training Loop**:  
   - The training loop continuously updates the NRF using state transitions (`all_trans_tensors`).
   - If the training loss is not zero, the process is aborted.
   - After training, the NRF is verified using the `nrfSMT_verify` function.
   - If no counterexamples are found, the NRF is considered valid and the process is marked as successful.

3. **Counterexample Handling**:  
   - If counterexamples are found during verification, the relevant state transitions are added to the training set and the loop continues.

4. **Completion**:  
   - Once a valid ranking function is found or training fails, the result is communicated back to the main process through the `result_queue`.
   - The `exit_event` is set to signal the process has completed.

Notes:
-------
- The function can be (BUT IS NOT in our experiments) runs in parallel with other training processes and uses multiprocessing to handle inter-process communication.
- If `TestEncoder` is set to `True`, the function tests the SMT encoding before beginning the training process.
"""


def train_an_nrf(bw_obj, curr_vars, next_vars, non_state_vars, all_trans_tensors, all_states, result_queue, exit_event, pnum, F_prec, bits, clamp_bits, nnP, idtext, lr, stt_acc, q_bits, q_max, SMTencode):
    print(f"{colours[pnum % col_num]}Process {pnum} started with colour {Style.RESET_ALL}")
    TestEncoding = False
    success = False
    seed = pnum
    torch.manual_seed(seed)
    random.seed(seed)
    begin = time.time()
    scale = torch.max(all_states,0).values / norm_range
    if nnP == "Monolithic":
    	print("\t\t\t---MONOLITHIC NEURAL NETWORK (NOT RECOMMENDED)---")
    	nn_mono = neuralrf.NRF_Clamp(len(scale), scale, clamp_bits, "A3-Default")
    	pnrf = {q: nn_mono for q in range(q_max+1)}
    else:
    	pnrf = {q: neuralrf.NRF_Clamp(len(scale), scale, clamp_bits, nnP) for q in range(q_max+1)}
    optimiserPNRF = {q: optim.AdamW(pnrf[q].parameters(), lr=lr, weight_decay=.01) for q in pnrf}
    if TestEncoding: 
    	TestEncoder(pnrf, bw_obj, curr_vars, next_vars, F_prec, bits, all_trans_tensors, SMTencode)
    	TestEncoder(pnrf, bw_obj, curr_vars, next_vars, F_prec, bits, all_trans_tensors, SMTencode)
    try:
    	while(True):
    		loss = neuralrf.training(pnrf, optimiserPNRF, all_trans_tensors, all_states, stt_acc, pnum, delta)
    		#breakpoint()
    		if(loss != 0.0):
    			print(f"Training Failed (Aborting)")
    			break
    		cex = nrfSMT_verify(pnrf, bw_obj, curr_vars, next_vars, non_state_vars, F_prec, bits, pnum, idtext, stt_acc, q_bits, q_max, SMTencode)
    		#print(f"VERIFICATION IS SWITCHED OFF")
    		#break
    		if (cex[0] == None and cex[1] == None):
    			print(f"{colours[pnum % col_num]}Yay!!! We have a Ranking function{Style.RESET_ALL}")
    			success = True
    			break
    		if(cex[0] != None):
    			q_cex = cex[0][1][-1]
    			all_trans_tensors[q_cex] = torch.cat((all_trans_tensors[q_cex], torch.Tensor([cex[0]])))
    		if(cex[1] != None):
    			q_cex = cex[1][1][-1]
    			all_trans_tensors[q_cex] = torch.cat((all_trans_tensors[q_cex], torch.Tensor([cex[1]])))
    	print(f"Result from the first finishing process: {pnrf}")    
    	result_queue.put(success)
    	print(f"{colours[pnum % col_num]}Process {pnum} Ended.{Style.RESET_ALL}")
    	#print_nrf(nrf, curr_vars, F_prec, bw_obj, bits)
    finally:
    	exit_event.set()  # Signal that the process has finished

def runExperiment( name, bw_obj, curr_vars, next_vars, non_state_vars, F_prec, bits, clamp_bits, nnP, idtext, lr, stt_acc, q_bits, q_max, SMTencode = "new"):
    seed = 2
    torch.manual_seed(seed)
    random.seed(seed)
    cores = 8
    begin = time.time()
    run_verilator(name)
    all_trans_tensors, all_states = trace_file_as_tensor(name, stt_acc, q_max)
    result_queue = multiprocessing.Queue()
    exit_event = multiprocessing.Event()

    pnum = seed
    train_an_nrf( bw_obj, curr_vars, next_vars, non_state_vars, all_trans_tensors, all_states, result_queue, exit_event, pnum, F_prec, bits, clamp_bits, nnP, idtext, lr, stt_acc, q_bits, q_max, SMTencode)
    end = time.time()
    return


