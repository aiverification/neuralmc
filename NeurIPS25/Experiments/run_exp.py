"""
exec_exp: Run classical and neural model checking experiments.

Prompts to select a backend—EBMC, ABC, nuXmv, rIC3, or neural (auto/Linear)—and
configures options for learning engines if needed. For classical choices, it
runs each design and logs timing. For neural choices, it
asks for the solver, sample counts (ablation study), bound-option then loops over designs:
  1. Extracts variable metadata (readForVars)
  2. Converts Verilog to SMT (verilogSMT)
  3. Trains and verifies neural certificates (runExperiment)

Outputs: writes results to data.txt and prints progress to stdout.
"""
def exec_exp(start_ex, N_lims, CBITSs, specTXT, module_name, file_name, LTLSpec, SVSpec, range_vals_list, scale, size, gap, F_prec, bits, is_acc, is_init, init_samp, spec_automata, Ps, Ms):
	import sys
	import math
	import time
	import itertools
	import numpy as np
	
	print("1) EBMC 2)ABC 3)nuXmv 4)nuXmv(IC3) 5)our(auto) 6)our(Linear)")
	ch = int(input("Enter your choice: "))
	sys.path.append('../../')
	
	if ch == 1:
		import Tools.ebmc.ebmc_run as ebmc_run
	if ch == 2:
		import Tools.abc_mc.abc_run as abc_run
	elif ch in {3,4}:
		import Tools.nuXmv.nuxmv_run as nuxmv_run
	else:
		import Tools.neuralmc.cav_nuR as nuR
	isAuto = (ch == 5)
	cmp_times = []
	if ch in [1, 2, 3, 4]:
		for dut_i in range(start_ex, len(N_lims)):
			name = f"{file_name}_{dut_i+1}"
			N_lim = N_lims[dut_i]
			idtxt = f"{name} ({specTXT}) {N_lim}"
			print(idtxt)
			if ch == 1:
				ebmcT = ebmc_run.runEBMC(name, module_name, LTLSpec, idtxt, SVSpec)
				cmp_times.append(ebmcT)
			elif ch == 2:
				abcT = abc_run.runABC(name, module_name, LTLSpec, idtxt, SVSpec)
				cmp_times.append(abcT)
			else: 
				nuXT = nuxmv_run.runNuXmv(name, module_name, LTLSpec, idtxt, SVSpec, (ch == 4))
				cmp_times.append(nuXT)
			with open(f"data.txt", "a") as file:
				file.write(f"{idtxt}\n")
				chs = "EBMC" if ch == 1 else "ABC" if ch == 2 else "nuXmv --BDD" if ch == 3 else "nuXmv --IC3"
				file.write(f"{chs}:[{cmp_times}]\n\n\n\n")
			sys.stdout.flush()
			continue
		exit()
	choice_map = { 1: "gurobi", 2: "cvc5", 3: "z3", 4: "msat"}
	print(f"Learning Engine: 1) gurobi 2) cvc5 3) z3 4) msat")
	ch = int(input("Enter your choice: "))
	rnd_smpC = int(input("No of Initial Random Samples: "))
	engine = choice_map.get(ch, None)
	if engine != "gurobi":
		ch = input("Switch to Unbounded parameters? (y/n): ")
		if (ch in 'yY1') :
			Ps = [None]*len(Ps)
	gu_times, bw_times, total_times, guess_cnts = [], [], [], []
	for dut_i in range(start_ex, len(N_lims)):
		begin = time.time()
		N_lim = N_lims[dut_i]
		P = Ps[dut_i]
		M = Ms[dut_i]
		kappa = None#N_lim
		hyperparameters = (scale, P, size, gap, M, kappa)
		name = f"{file_name}_{dut_i+1}"
		idtxt = f"{name} ({specTXT}) {N_lim}"
		print(f"\t\t\t\t {idtxt}\n\t\t\t\t")
		range_vals = iter(range_vals_list)
	
		#nuR.getLGC_SSS(name, module_name, idtxt)
		#exit()
		
		state_vars, inp_out_vars = nuR.readForVars(name, module_name, range_vals)
		        
		bw_obj, curr_vars, next_vars, non_state_vars, state_names = nuR.verilogSMT(name, module_name, state_vars, bits, inp_out_vars)
		ctx = state_vars, inp_out_vars, bw_obj, bits
		gu_time, bw_time, guess_cnt, size_success = nuR.runExperiment(name, hyperparameters, bw_obj, curr_vars, next_vars, non_state_vars, spec_automata, ctx, F_prec, bits, is_acc, init_samp, is_init, state_vars, state_names, rnd_smpC, engine, isAuto)
		end = time.time()
		print(f"BITS ---------->>>>>>>>> {bits} {idtxt} E: {engine} P: {P} RndSmps: {rnd_smpC} isAuto: {isAuto} Arch: {size_success}")
		print(f'Learn Time: {gu_time}; Check Time: {bw_time}; Guess cnt: {guess_cnt}')
		print(f"Total Time: {end - begin}\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
		gu_times.append(gu_time)
		bw_times.append(bw_time)
		if gu_times is None:
			total_times.append(None)
		else:
			total_times.append(end-begin)
		guess_cnts.append(guess_cnt)
		with open(f"data.txt", "a") as file:
			file.write(f"BITS ---------->>>>>>>>> {bits} {idtxt} E: {engine} P: {P} RndSmps: {rnd_smpC} isAuto: {isAuto} Arch: {size_success}\n")
			file.write(f'Gurobi Time: {gu_time}; Bitwuzla Time: {bw_time}\n')
			file.write(f"Total Time: {end - begin}\nLearn Time:[{gu_times}]\nCheck Time:[{bw_times}]\nGuess Count: {guess_cnts}\nTotal Time:[{total_times}]\n\n\n\n")
		print("Start 2 sec Sleep")
		#time.sleep(2)  # Pause for 2 seconds