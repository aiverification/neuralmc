import math
import time
import itertools

lrs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
q_skews = [50, 50, 5, 5, 5, 5, 5, 5, 5]
sizes = [(20000, 100), (5000, 100), (5000, 100), (5000, 100), (5000, 100), (5000, 100), (10000, 50), (10000, 50), (10000, 50), (10000, 50)]
caps = [10, 7, 7, 7, 7, 7, 7, 7, 7]
Sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
specTXT = "F G ! rst-> G F disp_ena"
network_type = "A3-Default"
SMTencode = "new"
module_name = "VGA"

import math
import time
import itertools
import sys

print("1)ABC 2)nuXmv 3)our")
ch = int(input("Enter your choice: "))
sys.path.append('../../')
if ch == 1:
	import Tools.abc_mc.abc_run as abc_run
elif ch == 2:
	import Tools.nuXmv.nuxmv_run as nuxmv_run
else:
	import Tools.neuralmc.nuR as nuR
	
if ch in [1, 2]:
	for dut_i in range(len(Sizes)):
		name = f"vga_{dut_i+1}"
		size = Sizes[dut_i]
		idtxt = f"{name} ({specTXT}) {size}"
		print(idtxt)
		LTLSpec = "LTLSPEC F G (Verilog.VGA.rst = FALSE) -> G F (Verilog.VGA.disp_ena = TRUE)"
		if ch == 1:
			abc_run.runABC(name, module_name, LTLSpec, idtxt)
		else: 
			nuxmv_run.runNuXmv(name, module_name, LTLSpec, idtxt)
		sys.stdout.flush()
		continue
	exit()


for dut_i in range(len(Sizes)):
	lr = lrs[dut_i]
	q_skew = q_skews[dut_i]
	smp, trace_len = sizes[dut_i]
	cap = caps[dut_i]
	size = Sizes[dut_i]

	begin = time.time()

	range_vals = []
	if(dut_i == 0):
		range_vals = iter(['n', 'n', 'y', '0', '66', 'n', 'y', '0', '28', 'n', 'n', 'y', '50', 'y', '0', '66', 'y', '0', '28'])
	if(dut_i == 1):
		range_vals = iter(['y', 0, 66*size-1, 'n', 'n', 'y', 0, 28*size-1, 'n', 'n', 'n',  'y', 0, 66*size-1, 'y', 50*size-1, 'y', 0, 28*size-1])
	if(dut_i == 2):
		range_vals = iter(['Y', 0 , 28*size-1, 'n', 'n', 'y', '0', 66*size-1, 'n', 'n', 'y', '0', 66*size-1, 'y', '0', 28*size-1, 'n', 'y', '50'])
	if(dut_i == 3):
		range_vals = iter(['y', '0', 28*size-1, 'y', '0', 66*size-1, 'n', 'n', 'n', 'y', '0', 66*size-1, 'n', 'n', 'y', '0', 28*size-1, 'y', '50'])
	if(dut_i == 4):
		range_vals = iter(['n', 'n', 'y', '0', 66*size-1, 'y', '0', 66*size-1, 'n', 'y', '0', 28*size-1, 'n', 'y', '0', 28*size-1, 'n', 'y', '50'])
	if(dut_i == 5):
		range_vals = iter(['y','0',66*size-1,'n','n','y','0',28*size-1,'n','n','n','y','0',66*size-1,'y','50 ','y','0',28*size-1])
	if(dut_i == 6):
		range_vals = iter(['y','0',66*size-1,'n','y',0,66*size-1,'n','n','y','0',28*size-1,'n','y',0, 28*size-1,'n','y',50])
	if(dut_i == 7):
		range_vals = iter(['y', '0', 28*size-1, 'y', '0', 66*size-1, 'n', 'n', 'y', '0', 66*size-1, 'n', 'n', 'n', 'y', '0', 28*size-1, 'y', '50'])
	if(dut_i == 8):
		range_vals = iter(['y', '0', 28*size-1, 'y', '0', 66*size-1, 'n', 'n', 'n', 'y', '0', 66*size-1, 'n', 'n', 'y', '0', 28*size-1, 'y', '50'])
	
	range_vals = iter(['y', 0, 66*size-1, 'n', 'y', 0, 66*size-1, 'n', 'n', 'n',  'y', 0, 28*size-1, 'y', 50, 'y', 0, 28*size-1, 'n'])

	name = f"vga_{dut_i+1}"
	smt_file_name = f"1_{name}_{network_type}_{SMTencode}"
	module_name = "VGA"
	F_prec = 14
	idtxt = f"{name} {module_name} {size}"
	print(f"\t\t\t\t {idtxt}\n\t\t\t\t ({lr}, {q_skew}, {network_type}, ({smp}, {trace_len}), {cap})")
	
	state_vars, inp_out_vars = nuR.readForVars(name, module_name, range_vals)
	next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
	clamp_bits = max(int(math.log2(2**max([value['size'] for value in state_vars.values()])/10000)),0) + cap
	bits = max(max([value['size'] for value in state_vars.values()]), clamp_bits + 4) + F_prec + 5  # 2*F_prec as its needed for quant dot product, the max(max(), 8) as clamp is 2**8
	spec_str = ["isValid = 1;",
		   "if (q == 0 && (disp_ena != 0 && rst == 0)) begin",
		   	f"\tif ($urandom % {q_skew} == 0)",
		   	 	"\t\tq = 1;",
		   	 "\telse",
		   	 	"\t\tq = 0;",
		   	 "end",
		   "else if (q == 0)",
		   	"\tq = 0;",
		   "else if (q == 1 && (disp_ena!= 0 && rst == 0))",
		   	"\tq = 1;", 
		   "else",
		   	"\tisValid = 0;"]
	stt_acc = [1]
	q_bits = 1
	q_max = 1
	nuR.makeVerilogSampler(name, module_name, smp, trace_len, spec_str, state_vars, inp_out_vars, stt_acc, q_bits, q_max)
	# We explicitly state the spec_vars, as its not part of the verilog
	bw_obj, curr_vars, next_vars, non_state_vars = nuR.verilogSMT(name, module_name, state_vars, bits, inp_out_vars, spec_vars = ['q'])
	ctx = state_vars, inp_out_vars, bw_obj, bits
	spec = [ [nuR.Bset(curr_vars, 'q', 0, ctx), nuR.Bset(next_vars, 'q',0, ctx)],
			 [nuR.Bset(curr_vars, 'q', 0, ctx), nuR.BUnSet(non_state_vars, 'disp_ena', 0, ctx), nuR.Bset(non_state_vars, 'rst', 0, ctx), nuR.Bset(next_vars, 'q', 1, ctx)],
			 [nuR.Bset(curr_vars, 'q', 1, ctx), nuR.BUnSet(non_state_vars, 'disp_ena', 0, ctx), nuR.Bset(non_state_vars, 'rst', 0, ctx), nuR.Bset(next_vars, 'q', 1, ctx)]]	
	
	bw_obj[2].bitwuzla().assert_formula(nuR.bOrOfAnd(spec, bw_obj))
	nuR.runExperiment(name, bw_obj, curr_vars, next_vars, non_state_vars, F_prec, bits, clamp_bits, network_type, idtxt, lr, stt_acc, q_bits, q_max, SMTencode, smt_file_name)
	end = time.time()
	print(f"BITS ---------->>>>>>>>> {bits} {idtxt}")
	print(f"Total Time: {end - begin}")

'''
T = FG RST -> GF good

T = ~(FG RST) or (GF good)

~T = ~(~(FG RST) or (GF good)) 
   = FG RST and FG not good
'''
#527, 223
# 0 - 1 iter(['n', 'n', 'y ', '0', '28', 'n', 'y', '0', '66', 'n', 'n', 'y', '50', 'y', '0', '66', 'y', '0', '28'])

# 1 - 2 iter(['y', 0, 66*size-1, 'n', 'n', 'y', 0, 28*size-1, 'n', 'n', 'n',  'y', 0, 66*size-1, 'y', 50*size-1, 'y', 0, 28*size-1])

# 2 - 3 iter(['Y', 0 , 28*size-1, 'n', 'n', 'y', '0', 66*size-1, 'n', 'n', 'y', '0', 66*size-1, 'y', '0', 28*size-1, 'n', 'y', '50'])

# 3 - 4 iter(['y', '0', 28*size-1, 'y', '0', 66*size-1, 'n', 'n', 'n', 'y', '0', 66*size-1, 'n', 'n', 'y', '0', 28*size-1, 'y', '50'])

# 4 - 5 iter(['n', 'n', 'y', '0', 66*size-1, 'y', '0', 66*size-1, 'n', 'y', '0', 28*size-1, 'n', 'y', '0', 28*size-1, 'n', 'y', '50'])

# 5 - 6 iter(['y','0',66*size-1,'n','n','y','0',28*size-1,'n','n','n','y','0',66*size-1,'y','50 ','y','0',28*size-1])

# 6 - 8 iter(['y','0',66*size-1,'n','y',0,66*size-1,'n','n','y','0',28*size-1,'n','y',0, 28*size-1,'n','y',50])

# 7 - 10 iter(['y', '0', 28*size-1, 'y', '0', 66*size-1, 'n', 'n', 'y', '0', 66*size-1, 'n', 'n', 'n', 'y', '0', 28*size-1, 'y', '50'])

# 8 - 12 iter(['y', '0', 28*size-1, 'y', '0', 66*size-1, 'n', 'n', 'n', 'y', '0', 66*size-1, 'n', 'n', 'y', '0', 28*size-1, 'y', '50'])



