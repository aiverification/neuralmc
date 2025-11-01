"""
This function wraps the invocation of the EBMC hardware model checker for a
given Verilog design:

Function:
  runEBMC(name, module_name, nuXmvSpec, idtext, SVSpec=None)

Purpose:
  1. Validate that a SystemVerilog specification `SVSpec` is provided.
  2. Clean up any previous SMV file for the design under `Tools/nuXmv/SMV/`.
  3. Read the original Verilog file from `Benchmarks/`.
  4. Inject an `assert property` block with the given temporal spec (`SVSpec`) before
     the `endmodule` keyword.
  5. Write the modified Verilog to `Tools/nuXmv/SMV/{name}.sv`.
  6. Invoke EBMC via a system call (`ebmc --bdd`) on the modified file.
  7. Measure and return the total execution time of the EBMC call.

Returns:
    ebmcT (float):    Elapsed wall-clock time (seconds) for the EBMC run.
"""

import os
import time
import re

def runEBMC(name, module_name, nuXmvSpec, idtext, SVSpec = None):
	assert SVSpec is not None
	begin = time.time()
	os.system(f"rm ../../Tools/nuXmv/SMV/{name}.smv")
	with open(f"../../../Benchmarks/{name}.sv", 'r') as file:
	    data = file.read()
	
	prop = f"\tp1: assert property  ({SVSpec}) ;\nendmodule"
	data = re.sub("endmodule", prop, data)
	with open(f"../../Tools/nuXmv/SMV/{name}.sv", "w") as file:
		file.write(data)
	print(data)
	print(f"cmd> ebmc ../../Tools/nuXmv/SMV/{name}.sv --bdd")
	os.system(f"ebmc ../../Tools/nuXmv/SMV/{name}.sv --bdd")
	end = time.time()
	ebmcT = end - begin
	print(f"\n\nEBMC TIME: {ebmcT}\n")
	return ebmcT

