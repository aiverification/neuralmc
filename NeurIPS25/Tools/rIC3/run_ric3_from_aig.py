#!/usr/bin/env python3
"""
Run 16-threads Portfolio rIC3 on each *.aig file in a directory,
log full stdout+stderr and time taken into liveness_full.log.
"""

import subprocess, time, sys
from pathlib import Path
import re

TOOL        = "rIC3"
TIMEOUT     = 1000  # seconds
LOGFILE     = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path.cwd()
TARGET_DIR  = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()

def extract_number(path):
    match = re.search(r'(\d+)(?=\.aig$)', path.name)
    return int(match.group(1)) if match else float('inf')

with open(LOGFILE, "w") as log:
    print( sorted(TARGET_DIR.glob("*.aig"), key=extract_number))
    for aig in sorted(TARGET_DIR.glob("*.aig"), key=extract_number):
        log.write(f"\n=== Running {aig.name} ===\n")
        print(f"Running {aig.name}...")

        start = time.time()
        try:
            completed = subprocess.run([TOOL, str(aig)],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           universal_newlines=True,   
                           timeout=TIMEOUT)

            elapsed = time.time() - start
            #log.write(completed.stdout)
            log.write(f"\n[RESULT] {aig.name}: OK in {elapsed:.6f} sec\n")
        except subprocess.TimeoutExpired as e:
            elapsed = TIMEOUT
            #if e.stdout:
                #log.write(e.stdout.decode('utf‑8', errors='replace')
                         #if isinstance(e.stdout, (bytes, bytearray))
                         # else e.stdout)
            log.write(f"\n[RESULT] {aig.name}: TIMEOUT after {TIMEOUT} sec\n")


        log.flush()
        print(f"Finished {aig.name} ({elapsed:.6f}s)")

# python3 run_ric3_from_aig.py ../abc_mc/AIG/Safety safety.log
# python3 run_ric3_from_aig.py ../abc_mc/AIG/Liveness liveness.log
# python3 run_ric3_from_aig.py ../abc_mc/AIG/Safety+Liveness safetyliveness.log