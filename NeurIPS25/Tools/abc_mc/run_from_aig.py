"""
Run simple_liveness.sh on each *.aig file in a directory,
log full stdout+stderr and time taken into liveness_full.log.
"""

import subprocess, time, sys
from pathlib import Path
import re

TOOL        = "../../super-prove-build/build/super_prove/bin/simple_liveness.sh"
TIMEOUT     = 20000  # seconds
LOGFILE     = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path.cwd()
TARGET_DIR  = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()

def extract_number(path):
    match = re.search(r'(\d+)(?=\.aig$)', path.name)
    return int(match.group(1)) if match else float('inf')

with open(str(LOGFILE), "w") as log:
    print( sorted(TARGET_DIR.glob("*.aig"), key=extract_number))
    for aig in sorted(TARGET_DIR.glob("*.aig"), key=extract_number):
        log.write("\n=== Running " + aig.name +" ===\n")
        print("Running "+ aig.name +"...")

        start = time.time()
        try:
            completed = subprocess.run([TOOL, str(aig)],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           universal_newlines=True,   
                           timeout=TIMEOUT)

            elapsed = time.time() - start
            log.write(completed.stdout)
            log.write("\n[RESULT] "+ aig.name +": OK in "+ str(elapsed) +" sec\n")
        except subprocess.TimeoutExpired as e:
            elapsed = TIMEOUT
            log.write(e.stdout or "")
            log.write("\n[RESULT] "+ aig.name +": TIMEOUT after "+ TIMEOUT +" sec\n")

        log.flush()
        print("Finished "+ aig.name +": "+ str(elapsed) +"sec\n")

# python3 run_from_aig.py ./AIG/Safety safety.log
# python3 run_from_aig.py ./AIG/Liveness liveness.log
# python3 run_from_aig.py ./AIG/Safety+Liveness safetyliveness.log