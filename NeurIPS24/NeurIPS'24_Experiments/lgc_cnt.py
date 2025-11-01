import subprocess
import os
import re
def run_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True, shell=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        #print(f"An error occurred: {e}")
        return None

pattern = r'^Latches:.*$'
directory_path = "../../Benchmarks/"
for filename in sorted(os.listdir(directory_path)):
    if filename.endswith('.sv'):
        file_path = os.path.join(directory_path, filename)
        command = f"../Tools/ebmc/ebmc_v5_x {file_path} --verbosity 8 --aig > lgc.txt"
        command = f"../Tools/ebmc/ebmc_v5_x {file_path} --verbosity 8 --aig > lgc.txt"        
        output = run_command(command)
        with open(f"lgc.txt", 'r') as file:
            output = file.read()
        if output is not None:
            matches = re.findall(pattern, output, re.MULTILINE)
            for match in matches:
                print(f"{filename}\n\t{match}")
