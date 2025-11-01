print("1) EBMC 2)ABC 3)nuXmv 4)nuXmv(IC3) 5)our(auto) 6)our(Linear)")
ch1 = input("Enter your choice: ")
if ch1 in ['5', '6']:    
    print(f"Learning Engine: 1) gurobi 2) cvc5 3) z3 4) msat")
    ch2 = input("Enter your choice: ")
    ch3 = input("No of Initial Random Samples: ")
    choices = [ch1, ch2, ch3]
    if ch2 != '1':
        ch = input("Switch to Unbounded parameters? (y/n): ")
        choices.append(ch)
else:
    choices = [ch1]

ch_stdin = "\n".join(choices) + "\n"

from pathlib import Path
import subprocess
import sys

def menu(files):
    """Return the list of Path objects the user wants to run."""
    print("\nPython files found:")
    for i, f in enumerate(files, 1):
        print(f"  {i:2d}) {f.name}")

    choice = input(
        "\nSelect numbers separated by space / comma "
        "(or type 'all' to run everything): "
    ).strip().lower()

    if choice == "all":
        return files

    try:
        idx = {int(x) for x in choice.replace(",", " ").split()}
        chosen = [files[i - 1] for i in sorted(idx) if 1 <= i <= len(files)]
    except ValueError:
        sys.exit("Invalid selection – only numbers or 'all' are allowed.")

    if not chosen:
        sys.exit("Nothing selected – exiting.")
    return chosen


T = 900*6              # ⟵ max seconds you’ll allow 

def main():
    this_file = Path(__file__).resolve()
    py_files   = sorted(p for p in Path(".").glob("*.py") if p.resolve() != this_file)

    if not py_files:
        sys.exit("No other .py files found in this directory.")

    chosen_files = menu(py_files)
    print(chosen_files)

    for script in chosen_files:
        print(f"\n=== Running {script.name} (timeout {T}s) ===")
        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                input=ch_stdin,
                text=True,
                timeout=T,                # ⟵ the only required extra line
            )
            if completed.returncode:
                print(f"{script.name} exited with code {completed.returncode}")
        except subprocess.TimeoutExpired:
            print(f"{script.name} timed out after {T} s and was terminated.")

if __name__ == "__main__":
    main()
