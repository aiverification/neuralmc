# Introduction

This artefact bundles **all experiments** for our NeurIPS’25 submission. The
*README* explains the directory layout, installation steps, experimental
setup, and how to reproduce every result in Docker.
All experiments are fully reproducible with the files provided here.

> **Licensing** – all third‑party dependencies keep their original licences; see
> the upstream projects referenced in this document.

---

# Directory structure

```text
.
├── README.md   ←    you are here
├── LICENSE
├── ebmc_5.6_amd64.deb
│
├── Benchmarks/
│   └── <design-name>_<index>.sv
│
├── Experiments/
│   ├── Liveness/
│   │   ├── <spec‑id>_<design>.py
│   │   └── l_runall.py
│   │
│   ├── Safety/
│   │   ├── <spec‑id>_<design>.py
│   │   └── s_runall.py
│   │
│   ├── Safety+Liveness/
│   │   ├── <spec‑id>_<design>.py
│   │   └── sl_runall.py
│   │
│   ├── plot_ablation.py
│   ├── plot_cmp.py
│   └── run_exp.py
│
└── Tools/
    ├── abc_mc/
    │   ├── AIG/
    │   ├── SMV/
    │   ├── abc_run.py
    │   ├── run_from_aig.py
    │   ├── ltl2smv
    │   └── smvtoaig
    │
    ├── nuXmv/
    │   ├── SMV/
    │   ├── nuxmv_run.py
    │   ├── nuXmv_IC3.sh
    │   └── nuXmv           
    │
    ├── ebmc/
    │   └── ebmc_run.py
    │
    └── neuralmc/
        ├── Model-SMT/
        ├── cav_nuR.py
        ├── check_for_gurobi.py
        ├── train_by_gurobi.py
        └── train_by_SMT.py
```

*(Auxiliary files not referenced in the paper are omitted.)*

---

# Content overview

### `Benchmarks/<design-name>_<index>.sv`

Contains all SystemVerilog designs under test (DUTs).  The *design‑name* matches
the identifier used in the paper, and *index* orders the parametrised instances
by increasing state‑space size.

### `Experiments/[✱]/[Spec-number]_[Design-name].py`

The experiment drivers are grouped in three folders:

    * `Experiments/Liveness`
    * `Experiments/Safety`
    * `Experiments/Safety+Liveness`

Each script targets one specification for one design.  The spec is appended at
runtime.  Batch execution helpers
(`l_runall.py`, `s_runall.py`, `sl_runall.py`) run an entire folder in one go.

### `Experiments/plot_cmp.py` & `Experiments/plot_ablation.py`

Regenerate every figure and table shown in the paper.  Timings are hard‑coded
to avoid the multi‑day cost of rerunning every benchmark.

### `Tools/abc_mc/`

Wrappers and helper scripts for the ABC model checker.  Requires:

* `ltl2smv`      [1] – converts LTL to SMV
* `smvtoaig`     [2] – converts SMV to AIG (ABC format)
* `run_from_aig`     - Script to run all experiments of a class already converted to AIG
* Note *SuperProve* tool suite [3] built on Ubuntu 16.04 must be present at `../../super-prove-build/build/super_prove/bin/simple_liveness.sh`

A complied binary isn't provided due to its size.

`libpython2.7` must be installed for the ABC binaries. 

### `Tools/nuXmv/`

nuXmv 2.1.0 binaries [4] plus the wrapper script `nuxmv_run.py` (supports both
BDD and IC3 modes).

### `Tools/ebmc/`

EBMC v5.6 binary [5, 6] to be installed following instructions in `Docker quick‑start`, a wrapper `ebmc_run.py` is provided here.

### `Tools/neuralmc/`

Our neural model‑checking implementation:

* `cav_nuR.py`  – orchestrates learning, synthesis, and Bitwuzla queries.
* `train_by_gurobi.py` / `train_by_SMT.py`    – MILP and SMT learners.
* `check_for_gurobi.py`    – verification/pass‑fail checking.
* `Model-SMT/`    – scratch directory for EBMC‑generated SMT2 files.

---

# References

1. [https://github.com/felipeblassioli/nusmv](https://github.com/felipeblassioli/nusmv)
2. [https://github.com/zimmski/aiger/](https://github.com/zimmski/aiger/)
3. [https://github.com/sterin/super-prove-build](https://github.com/sterin/super-prove-build)
4. [https://nuxmv.fbk.eu/download.html](https://nuxmv.fbk.eu/download.html)
5. [http://www.cprover.org/ebmc/](http://www.cprover.org/ebmc/)
6. [https://github.com/diffblue/hw-cbmc](https://github.com/diffblue/hw-cbmc)

---

# Operating‑system & hardware requirements

* All components (except ABC) build and run on modern Linux; we use Ubuntu 24.04.
* ABC **must** be built on Ubuntu  16.04 (Python 2 dependency). After building,
  you can copy the binaries to newer systems provided `libpython2.7` is present.
* Our neural network is trained on CPU; no GPU is required.

---

# Docker quick‑start

```bash
# host: clone repo next to /NeurIPS-25
[ec2-user@ip-x-x-x-x Artefact_NeurIPS_25]$ docker pull ubuntu:24.04
[ec2-user@ip-x-x-x-x Artefact_NeurIPS_25]$ docker run -it --rm -v "$PWD/NeurIPS-25:/NeurIPS-25" ubuntu:24.04
```

Inside the container:

```bash
apt update && apt upgrade -y
apt install -y python3 python3-pip git software-properties-common
```

Create a Python Environment:

```bash
apt install python3.12-venv
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
pip install colorama numpy matplotlib
``` 

---

# EBMC installation

```bash
cd NeurIPS-25/
dpkg -i ebmc_5.6_amd64.deb
```

## Bitwuzla

```bash
apt install pkg-config libgmp-dev
pip install cmake meson ninja cython

git clone https://github.com/bitwuzla/bitwuzla
cd bitwuzla
pip install .
cd ..
```

## Gurobi & other Learning Engines

Obtain a licence and install the Python binding:

```bash
pip install gurobipy
```

**(Ensure `gurobipy` can find a valid licence.)**


Other SMT back‑ends (z3, cvc5, msat) can be installed with PySMT:

```bash
pip install setuptools pysmt
pysmt-install --z3   # repeat for --cvc5 / --msat
pysmt-install --check
```
The last command should output:

```bash
Installed Solvers:
  msat      True (5.6.10)            
  cvc5      True (1.1.2)             
  cvc4      False (None)              Not in Python's path!
  z3        True (4.13.0)            
  yices     False (None)              Not in Python's path!
  btor      False (None)              Not in Python's path!
  picosat   False (None)              Not in Python's path!
  bdd       False (None)              Not in Python's path!

Solvers: z3, msat, cvc5
```
Make sure the versions match for experimental consistency.

If for `msat` you hit HTTP 500 errors see [7], which recommends using 

```bash
pip uninstall pysmt
pip install pysmt --pre
pysmt-install --msat 
```

7. https://github.com/pysmt/pysmt/issues/795

---

# Running experiments

## Safety experiments (fastest)

```bash
cd /NeurIPS-25/Experiments/Safety
python3 s_runall.py
```

Follow the prompts:

1. **Tool** →    choose `5` (our-auto) for neural model checking.
2. **Engine** →    choose `3` (z3) to avoid licence issues, to begin with.
3. **Random samples** →   enter `0`.
4. **Unbounded params** →    `y`.
5. **Files** →    `all` or list indices.

## rIC3
1. **Launch an Ubuntu 24.04 container**

```bash
docker pull ubuntu:24.04
docker run -it --rm --volume ./NeurIPS-25:/NeurIPS-25 ubuntu:24.04
 ```
1. **Install rIC3**

```bash
apt update
apt upgrade
apt install -y build-essential clang cmake pkg-config libssl-dev git curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup toolchain install nightly
cargo +nightly install rIC3
 ```
 
3. **Run rIC3 on each property class**
   These helpers read the AIGs and append runtimes to log files:

```bash
cd /NeurIPS-25/Tools/rIC3
python3 run_ric3_from_aig.py ../abc_mc/AIG/Safety          safety.log
python3 run_ric3_from_aig.py ../abc_mc/AIG/Liveness        liveness.log
python3 run_ric3_from_aig.py ../abc_mc/AIG/Safety+Liveness safetyliveness.log
```

## nuXmv

Run the same driver but pick option  `3` (BDD) or   `4` (IC3) at the first prompt.

## ABC

### Running ABC— Ubuntu 16.04 workflow

ABC must be built and executed on **Ubuntu 16.04** (due to its Python 2.x
dependency), whereas the rest of the artefact targets Ubuntu 24.04.
To save you from repeating the AIG-conversion step, we have already placed the
generated files in `Tools/abc_mc/AIG/[*]/`.

1. **Launch an Ubuntu 16.04 container**

   ```bash
   docker pull ubuntu:16.04
   docker run -it --rm --volume ./NeurIPS-25:/NeurIPS-25 ubuntu:16.04
   ```

2. **Inside the container**
   *Compile the Super Prove tool suite* as described in [3].
   The built script must be at, a complied binary isn't provided due to its size.
   `NeurIPS-25/super-prove-build/build/super_prove/bin/simple_liveness.sh`.

   ```bash
   apt update
   apt install -y build-essential libpython2.7  # runtime dep for Super Prove
   ```

3. **Run ABC on each property class**
   These helpers read the AIGs and append runtimes to log files:

   ```bash
   cd /NeurIPS-25/Tools/abc_mc
   python3 run_from_aig.py ./AIG/Safety          safety.log
   python3 run_from_aig.py ./AIG/Liveness        liveness.log
   python3 run_from_aig.py ./AIG/Safety+Liveness safetyliveness.log
   ```

4. **Verify ABC completed correctly**

   The log output is intentionally minimal.
   If a job appears to finish instantly, ABC may have aborted with an error;
   Run a single task and ensure you see the expected three-line footer:

   ```bash
    root@f5d2f9f7bc76:/NeurIPS-25/Tools/abc_mc# ../../super-prove-build/build/super_prove/bin/  simple_liveness.sh ./AIG/Safety/s1_delay_1.aig 
    WARNING! Stack size could not be changed.
    0
    j0
    .
    ```

---

# Generating plots & tables

```bash
cd /NeurIPS-25/Experiments
python3 plot_cmp.py       # comparison with SOTA
python3 plot_ablation.py  # ablation study
```

Both scripts print summary tables then render figures using the hard‑coded
runtime data from our experiments.


---
# Sample out-auto Execution

```shell
root@299d409623c1:/# cd NeurIPS-25/Experiments/Safety
root@299d409623c1:/NeurIPS-25/Experiments/Safety# python3 s_runall.py 
1) EBMC 2)ABC 3)nuXmv 4)nuXmv(IC3) 5)our(auto) 6)our(Linear)
Enter your choice: 5
Learning Engine: 1) gurobi 2) cvc5 3) z3 4) msat
Enter your choice: 3
No of Initial Random Samples: 0
Switch to Unbounded parameters? (y/n): y

Python files found:
   1) s1_delay.py
   2) s1_lcd.py
   3) s1_vga.py
   4) s2_delay.py
   5) s2_lcd.py
   6) s2_vga.py
   7) s_PWM.py
   8) s_blink.py
   9) s_gray.py
  10) s_i2c.py
  11) s_load_store.py
  12) s_seven_seg.py
  13) s_thermocouple.py
  14) s_uart.py

Select numbers separated by space / comma (or type 'all' to run everything): all
```

We show output for one of 206 tasks the script runs

```shell
                                delay_16 (X G !err) 400000

Making DELAY.sig a wire
Making DELAY.err a wire
Making DELAY.flg a wire
/NeurIPS-25/Experiments/Safety
Parsing ../../Benchmarks/delay_16.sv
Converting
Type-checking Verilog::DELAY
Making DELAY.sig a wire
Making DELAY.err a wire
Making DELAY.flg a wire
Generating Decision Problem
Writing SMT2 formula to `../../Tools/neuralmc/Model-SMT/delay_16.smt2'
Properties
sat
((|Verilog::DELAY.cnt@0| #b0000000000000000000))
((|Verilog::DELAY.cnt@1| #b0000000000000000001))
((|Verilog::DELAY.cnt@2| #b0000000000000000010))
((|Verilog::DELAY.cnt_aux0@0| #b0000000000000000001))
((|Verilog::DELAY.cnt_aux0@1| #b0000000000000000010))
((|Verilog::DELAY.err@0| false))
((|Verilog::DELAY.err@1| false))
((|Verilog::DELAY.flg@0| true))
((|Verilog::DELAY.flg@1| true))
((|Verilog::DELAY.rst@0| false))
((|Verilog::DELAY.rst@1| false))
((|Verilog::DELAY.sig@0| false))
((|Verilog::DELAY.sig@1| false))
Time Range Invar: 0.0005905628204345703
Range is an Invar [PASS]
====================
Initial Samples
====================
q = 0 to q = 1 is SAT (0, array([1]), 1, array([2]))
q = 1 to q = 1 is SAT (1, array([200640]), 1, array([200641]))
q = 1 to q = 2 is UNSAT
q = 2 to q = 2 is SAT (2, array([0]), 2, array([1]))
====================
Random Samples
====================
q = 0 to q = 0
q = 0 to q = 1
q = 0 to q = 2
q = 1 to q = 0
q = 1 to q = 1
q = 1 to q = 2
q = 2 to q = 0
q = 2 to q = 1
q = 2 to q = 2
====================
 guess # 0
 Engine :  z3
====================
[[A0_0_0]]
[[A1_0_0]]
[[A1_0_0]]
[[A1_0_0]]
[[A2_0_0]]
[[A2_0_0]]
[[A0_0_0]]
nnparam: [None, None, None];
linparam: [(array([[0.]]), array([0.])), (array([[0.]]), array([0.])), (array([[-1.]]), array([0.]))];
kappa: 0.0 
====================
 testing
====================
BEST F: 0
quant_kappa: 0
 V(q=0,[1]) = 0 -> V(q=1,[2]) = 0 : OK 
 V(q=1,[200640]) = 0 -> V(q=1,[200641]) = 0 : OK 
 V(q=2,[0]) = 0 -> V(q=2,[1]) = -1 : OK 
-------[Init Invar]-------
 V(q=0,[0.]) = 0 : OK
====================
 checking
====================
PRECISION: 0
q = 0 to q = 1 is UNSAT
q = 1 to q = 1 is UNSAT
q = 1 to q = 2 is UNSAT
q = 2 to q = 2 is SAT (2, array([0]), 2, array([0])) 
        Bitwuzla Rank [0.0 -> 0.0]; Numpy Rank [0.0 -> 0.0]; ; Numpy RankQ [0.0 -> 0.0]  
q = 0 [InVar] is UNSAT
====================
 guess # 1
 Engine :  z3
====================
nnparam: [None, None, None];
linparam: [(array([[0.]]), array([0.])), (array([[0.]]), array([0.])), (array([[0.]]), array([1.]))];
kappa: 0.0 
====================
 testing
====================
BEST F: 0
quant_kappa: 0
 V(q=0,[1]) = 0 -> V(q=1,[2]) = 0 : OK 
 V(q=1,[200640]) = 0 -> V(q=1,[200641]) = 0 : OK 
 V(q=2,[0]) = 1 -> V(q=2,[1]) = 1 : OK 
 V(q=2,[0]) = 1 -> V(q=2,[0]) = 1 : OK 
-------[Init Invar]-------
 V(q=0,[0.]) = 0 : OK
====================
 checking
====================
PRECISION: 0
q = 0 to q = 1 is UNSAT
q = 1 to q = 1 is UNSAT
q = 1 to q = 2 is UNSAT
q = 2 to q = 2 is UNSAT
q = 0 [InVar] is UNSAT
        *********  Yay! We've got a ranking function.  ************
BITS ---------->>>>>>>>> 50 delay_16 (X G !err) 400000 E: z3 P: None RndSmps: 0 isAuto: True Arch: [1]
Learn Time: 0.022541522979736328; Check Time: 0.003387451171875; Guess cnt: 1
Total Time: 0.03415274620056152
```



---
# Acknowledgements

ABC, EBMC, nuXmv, Bitwuzla, z3, cvc5, MathSAT and Gurobi are third‑party
projects; please see their respective licences.
