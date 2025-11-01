"""
PySMT-Based Neural Model Checking Learning Engine
==========================================

This module mirrors the MILP and Bitwuzla pipelines using an SMT-based
encoding with PySMT. It constructs symbolic constraints for neural ranking
functions and transition invariants, and solves them with an SMT solver.

Global Imports:
  - PySMT: Symbol, And, Or, Plus, Times, LE, GE, GT, Equals, Implies,
    Ite, Solver, Real, Int, is_sat; types REAL, INT.
  - numpy: Numeric arrays and operations.
  - colorama: Console color output.

Helper Functions:
 1. to_python_value(val_expr)
    • Convert a PySMT constant expression to a native Python float or int.

Linear & Neural Parameter Construction:
 2. add_lin_param(n_in, n_out, mat_name, vec_name, vtype="int", P=None)
    • Create weight matrix W and bias vector b as PySMT Symbols.
    • Optionally constrain each coefficient within [-P, P].

 3. add_nn_param(sizes, W_prefix, b_prefix, psfx, vtype, P)
    • Build a list of (W,b) layer parameters and collect bound constraints.

Sign-Activation Encoding:
 4. add_lin_sign_layer(W, b, x, gap, ptype, psfx, P)
    • Encode one binary sign-activation layer:

 5. add_lin_sign_nn(param, x, gap, ptype, psfx, P)
    • Stack multiple sign layers to produce a binary vector y from input x.

Sign Neural + Linear Combination:
 6. add_funky_nn(nnparam, A, c, x, M, gap, ptype, psfx, P)
    • Combine a sign-activation network with a linear layer
      A*x + c, producing symbolic alpha variables.
    • Enforce: if y[i]==1 then A*x+c == alpha[i], else alpha[i]==0.
    • Return V = sum(alpha), the ranking expression.

SMT Model Class:
 7. class SMTLearn:
    - __init__(accept, size, P, M, ptype)
      • Initialize parameters per control state: nnparam layers and linparam (A,c).
      • Create symbolic kappa with optional bounds.
      • Store all constraints in `self.constraints`.

    - get_V(q, x)
      • Build V(q,x) expression via `add_funky_nn`.

    - add_samples(samples)
      • For each (q, s, q_next, s_next), add binary z to implement or:
        z==0 → V_next <= V - accept[q]
        z==1 → V >= kappa + 1
      • output constraints.

    - add_init_invar(init_samples)
      • Enforce V(q,s) <= kappa for initial states.

    - check_smt(engine_name)
      • Combine constraints into one formula and invoke the SMT `engine_name`.
      • Return model if SAT, else None.

    - get_solution(model)
      • Extract numeric arrays for nnparam, linparam, and kappa from the model.

    - guess(engine)
      • Run `check_smt` and return the learned parameters or (None,None,None).

Quantization & Evaluation:
[Quantisation, though implemented, is not required, as we learn integers;
 F will always be zero for current set of experimets]
 8. quant(a, F)
    • Recursively scale numpy arrays or lists/tuples to fixed-point ints with F bits.

 9. optimal_F(samples, accept, nnparam, linparam, kappa, minimum, maximum)
    • Find the smallest F such that all sample transitions satisfy ranking under quantization.

10. evalFunkyNN(param, A, c, x)
    • Offline evaluate the (unquantized) ranking: sign-network + linear output.

11. evalQuantNN(param, A, c, x, scale, gap, F_prec, isDebug)
    • Offline simulate quantized fixed-point evaluation and return numeric ranking.

12. smtNNtrain(try_i, smtL, new_samples, new_inits, samples, init_samp,
     scale, P, accept, size, gap, kappa, engine)
    • One training iteration:
      - Add new counterexamples and initial invariants.
      - Call `smtL.guess(engine)` to solve for parameters.
      - Determine optimal F via `optimal_F`, then report and test sample outcomes.
      - Return learned (nnparam, linparam, kappa, best_F) or (None,None,None,None).

"""

from pysmt.shortcuts import Symbol, And, Or, Plus, Times, LE, GE, GT, Equals, Implies, Solver, Real, Int, Ite, Minus, is_sat
from pysmt.typing import REAL, INT
import numpy as np
from colorama import init, Fore, Back, Style

#############################
# Helper Functions
#############################
def to_python_value(val_expr):
    cval = val_expr.constant_value()
    return float(cval) if hasattr(cval, 'numerator') else cval

def add_lin_param(n_in, n_out, mat_name, vec_name, vtype="int", P=None):
    typ = INT if vtype=="int" else REAL
    W = [[Symbol(f"{mat_name}_{i}_{j}", typ) for j in range(n_in)] for i in range(n_out)]
    b = [Symbol(f"{vec_name}_{i}", typ) for i in range(n_out)]
    constraints = []
    if P != None:
        for i in range(n_out):
            for j in range(n_in):
                constraints.append(And(GE(W[i][j], Int(-P)), LE(W[i][j], Int(P))))
            constraints.append(And(GE(b[i], Int(-P)), LE(b[i], Int(P))))
    return W, b, constraints

def add_nn_param(sizes, W_prefix="W", b_prefix="b", psfx="", vtype="real", P=None):
    nn_layers = []
    nn_constraints = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        W, b, cons = add_lin_param(n_in, n_out, f"{W_prefix}{i}{psfx}", f"{b_prefix}{i}{psfx}", vtype=vtype, P=P)
        nn_layers.append((W,b))
        nn_constraints.extend(cons)
    return nn_layers, nn_constraints


def add_lin_sign_layer(W, b, x, gap, ptype, psfx='', P=None):
    #Note: here sign(x) gives 1 if x is positve and 0 otherwise; for W*sign(x) in numpy we give -W if sign(x) is 0 and W if sign(x) is 1.
    constraints = []
    n_out = len(W)
    n_in = len(W[0])
    
    # Create an auxiliary matrix C of the same shape as W (with bounds -P <= C_ij <= P)
    C = [[Symbol(f"C{psfx}_{i}_{j}", REAL if ptype=="real" else INT) for j in range(n_in)] for i in range(n_out)]
    #breakpoint()
    
    for i in range(n_out):
        for j in range(n_in):
            if P != None:
                constraints.append(And(GE(C[i][j], -P), LE(C[i][j], P)))
            constraints.append(Implies(Equals(x[j], Int(0)), Equals(C[i][j],Minus(Int(0), W[i][j]))))
            constraints.append(Implies(Equals(x[j], Int(1)), Equals(C[i][j], W[i][j])))

    
    # Create binary output variable y for the layer (one per row)
    y = [Symbol(f"y{psfx}_{i}", INT) for i in range(n_out)]
    for i in range(n_out):
        constraints.append(And(GE(y[i], Int(0)), LE(y[i], Int(1))))
        # For each row, compute the sum over the row of C and add b[i]
        row_sum = Plus(C[i])
        constraints.append(Implies(Equals(y[i], Int(0)), LE(Plus(row_sum, b[i]), Int(0))))
        constraints.append(Implies(Equals(y[i], Int(1)), GT(Plus(row_sum, b[i]), Int(0))))
    
    return y, constraints


def add_lin_sign_nn(param, x, gap, ptype, psfx='', P=None):
    constraints = []
    # First layer:
    W0, b0 = param[0]
    n_out = len(W0)
    y = [Symbol(f"y0{psfx}_{i}", INT) for i in range(n_out)]
    for i in range(n_out):
        constraints.append(And(GE(y[i], Int(0)), LE(y[i], Int(1))))
        # dot product W0[i] . x
        dot_expr = Plus([Times(W0[i][j], Int(int(x[j]))) for j in range(len(W0[i]))])
        constraints.append(Implies(Equals(y[i], Int(0)), LE(Plus(dot_expr, b0[i]), Int(0))))
        constraints.append(Implies(Equals(y[i], Int(1)), GT(Plus(dot_expr, b0[i]), Int(0))))
    
    # Subsequent layers: use the sign layer transformation
    for i, (W, b) in enumerate(param[1:], 1):
        y, cons_layer = add_lin_sign_layer(W, b, x=y, gap=gap, ptype=ptype, psfx=str(i)+psfx, P=P)
        constraints.extend(cons_layer)
    
    return y, constraints


def add_funky_nn(nnparam, A, c, x, M, gap, ptype, psfx='', P=None):
    """
    This function “combines” a sign neural network with additional linear constraints.
    First it obtains a binary output y by processing x through the neural network.
    Then it creates a vector alpha and adds constraints:
       - (y == 1) implies A*x + c equals alpha
       - (y == 0) implies alpha == 0.
    Returns:
       V: a symbolic value equal to sum(alpha) (used as the network’s “ranking” value),
       cons: the list of constraints generated,
       alpha: the list of newly introduced symbols.
    """
    print(A)
    constraints = []
    n_out = len(A)
    alpha = [Symbol(f"alpha{psfx}_{i}", REAL if ptype=="real" else INT) for i in range(n_out)]
    for i in range(n_out):
        if P is not None:
            constraints.append(And(GE(alpha[i], Int(-P)), LE(alpha[i], Int(P))))
    
    if nnparam == None:
        for i in range(n_out):
            dot_expr = Plus([Times(A[i][j], Int(int(x[j]))) for j in range(len(A[i]))])
            constraints.append(Equals(Plus(dot_expr, c[i]), alpha[i]))
        V = Plus(alpha)
        return V, constraints, alpha

    y, cons_nn = add_lin_sign_nn(nnparam, x, gap, ptype, psfx, P)
    constraints.extend(cons_nn)
    
    
    for i in range(n_out):
        # A*x (dot product per row)
        dot_expr = Plus([Times(A[i][j], Int(x[j])) for j in range(len(A[i]))])
        constraints.append(Implies(Equals(y[i], Int(1)), Equals(Plus(dot_expr, c[i]), alpha[i])))
        constraints.append(Implies(Equals(y[i], Int(0)), Equals(alpha[i], Int(0))))
    V = Plus(alpha)
    return V, constraints, alpha

#############################
# SMT Model Class
#############################

class SMTLearn:
    """
    This class mirrors your MILP‐based MIPLearn but builds an SMT formula.
    Instead of “optimizing,” we collect constraints and then use a specified SMT solver engine 
    to check satisfiability. (In a typical SMT workflow you would then extract a satisfying model.)
    """
    def __init__(self, accept, size, P, M, ptype="int"):

        # size: list of layer sizes for the neural net
        self.P = P
        self.M = M  
        
        self.ptype = ptype
        self.accept = accept
        self.gap = 1e-3  # as used in your constraints

        # Collect all constraints in a list
        self.constraints = []

        # Create network parameters (one per mode q)
        self.nnparam = []   # list of nn parameter layers per q
        self.linparam = []  # list of (A,c) pairs per q
        for i in range(len(accept)):
            # Each network parameter is a set of layers
            if(len(size) != 1):
                nn_layers, cons_nn = add_nn_param(size, W_prefix=f"W", b_prefix=f"b", psfx=f"_q{i}", vtype=ptype, P=P)
                self.constraints.extend(cons_nn)
                self.nnparam.append(nn_layers)
            else:
                self.nnparam.append(None)
            
            # Linear parameters for the “Funky” part
            if(len(size) != 1):
                A, c, cons_lin = add_lin_param(n_in=size[0], n_out=size[-1], mat_name=f"A{i}", vec_name=f"c{i}", vtype=ptype, P=P)
            else:
                A, c, cons_lin = add_lin_param(n_in=size[0], n_out=1, mat_name=f"A{i}", vec_name=f"c{i}", vtype=ptype, P=P)
            self.constraints.extend(cons_lin)
            self.linparam.append((A, c))
        
        # Add a kappa variable
        typ = REAL if ptype=="real" else INT
        self.kappa = Symbol("kappa", typ)
        if P is not None:
            self.constraints.append(And(GE(self.kappa, Int(-P)), LE(self.kappa, Int(P))))
        
        self.cache = {}       # To avoid rebuilding parts of the network
        self.debug_cache = [] # For logging which constraints are associated with which sample

    def get_V(self, q, x):
        """
        For a given mode q and input x (assumed to be a list of numbers or PySMT constants),
        build the constraints corresponding to the network output (via add_funky_nn).
        Uses caching to avoid redundant constraints.
        Returns a PySMT term representing the “ranking” value V.
        """
        key = (q, tuple(x))
        if key in self.cache:
            V, _ = self.cache[key]
            return V
        
        idx = str(len(self.cache))
        # In this encoding the FUnky network uses the nn parameters and linear parameters.
        # Note: here x is assumed to be given as a list of PySMT terms or numerals.
        A, c = self.linparam[q]
        V, cons, _ = add_funky_nn(self.nnparam[q], A, c, x, self.M, gap=self.gap, ptype=self.ptype, psfx="_s"+idx, P=self.P)
        self.constraints.extend(cons)
        self.cache[key] = (V, idx)
        return V

    def add_samples(self, samples):
        """
        samples: list of tuples (q, s, q_next, s_next)
        For each sample, we add a binary variable z (with z ∈ {0,1}) and add constraints 
        that couple the “ranking” values V and V_next.
        """
        for (q, s, q_next, s_next) in samples:
            # Create a binary variable z for the sample
            typ = INT if self.ptype=="int" else REAL
            z = Symbol(f"z_{q}_{s}_{q_next}_{s_next}", typ)
            self.constraints.append(And(GE(z, Int(0)), LE(z, Int(1))))  # enforce binary
            
            V = self.get_V(q, s)
            V_next = self.get_V(q_next, s_next)
            # (z == 0) implies (V_next <= V - accept[q])
            self.constraints.append(Implies(Equals(z, Int(0)), LE(V_next, Plus(V, Int(-self.accept[q])))))
            # (z == 1) implies (V >= kappa + 1)
            self.constraints.append(Implies(Equals(z, Int(1)), GE(V, Plus(self.kappa, Int(1)))))
            self.debug_cache.append((q, s, q_next, s_next, V, V_next, z))
    
    def add_init_invar(self, samples):
        """
        For each initial sample (q,s) add V <= kappa.
        """
        for (q, s) in samples:
            V = self.get_V(q, s)
            self.constraints.append(LE(V, self.kappa))
    
    def add_sink(self, min_rank, sink):
        """
        For each sink sample (q,s) add V <= min_rank.
        """
        if sink is None: 
            return
        for (q, s) in sink:
            V = self.get_V(q, s)
            self.constraints.append(LE(V, min_rank))
    
    def check_smt(self, engine_name="z3"):
        """
        Combine all constraints into a single formula and use the given SMT engine (e.g. 'z3' or 'cvc4')
        to check satisfiability. If the formula is SAT, returns the model, otherwise None.
        """
        formula = And(self.constraints)
        with Solver(name=engine_name) as solver:
            solver.add_assertion(formula)
            if solver.solve():
                return solver.get_model()
            else:
                return None
    
    def get_solution(self, model):
        nnparam_sol = []
        if self.nnparam[0] == None:
            nnparam_sol = self.nnparam
        else:
            for network_layers in self.nnparam:
                wb_list = []
                for (W_matrix, b_vector) in network_layers:
                    W_vals = []
                    for row in W_matrix:
                        row_vals = [to_python_value(model.get_value(sym)) for sym in row]
                        W_vals.append(row_vals)
                    b_vals = [to_python_value(model.get_value(sym)) for sym in b_vector]
                    wb_list.append((np.array(W_vals), np.array(b_vals)))
                nnparam_sol.append(wb_list)

        linparam_sol = []
        for (A_matrix, c_vector) in self.linparam:
            A_vals = []
            for row in A_matrix:
                row_vals = [to_python_value(model.get_value(sym)) for sym in row]
                A_vals.append(row_vals)
            c_vals = [to_python_value(model.get_value(sym)) for sym in c_vector]
            linparam_sol.append((np.array(A_vals), np.array(c_vals)))

        kappa_val = to_python_value(model.get_value(self.kappa))

        return nnparam_sol, linparam_sol, kappa_val

    def guess(self, engine):
        #self.hard_code_sol()
        model = self.check_smt(engine)
        if model is None:
            print("Model infeasible")
            return None, None, None
        return self.get_solution(model)
        


#=====================
# Quantisation  
#=====================

def quant(a, F):
    if a is None:
        return None
    elif isinstance(a, list):
        return [quant(item, F) for item in a]
    elif isinstance(a, tuple):
        return tuple(quant(item, F) for item in a)
    elif isinstance(a, float):
        return (a*np.power(2,F)).astype(int)
    else:
        if not isinstance(a, np.ndarray):
            breakpoint()
        return (a*np.power(2,F)).astype(int)
    
def optimal_F(samples, accept, nnparam, linparam, kappa, minimum=0, maximum=32):
    #this assumes no scaling of the input
    for (q, s, q_next, s_next) in samples:
        V = evalFunkyNN(nnparam[q], *linparam[q], s.astype(int))
        V_next = evalFunkyNN(nnparam[q_next], *linparam[q_next], s_next.astype(int))
        #print(f"{Fore.RED} V(q={q},{s}) = {V} -> V(q={q_next},{s_next}) = {V_next} : {Style.RESET_ALL}")
        if  (V < kappa) and (V < V_next + accept[q]):
            print(f"{Fore.RED} V(q={q},{s}) = {V} -> V(q={q_next},{s_next}) = {V_next} : {Style.RESET_ALL}")
            breakpoint()

    for F in range(minimum, maximum):
        qnnparam, qlinparam = quant(nnparam, F), quant(linparam, F)
        quant_kappa = int(2**F*kappa)
        if_success = True
        for (q, s, q_next, s_next) in samples:
            V = evalFunkyNN(qnnparam[q], *qlinparam[q], s.astype(int))
            V_next = evalFunkyNN(qnnparam[q_next], *qlinparam[q_next], s_next.astype(int))
            #print(f"{Fore.RED} [{F}] V(q={q},{s}) = {V} -> V(q={q_next},{s_next}) = {V_next} : {Style.RESET_ALL}")
            assert V.dtype == int
            assert V_next.dtype == int
            if (V < quant_kappa) and (V < V_next + accept[q]):
                if_success = False
                break
        if if_success:
            return F

    return None

def evalFunkyNN(param, A, c, x):
    if param == None:
        return np.sum(A @ x + c)
    y = x    
    for W,b in param:
        y = np.sign(W @ y + b) 
        
    return np.dot(A @ x + c, np.maximum(y, 0))

def evalQuantNN(param, A, c, x, scale, gap, F_prec, isDebug = False):
    x_quant = (x * (2**F_prec)).astype(int)
    scale_q = int(scale * (2**F_prec))
    x_quant_sc = (x_quant * scale_q) >> F_prec
    y_quant = x_quant_sc

    if isDebug:
        breakpoint()
    
    A_quant = (A * (2**F_prec)).astype(int)
    c_quant = (c * (2**F_prec)).astype(int)
    
    if param == None:
        z1_quant = ((A_quant @ x_quant_sc) >> F_prec) + c_quant
        return np.sum(z1_quant)
        
    for W,b in param:
        W_quant = (W * (2**F_prec)).astype(int)
        b_quant = (b * (2**F_prec)).astype(int)
        z_quant = ((W_quant @ y_quant) >> F_prec) + b_quant
        y = np.sign(z_quant) #np.where(np.abs(z) <= gap, 0, np.sign(z))
        y_quant = (y * (2**F_prec)).astype(int)

    z1_quant = ((A_quant @ x_quant_sc) >> F_prec) + c_quant

    if isDebug:
        print(f"x :{x}; x_quant: {x_quant}")
        print(f"scale :{scale}; scale_q :{scale_q}")
        print(f"x_quant_sc :{x_quant_sc}")
        print(f"W :{W}; W_quant: {W_quant}")
        print(f"W :{b}; b_quant: {b_quant}")
        print(f"z :{z_quant/2**F_prec}; z_quant: {z_quant}")
        print(f"y :{y}; y_quant: {y_quant}\n\n")

        print(f"A :{A}; A_quant: {A_quant}")
        print(f"c :{c}; c_quant: {c_quant}")
        print(f"z :{z1_quant/2**F_prec}; z_quant: {z1_quant}")
        breakpoint()

    return np.dot(z1_quant, np.maximum(y, 0.))

def smtNNtrain(try_i, smtL, new_samples, new_inits, samples, init_samp, scale, P, accept, size, gap, kappa, engine):
    #scale = .01
    #P = 20000.
    M = 1 + int(max(abs(element) for sample in samples for array in [sample[1], sample[3]] for element in array ))

    print(20*'=')
    print(' guess #', try_i)
    print(' Engine : ', engine)
    print(20*'=')
    #nnparam, linparam = guess(samples, accept=[0,1], size=[1,2], P=P, M=M, scale=scale)
    if True:
        smtL.add_samples(new_samples)
        smtL.add_init_invar(new_inits)
        nnparam, linparam, kappa = smtL.guess(engine)
        if nnparam == None and linparam == None and kappa == None:
            return None, None, None, None
        print(f"nnparam: {nnparam};\nlinparam: {linparam};\nkappa: {kappa} ")
    
    print(20*'=')
    print(' testing')
    print(20*'=')
    never_failed = True
    best_F = optimal_F(samples, accept, nnparam, linparam, kappa)
    print(f"BEST F: {best_F}")
    quant_kappa = int(2**best_F*kappa)
    print(f"quant_kappa: {quant_kappa}")
    for (q, s, q_next, s_next) in samples:# + [(0, np.array([float(800)]), 1, np.array([float(0)])), (1, np.array([float(800)]), 1, np.array([float(0)]))]:
        V = evalFunkyNN(quant(nnparam[q], best_F), *(quant(linparam[q], best_F)), s.astype(int))
        V_next = evalFunkyNN(quant(nnparam[q_next], best_F), *(quant(linparam[q_next], best_F)), s_next.astype(int))
        
        if V > quant_kappa:
            res = "OK"
        elif accept[q] == 0:
            res = "OK" if V >= V_next else "FAIL"
        else:
            res = "OK" if V - 1 >= V_next else "FAIL"
    
        if res == "FAIL":
            never_failed = False
        print(f"{Fore.YELLOW} V(q={q},{s}) = {V} -> V(q={q_next},{s_next}) = {V_next} : {res} {Style.RESET_ALL}")

        #V_q = evalQuantNN(nnparam[q], *linparam[q], s, scale, gap, F_prec)
        #V_next_q = evalQuantNN(nnparam[q_next], *linparam[q_next], s_next, scale, gap, F_prec)
        #if q == 0:
        #    res = "OK" if V_q >= V_next_q else "FAIL"
        #else:
        #    res = "OK" if V_q - scale*0.9**2*F_prec >= V_next_q else "FAIL"
        #if res == "FAIL":
        #    never_failed = False
        #print(f"{Fore.MAGENTA} V(q={q},{s}) = {V_q/2**F_prec} -> V(q={q_next},{s_next}) = {V_next_q/2**F_prec} : {res} {Style.RESET_ALL}")
    
    print("-------[Init Invar]-------")
    for (q, s) in init_samp:
        V = evalFunkyNN(quant(nnparam[q], best_F), *(quant(linparam[q], best_F)), s.astype(int))
        res = "OK" if V <= quant_kappa else "FAIL"
        if res == "FAIL":
            never_failed = False
        print(f"{Fore.YELLOW} V(q={q},{s}) = {V} : {res}{Style.RESET_ALL}")

    if not never_failed:
        print(f"{Fore.RED} [Exiting] Failed at a sample; Tune hyperparameter and retry! {Style.RESET_ALL}")
        breakpoint()
    print(20*'=')
    print(' checking')
    print(20*'=')
    return nnparam, linparam, kappa, best_F
    '''
    cex = check(nnparam, linparam, scale, N)

    if len(cex) == 0:
        print('Yay! We\'ve got a ranking function')
        break

    assert not any(c in samples for c in cex)
    samples += cex
    assert (print(20*'='+'\n always remember to use the -O flag when measuring time\n'+20*'=') or True)
    '''
