"""
Gurobi MIP Learning Engine
=============================

This module provides Mixed-Integer Programming interfaces using Gurobi to
train neural ranking and invariant functions by encoding network layers and
logical constraints as MIP.

Core Functions:
 1. `addLinParam(m, n_in, n_out, mat_name, vec_name, **kwargs)`
    • Add weight matrix `W` and bias vector `b` as Gurobi MVars to model `m`.

 2. `addNNParam(m, sizes, W_prfx, b_prfx, psfx, **kwargs)`
    • Construct a list of `(W,b)` parameter pairs for each NN layer,
      according to `sizes`.

 3. `addLinSignLayer(m, W, b, x, gap, ptype, psfx)`
    • Encode one binary sign-activation layer:

 4. `addLinSignNN(m, param, x, M, gap, ptype, psfx)`
    • Stack multiple sign layers to implement a full sign-activation NN.

 5. `addFunkyNN(m, nnparam, A, c, x, M, gap, ptype, psfx)`
    • Integrate a classical linear function `A x + c` with 
      neural sign-activation layers to produce a piecewise-linear output.
    • Returns the sum of final alpha variables representing the ranking value.

Data Structures:
  - `MIPLearn` class encapsulates a Gurobi model and learning workflow:
    • `__init__`: Initializes the Gurobi model, configures solver parameters,
      and allocates MVars for NN parameters, linear parameters, and `kappa`.
    • `getV(q, x)`: Create the MIP expression for the ranking
      function `V(q, x)` using `addFunkyNN`.
    • `addSamples(samples, kappa)`: Add transition constraints (samples of
      `(q, state, q_next, next_state)`) enforcing ranking conditions.
    • `addInitInvar(init_samples, kappa)`: Add initial-state invariants.
    • `guess()`: Solve the MIP, return extracted `(nnparam, linparam, kappa)`
      or `(None, None, None)` on infeasibility.

Training Loop Function:
  - `gurobiNNtrain(try_i, mipL, new_samples, new_inits, samples, init_samp,
    scale, P, accept, size, gap, kappa)`:
    • Orchestrates one iteration of MIP-based learning:
      1. Adds existing counterexample samples and initial invariants.
      2. Solves the MIP via `mipL.guess()`.
      3. Re-evaluate MIP solution with numpy, to check consistency
        

Quantization Helpers:
    [Quantisation, though implemented, is not required, as we learn integers;
     F will always be zero for current set of experimets]
  - `quant(a, F)`: Recursively quantize arrays/tuples of numpy arrays by
    scaling to integer fixed-point representation.
  - `optimal_F(samples, accept, nnparam, linparam, kappa, minimum, maximum)`:
    Search for the smallest fractional bit-width `F` that ensures all
    transitions satisfy the ranking constraints under quantization.

Evaluation Functions (Debug):
  - `evalFunkyNN(param, A, c, x)`: Evaluate the unquantized (floating-point)
    network and linear functions offline using numpy.
  - `evalQuantNN(param, A, c, x, scale, gap, F_prec, isDebug)`:
    Evaluate the quantized network and linear functions offline, simulating
    fixed-point arithmetic, and optionally print debug traces.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from colorama import init, Fore, Back, Style
M_enc = False

#=====================
# guess 
#=====================

def addLinParam(m, n_in, n_out, mat_name, vec_name, **kwargs):
    W = m.addMVar(shape=(n_out,n_in), name=mat_name, **kwargs)
    b = m.addMVar(shape=(n_out), name=vec_name, **kwargs)


    return (W,b)

def addNNParam(m, sizes, W_prfx="W", b_prfx="b", psfx="", **kwargs):
    return [addLinParam(m, n_in, n_out, W_prfx+str(i)+psfx, b_prfx+str(i)+psfx, **kwargs)
            for i,(n_in,n_out) in enumerate(zip(sizes[:-1],sizes[1:]))]


def addLinSignLayer(m, W, b, x, gap, ptype, psfx=''):
    m.update()
    assert(np.all(np.isfinite(W.lb)))
    assert(np.all(np.isfinite(W.ub)))
    assert(np.all(W.lb == -W.ub))
    assert (np.all(b.lb == -b.ub))

    P = W.ub
    C = m.addMVar(shape=W.shape, name="C"+psfx, vtype=ptype, lb=-P, ub=P)

    m.addConstr(C - W + 2*P*x.T <= 2*P)
    m.addConstr(C - W - 2*P*x.T >= - 2*P)

    m.addConstr(C + W - 2*P*x.T <= 0.)
    m.addConstr(C + W + 2*P*x.T >= 0.)

    y = m.addMVar(shape=np.size(W,0), vtype=GRB.BINARY, name="y"+psfx)

    M = sum(P.T)+b.ub #P*(np.size(W,1)+1)
    for C_row, b_row, y_row, M_row in zip(C, b, y, M):
        if M_enc:
            m.addConstr(sum(C_row) + b_row <= M_row*y_row - gap)
            m.addConstr(sum(C_row) + b_row >= M_row*(y_row - 1) + gap)
        else:
            m.addConstr((y_row == 0) >> (sum(C_row) + b_row <= - gap))
            m.addConstr((y_row == 1) >> (sum(C_row) + b_row >= + gap))

        
    return y

def addLinSignNN(m, param, x, M, gap, ptype, psfx=''):
    W_0, b_0 = param[0]
    y = m.addMVar(shape=np.size(W_0,0), vtype=GRB.BINARY, name="y0"+psfx)

    if M_enc:
        m.addConstr(W_0 @ x + b_0 <= M*y - gap)
        m.addConstr(W_0 @ x + b_0 >= M*(y - 1) + gap)
    else:
        m.addConstr((y == 0) >> (W_0 @ x + b_0 <= - gap))
        m.addConstr((y == 1) >> (W_0 @ x + b_0 >= + gap))

    for i, (W, b) in enumerate(param[1:], 1):
        y = addLinSignLayer(m, W, b, x=y, gap=gap, ptype=ptype, psfx=str(i)+psfx)

    return y
    
def addFunkyNN(m, nnparam, A, c, x, M, gap, ptype, psfx=''):
    m.update()
    alpha_ub = M*sum(A.ub.T)+c.ub
    alpha_lb = M*sum(A.lb.T)+c.lb
    alpha = m.addMVar(shape=(A.shape[0]), vtype=ptype, name='alpha'+psfx,
                      lb=alpha_lb, ub=alpha_ub) # just a guess - these bounds are to be checked
    if nnparam == None:
        m.addConstr((A @ x + c == alpha))
        return sum(alpha)
    
    y = addLinSignNN(m, nnparam, x, M, gap=gap, ptype=ptype, psfx=psfx)
    m.update()

    if M_enc:
        m.addConstr(alpha - (A @ x + c) <= 2*alpha_ub*(1 - y))
        m.addConstr(alpha - (A @ x + c) >= 2*alpha_lb*(1 - y))
        m.addConstr(alpha <= alpha_ub*y)
        m.addConstr(alpha >= alpha_lb*y)
    else:
        m.addConstr((y == 1) >> (A @ x + c == alpha))
        m.addConstr((y == 0) >> (0 == alpha))


    return sum(alpha)


class MIPLearn:

    def __init__(self, accept, size, P, M, ptype=GRB.INTEGER):
        self.m = gp.Model("mip1")

        # m.setParam("OutputFlag", 0)
        # m.setParam("LogToConsole", 0)
        self.m.setParam("FeasibilityTol", 1e-9)  
        #m.setParam("OptimalityTol", 1e-8) 
        self.m.setParam("IntFeasTol", 1e-9)      
        self.m.setParam("MIPGap", 1e-3)
        self.m.setParam('NumericFocus', 3)
        self.m.setParam('MIPFocus', 3)
        #m.setParam('Presolve', 2)
        #m.setParam('BarConvTol', 1e-8)  # Tighten barrier convergence tolerance
        #m.setParam('Crossover', 0)
        #m.setParam('Method', 1)
        self.gap = 1e-3
        
        self.M = M
        self.P = P
        self.ptype = ptype
        self.accept = accept
        if len(size) == 1:
            self.nnparam = [None for i in range(len(accept))]
        else:    
            self.nnparam = [addNNParam(self.m, size, vtype=ptype, lb=-P, ub=P, psfx='q'+str(i)) for i in range(len(accept))]
        
        if len(size) == 1:
            self.linparam = [addLinParam(self.m, size[0], 1, "A"+str(i), "c"+str(i), vtype=ptype, lb=-P, ub=P) for i in range(len(accept))]
        else: 
            self.linparam = [addLinParam(self.m, size[0], size[-1], "A"+str(i), "c"+str(i), vtype=ptype, lb=-P, ub=P) for i in range(len(accept))]

        self.kappa = self.m.addMVar(shape=(1), name="kappa", lb=-P, ub=P)
        
        self.cache = {}
        self.debug_cache = []
        self.m.setObjective(0., GRB.MINIMIZE)
            
    def getV(self,q,x):
        key = (q,x.tobytes())
        if key in self.cache:
            V,_ = self.cache[key]
            return V

        idx = str(len(self.cache))
        V = addFunkyNN(self.m, self.nnparam[q], *self.linparam[q], x, self.M, gap=self.gap, ptype=self.ptype, psfx="_s"+idx)
        self.cache[key] = (V, idx)
        return V

    def addSamples(self, samples, kappa):
        if len(samples) == 0:
            return
        assert (all([all(np.mod(s0, 1) == 0) and all(np.mod(s1, 1) == 0) for _,s0,_,s1 in samples]))
        
        M_out = 2*self.M*self.P*len(samples[0][1])
        for i, (q, s, q_next, s_next) in enumerate(samples):
            z = self.m.addVar(vtype=GRB.BINARY, name=f"z_{q}_{s}_{q_next}_{s_next}")
            V, V_next = self.getV(q,s), self.getV(q_next, s_next)
            if M_enc:
                self.m.addConstr((z - 1)*M_out <= V - (self.kappa + 1) )
                self.m.addConstr(V - (self.kappa + 1) <= z*M_out)
                self.m.addConstr(V_next - V + self.accept[q] <= z*(M_out + self.accept[q]))
            else:
                self.m.addConstr((z == 0) >> (V_next <= V - self.accept[q]))
                self.m.addConstr((z == 1) >> (V >= self.kappa + 1))
            #self.m.addConstr(V_next <= V - self.accept[q])
            self.debug_cache.append((q, s, q_next, s_next, V, V_next, z))
        self.m.update()
        assert(self.m.getAttr("IsMIP"))
        assert(not self.m.getAttr("IsQCP"))

    def addInitInvar(self, samples, kappa):
        if len(samples) == 0:
            return

        for i, (q, s) in enumerate(samples):
            V = self.getV(q,s)
            self.m.addConstr(V <= self.kappa)

        self.m.update()
        assert(self.m.getAttr("IsMIP"))
        assert(not self.m.getAttr("IsQCP"))

    def addSink(self, min_rank, sink):
        if sink == None:
            return
        assert (len(sink) > 0)

        for i, (q, s) in enumerate(sink):
            V = self.getV(q,s)
            self.m.addConstr(V <= min_rank)

        self.m.update()
        assert(self.m.getAttr("IsMIP"))
        assert(not self.m.getAttr("IsQCP"))

    def guess(self):
        #self.hard_code_sol()
        self.m.optimize()
        if self.m.Status == GRB.INFEASIBLE:
            return None, None, None
        if self.nnparam[0] == None:
            nnparam_X = self.nnparam
        else:
            nnparam_X = [[(W.X, b.X) for W, b in paramq] for paramq in self.nnparam]
        linparam_X = [(Aq.X, cq.X) for Aq,cq in self.linparam]
        kappa_X = self.kappa.X[0]
        return nnparam_X, linparam_X, kappa_X
        

#=====================
# Quantisation  
#=====================

def quant(a, F):
    if a is None:
        return None
    if isinstance(a, list):
        return [quant(item, F) for item in a]
    elif isinstance(a, tuple):
        return tuple(quant(item, F) for item in a)
    else:
        assert isinstance(a, np.ndarray)
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



def gurobiNNtrain(try_i, mipL, new_samples, new_inits, samples, init_samp, scale, P, accept, size, gap, kappa):
    #scale = .01
    #P = 20000.
    M = 1 + int(max(abs(element) for sample in samples for array in [sample[1], sample[3]] for element in array ))

    print(20*'=')
    print(' guess #', try_i)
    print(' Engine : Gurobi')
    print(20*'=')
    #nnparam, linparam = guess(samples, accept=[0,1], size=[1,2], P=P, M=M, scale=scale)
    if True:
        mipL.addSamples(new_samples, kappa)
        mipL.addInitInvar(new_inits, kappa)
        nnparam, linparam, kappa = mipL.guess()
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
