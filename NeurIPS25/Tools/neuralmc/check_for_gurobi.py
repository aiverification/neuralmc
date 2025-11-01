"""
Bitwuzla Neural Arithmetic Utilities & Check Phase
=================================================

This module implements efficient bit-vector arithmetic primitives and neural
network quantization routines using Bitwuzla as the SMT backend. 
(note: quantization is switched of for current version as we use integer)

Global Constants:
  - colours: ANSI color palette for terminal logging (Fore.*).

Core Functions:
 1. bMul(bvar, number, bw_obj, bvsizeB_larger, dpt)
    • Perform constant multiplication via recursive shift-and-add decomposition.
    • Handles positive/negative factors by splitting into power-of-two and bias.

 2. bDotnew(arrVar, arrNum, bw_obj, F_prec, bits)
    • Balanced recursive summation of term-by-term multiplications.
    • For single-element lists, extends and shifts for fixed-point precision.

 3. bDotPositive(arrVar, arrNum, bw_obj, F_prec, bits)
    • Separates positive and negative coefficients, computes two dot-products,
      and subtracts negative sum from positive sum.

 4. bElemMul(bw_obj, mat1, scaled_weight, F_prec, bits, isDebug=False)
    • Element-wise multiplication of a BV array and integer weights,
      applying sign-extend, multiply, arithmetic shift, and extract.

 5. bSum(arrVar, bw_obj, F_prec, bits)
    • Balanced tree addition of BV terms for efficient summing.

 6. bSignFnc(bw_obj, arrVar, bits, isLast, F_prec, gap)
    • Compute discrete sign values (+1, 0, -1) for each term based on MSB
      and a gap threshold using ITE constructs.

Miscellaneous Helpers:
  - bShiftInc(arr, F_prec, bw_obj): Left-shift array elements by F_prec bits.
  - bMat(mat, bw_obj) / bVec(vec, bw_obj): Convert Python matrices/vectors to
    BV constant arrays.
  - bAnd(arr, bw_obj): Balanced conjunction over a list of terms.
  - bv2int(arr, bw_obj, bits) / todecimal(...): Decode BV terms to Python ints.
  - bPrint(arr, bw_obj): Print symbols and values of BV terms.
  - BLessThan / BLessThanEq / BLessThanEps: Comparison predicates over BV ranks.
  - bSetListUnEqual: Assert inequality between lists of BV values.

Neural Ranking Routines:
  - bLinear(bw_obj, W, b, inp, F_prec, bits, isDebug=False)
    • One-layer quantized neural inference: W*x + b.
  - bSignNN(bw_obj, param, x, F_prec, bits, gap, isDebug)
    • Multi-layer sign activation network producing binary masks.
  - bCAV_NRF(bw_obj, nnparam, clparam, inp, scale, F_prec, bits, gap, isDebug)
    • Combine quantized linear and sign networks to compute a ranking function.

Transition Checking:
  - check_tran(...)
    • Assert transition relation, ranking constraints, and collect SAT counterexamples.
  - check(...)
    • Iterate over all state-pair transitions to gather violations.
  - check_init(...)
    • Verify initial-state ranking precondition to gather violations.

"""

import bitwuzla as bw
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")
from itertools import product
import Tools.neuralmc.cav_nuR as nuR
import Tools.neuralmc.train_by_gurobi as nnGurobi

from colorama import init, Fore, Back, Style

colours = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTCYAN_EX]



"""
                                   ----------------------
                                    Bitwuzla Dot Product
                                   ----------------------

"""

def bMul(bvar, number, bw_obj, bvsizeB_larger, dpt):
    tm, opt, parser, bvsizeB = bw_obj
    if (number == 0):
        return tm.mk_term(bw.Kind.BV_MUL, [bvar, tm.mk_bv_value(bvsizeB_larger, 0)])
    
    sign = 1 if number > 0 else -1
    pow_2 = math.floor(math.log2(abs(number)))
    bias = abs(number) - 2**math.floor(math.log2(abs(number)))
    t1 = tm.mk_term(bw.Kind.BV_SHL, [bvar, tm.mk_bv_value(bvsizeB_larger, pow_2)])#tm.mk_term(bw.Kind.BV_MUL, [bvar, tm.mk_bv_value(bvsizeB, 2**pow_2)])#tm.mk_term(bw.Kind.BV_SHL, [bvar, tm.mk_bv_value(bvsizeB, pow_2)])
    if (bias > 2):
        t2 = bMul(bvar, bias, bw_obj, bvsizeB_larger, dpt +1) #tm.mk_term(bw.Kind.BV_MUL, [bvar, tm.mk_bv_value(bvsizeB, bias)])
    else:
        t2 = tm.mk_term(bw.Kind.BV_MUL, [bvar, tm.mk_bv_value(bvsizeB_larger, bias)])
    if sign == 1:
        return tm.mk_term(bw.Kind.BV_ADD, [t1, t2])
    else:
        return tm.mk_term(bw.Kind.BV_MUL, [tm.mk_term(bw.Kind.BV_ADD, [t1, t2]), tm.mk_bv_value(bvsizeB_larger, -1)])

def bDotPositive(arrVar, arrNum, bw_obj, F_prec, bits):
    tm, opt, parser, bvsizeB = bw_obj
    posVar, posNum, negVar, negNum = [], [], [], []
    for i in range(0, len(arrNum)):
        if arrNum[i] > 0:
            posVar.append(arrVar[i])
            posNum.append(arrNum[i])
        else:
            negVar.append(arrVar[i])
            negNum.append(arrNum[i]*-1) 
    negDot = bDotnew(negVar, negNum, bw_obj, F_prec, bits)
    posDot = bDotnew(posVar, posNum, bw_obj,F_prec, bits)
    return tm.mk_term(bw.Kind.BV_SUB, [posDot, negDot])


def bDotnew(arrVar, arrNum, bw_obj, F_prec, bits):
    tm, opt, parser, bvsizeB = bw_obj
    ln = len(arrVar)
    if(ln == 0):
        return tm.mk_bv_value(bvsizeB, 0)
    if(ln == 1):
        bvsizeB_larger = tm.mk_bv_sort(bits + F_prec)
        F_btor_larger = tm.mk_bv_value(bvsizeB_larger, F_prec)
        cnst = tm.mk_bv_value(bvsizeB_larger, arrNum[0])
        var = tm.mk_term(bw.Kind.BV_SIGN_EXTEND, [arrVar[0]], [F_prec])
        tmp = tm.mk_term(bw.Kind.BV_ASHR, [ bMul(var, arrNum[0], bw_obj, bvsizeB_larger, 0) , F_btor_larger])
        tmp = tm.mk_term(bw.Kind.BV_EXTRACT, [tmp], [bits-1, 0])
        return tmp
    part = ln // 2
    return tm.mk_term(bw.Kind.BV_ADD, [bDotnew(arrVar[:part], arrNum[:part], bw_obj, F_prec, bits), bDotnew(arrVar[part:], arrNum[part:], bw_obj, F_prec, bits)])


def bElemMul(bw_obj, mat1, scaled_weight, F_prec, bits, isDebug=False):
    tm, opt, parser, bvsizeB = bw_obj
    out = []
    for i in range(len(mat1)):
        if isDebug:
            breakpoint()
        bvsizeB_larger = tm.mk_bv_sort(bits + F_prec)
        F_btor_larger = tm.mk_bv_value(bvsizeB_larger, F_prec)
        cnst = tm.mk_bv_value(bvsizeB_larger, scaled_weight[i])
        var = tm.mk_term(bw.Kind.BV_SIGN_EXTEND, [mat1[i]], [F_prec])
        tmp = tm.mk_term(bw.Kind.BV_ASHR, [ bMul(var, scaled_weight[i], bw_obj, bvsizeB_larger, 0) , F_btor_larger])
        tmp = tm.mk_term(bw.Kind.BV_EXTRACT, [tmp], [bits-1, 0])
        out.append(tmp)
    return out

def bSum(arrVar, bw_obj, F_prec, bits):
    tm, opt, parser, bvsizeB = bw_obj
    ln = len(arrVar)
    if(ln == 0):
        return tm.mk_bv_value(bvsizeB, 0)
    if(ln == 1):
        return arrVar[0]
    part = ln // 2
    return tm.mk_term(bw.Kind.BV_ADD, [bSum(arrVar[:part], bw_obj, F_prec, bits), bSum(arrVar[part:], bw_obj, F_prec, bits)])

def bSignFnc(bw_obj, arrVar, bits, isLast, F_prec, gap):
    # msb -> sign bit
    # Determine if x is zero
    # is_zero_v1 = (inp == 0)
    # is_zero_v2 = (gap > inp) & (inp > -gap)
    # Compute the sign: 1 for positive, 0 for zero, -1 for negative
    # sign = is_zero ? 0 : (msb ? -1 : 1)
    tm, opt, parser, bvsizeB = bw_obj
    signs = []
    val = 1 if isLast else 2**(F_prec)
    bZero = tm.mk_bv_value(bvsizeB,  0)
    bneg1 = tm.mk_bv_value(bvsizeB, -val)
    bpos1 = tm.mk_bv_value(bvsizeB,  val)
    bgap = tm.mk_bv_value(bvsizeB,  int(gap*2**(F_prec)))
    bgap_neg = tm.mk_bv_value(bvsizeB,  -int(gap*2**(F_prec)))

    for i in range(len(arrVar)):
        is_zero_v1 =  tm.mk_term(bw.Kind.EQUAL, [arrVar[i], bZero])
        is_zero_v2 =  tm.mk_term(bw.Kind.AND, [ tm.mk_term(bw.Kind.BV_SGT, [bgap, arrVar[i]]), tm.mk_term(bw.Kind.BV_SGT, [arrVar[i], bgap_neg])])
        msb = tm.mk_term(bw.Kind.BV_EXTRACT, [arrVar[i]], [bits-1, bits-1])   
        msb = tm.mk_term(bw.Kind.EQUAL, [msb, tm.mk_bv_value(tm.mk_bv_sort(1),  1)])
        sign = tm.mk_term(bw.Kind.ITE, [is_zero_v1, bZero, tm.mk_term(bw.Kind.ITE, [msb, bneg1, bpos1])])
        signs.append(sign)

    return signs

"""
                               -----------------------------------
                                Miscellaneous Bitwuzla Functions
                               -----------------------------------
"""

def bShiftInc(arr, F_prec, bw_obj):
    tm, opt, parser, bvsizeB = bw_obj
    arr2 = []
    for i in range(len(arr)):
        arr2.append(tm.mk_term(bw.Kind.BV_SHL, [arr[i], F_prec]))
    return arr2

def bMat(mat, bw_obj):
    tm, opt, parser, bvsizeB = bw_obj
    matrix = [[tm.mk_bv_value(bvsizeB, mat[i][j]) for j in range(len(mat[i]))] for i in range(len(mat))]
    return matrix

def bVec(vec, bw_obj):
    tm, opt, parser, bvsizeB = bw_obj
    vector = [tm.mk_bv_value(bvsizeB, vec[i]) for i in range(len(vec))]
    return vector

def bAnd(arr, bw_obj):
    tm, opt, parser, bvsizeB = bw_obj
    if(len(arr) == 1):
        return arr[0]
    if(len(arr) == 2):                              # REMOVE THIS CASE ITS REDUNDANT
        return tm.mk_term(bw.Kind.AND, [arr[0], arr[1]])
    part = len(arr) // 2
    return tm.mk_term(bw.Kind.AND, [bAnd(arr[:part], bw_obj), bAnd(arr[part:], bw_obj)])

def bv2int(arr, bw_obj, bits):
    tm, opt, parser, bvsizeB = bw_obj
    arr2 = []
    for i in range(len(arr)):
        arr2.append(todecimal(arr[i], bw_obj, bits))
    return arr2

def todecimal(x, bw_obj, bits):
    tm, opt, parser, bvsizeB = bw_obj
    val = int(parser.bitwuzla().get_value(x).value(10))
    s = 1 << (bits - 1)
    return (val & s - 1) - (val & s)


def bPrint(arr, bw_obj):
    tm, opt, parser, bvsizeB = bw_obj
    for ar in arr:
        print(f" {ar.symbol()} --> {parser.bitwuzla().get_value(ar).value(10)} ")

def BLessThan(rank_before, rank_after, context):
    state_vars, inp_out_vars, bw_obj, bits = context
    tm, opt, parser, bvsizeB = bw_obj
    return tm.mk_term(bw.Kind.BV_SLT, [rank_before, rank_after])

def BLessThanEq(rank_before, rank_after, context):
    state_vars, inp_out_vars, bw_obj, bits = context
    tm, opt, parser, bvsizeB = bw_obj
    return tm.mk_term(bw.Kind.BV_SLE, [rank_before, rank_after])

def BLessThanEps(rank_before, rank_after, delta, context, F_prec, isDebug = False):
    state_vars, inp_out_vars, bw_obj, bits = context
    tm, opt, parser, bvsizeB = bw_obj
    dt = tm.mk_bv_value(bvsizeB, int(math.floor(delta)))
    res = tm.mk_term(bw.Kind.BV_SLT, [tm.mk_term(bw.Kind.BV_SUB, [rank_before, dt]), rank_after])
    if isDebug:
        breakpoint()
    return res

def bSetListUnEqual(l1, l2, bw_obj):
    tm, opt, parser, bvsizeB = bw_obj
    res = []
    for i in range(len(l1)):
        res.append(tm.mk_term(bw.Kind.NOT, [tm.mk_term(bw.Kind.EQUAL, [l1[i], tm.mk_bv_value(bvsizeB, int(l2[i]))])]))
    return res

"""
                               -----------------------------------
                                Bitwuzla Functions for CAV'25 NRF
                               -----------------------------------
"""

def bLinear(bw_obj, W, b, inp, F_prec, bits, isDebug = False):
    tm, opt, parser, bvsizeB = bw_obj
    scaled_W_py = (W * (2**F_prec)).astype(int).tolist()
    scaled_b_py = (b * (2**F_prec)).astype(int).tolist()
    scaled_b_bw = bVec(scaled_b_py, bw_obj)
    out1 = []
    for i in range(len(scaled_W_py)):
        tmp = bDotPositive(inp, scaled_W_py[i], bw_obj, F_prec, bits)
        out1.append(tmp)

    out = []
    for i in range(len(scaled_b_bw)):
        out.append(tm.mk_term(bw.Kind.BV_ADD, [out1[i], scaled_b_bw[i]]))

    if isDebug:
        print(f"bLin inp: {bv2int(inp, bw_obj, bits)}")
        print(f"bLin W: {scaled_W_py} b: {scaled_b_py}")
        print(f"bLin out1: {bv2int(out1, bw_obj, bits)}")
        print(f"bLin out: {bv2int(out, bw_obj, bits)}")
        breakpoint()
    return out           


def bSignNN(bw_obj, param, x, F_prec, bits, gap, isDebug):
    tm, opt, parser, bvsizeB = bw_obj
    W0, b0 = param[0]
    h_i = bLinear(bw_obj, W0, b0, x, F_prec, bits, isDebug)
    s_i = bSignFnc(bw_obj, h_i, bits, 1 == len(param), F_prec, gap)
    if isDebug:
        breakpoint()
    for i, (W, b) in enumerate(param[1:], 1):
        h_i = bLinear(bw_obj, W, b, s_i, F_prec, bits)
        s_i = bSignFnc(bw_obj, h_i, bits, i+ 1 == len(param), F_prec, gap)      
    return s_i

def bCAV_NRF(bw_obj, nnparam, clparam, inp, scale, F_prec, bits, gap, isDebug = False):
    tm, opt, parser, bvsizeB = bw_obj
    bZero = tm.mk_bv_value(bvsizeB, 0)

    F_btor = tm.mk_bv_value(bvsizeB, F_prec)
    scaled_inp1 = np.array(bShiftInc(inp, F_btor, bw_obj))

    scale_py = [int(scale * (2**F_prec))]*len(scaled_inp1)
    scaled_inp = bElemMul(bw_obj, scaled_inp1, scale_py, F_prec, bits)
    
    if nnparam == None:
        z = bLinear(bw_obj, *clparam, scaled_inp, F_prec, bits, isDebug)
        V = bSum(z, bw_obj, F_prec, bits)
        return V

    w = bSignNN(bw_obj, nnparam, scaled_inp, F_prec, bits, gap, isDebug)
    z = bLinear(bw_obj, *clparam, scaled_inp, F_prec, bits, isDebug )
    
    #If(wi > 0, zi, 0)
    V_arr = [tm.mk_term(bw.Kind.ITE, [tm.mk_term(bw.Kind.BV_SGT, [wi, bZero]), zi, bZero]) for wi, zi in zip(w,z)]
    V = bSum(V_arr, bw_obj, F_prec, bits)
    if isDebug:
        breakpoint()
    return V

def check_tran(q_cur, q_nex, trans_, V_cur, V_nex, nnparam, clparam, curr_vars, next_vars, non_state_vars, scale,  ctx, is_acc, F_prec, bw_obj, bits, gap, kappa_quant):
    tm, opt, parser, bvsizeB = bw_obj
    cex_trans = []
    for cex_i in range(1):
        parser.bitwuzla().push()
        for cex in cex_trans:
            parser.bitwuzla().assert_formula(bAnd(bSetListUnEqual(curr_vars, cex[1], bw_obj), bw_obj))
            parser.bitwuzla().assert_formula(bAnd(bSetListUnEqual(next_vars, cex[3], bw_obj), bw_obj))
                
        if len(trans_) > 0:
            parser.bitwuzla().assert_formula(bAnd(trans_, bw_obj))
        
        if is_acc[q_cur] == 1:
            eps = 1 #scale*.9*(2**F_prec)
            cnd_pre = BLessThanEq(V_cur, kappa_quant, ctx)
            cnd_post = BLessThanEps(V_cur, V_nex, eps, ctx, F_prec)
            parser.bitwuzla().assert_formula(bAnd([cnd_pre, cnd_post], bw_obj))
        else:
            cnd_pre = BLessThanEq(V_cur, kappa_quant, ctx)
            cnd_post = BLessThan(V_cur, V_nex, ctx)
            parser.bitwuzla().assert_formula(bAnd([cnd_pre, cnd_post], bw_obj))
        
        res = parser.bitwuzla().check_sat()
        if (res == bw.Result.SAT):
            #bPrint(curr_vars, bw_obj)
            #bPrint(next_vars, bw_obj)
            #bPrint(non_state_vars, bw_obj)
            c_cur = np.array(bv2int(curr_vars, bw_obj, bits)) 
            c_nex = np.array(bv2int(next_vars, bw_obj, bits))
            cex_trans.append((q_cur, c_cur, q_nex, c_nex))
            print(f"{Fore.CYAN}q = {q_cur} to q = {q_nex} is SAT {(q_cur, c_cur, q_nex, c_nex)} {Style.RESET_ALL}")
            
            V_eval_q = nnGurobi.evalQuantNN(nnparam[q_cur], *clparam[q_cur], c_cur, scale, gap, F_prec)
            V_nex_eval_q = nnGurobi.evalQuantNN(nnparam[q_nex], *clparam[q_nex], c_nex, scale, gap, F_prec)
            V_eval = nnGurobi.evalFunkyNN(nnparam[q_cur], *clparam[q_cur], c_cur)
            V_nex_eval = nnGurobi.evalFunkyNN(nnparam[q_nex], *clparam[q_nex], c_nex)
            print(f"{Fore.WHITE}\tBitwuzla Rank [{todecimal(V_cur, bw_obj, bits)/2**F_prec} -> {todecimal(V_nex, bw_obj, bits)/2**F_prec}]; Numpy Rank [{V_eval} -> {V_nex_eval}]; ; Numpy RankQ [{V_eval_q/2**F_prec} -> {V_nex_eval_q/2**F_prec}]  {Style.RESET_ALL}")
            
            #if(todecimal(V_cur, bw_obj, bits)/2**F_prec != V_eval):
            #    print("[POTENTIAL BUG]")
            #    V_cur = bCAV_NRF(bw_obj, nnparam[q_cur], clparam[q_cur], curr_vars, scale, F_prec, bits, gap, isDebug =  True)
            #    V_eval = nnGurobi.evalQuantNN(nnparam[q_cur], *clparam[q_cur], c_cur, scale, gap, F_prec, isDebug = True)
            #
            #if (todecimal(V_nex, bw_obj, bits)/2**F_prec != V_nex_eval):
            #    print("[POTENTIAL BUG]")
            #    V_nex = bCAV_NRF(bw_obj, nnparam[q_nex], clparam[q_nex], next_vars, scale, F_prec, bits, gap, isDebug = True)
            #    V_eval = nnGurobi.evalQuantNN(nnparam[q_nex], *clparam[q_nex], c_nex, scale, gap, F_prec, isDebug = True)
            #cnd = BLessThanEps(V_cur, V_nex, eps, ctx, F_prec, (c_cur==[255] and c_nex==[0]))
        else:
            print(f"{Fore.BLUE}q = {q_cur} to q = {q_nex} is UNSAT{Style.RESET_ALL}")
        parser.bitwuzla().pop()

    return cex_trans

def check(nnparam, clparam, curr_vars, next_vars, non_state_vars, scale, spec_automata, ctx, q_set, is_acc, F_prec, bw_obj, bits, gap, kappa):
    cex = []
    tm, opt, parser, bvsizeB = bw_obj

    kappa_quant = tm.mk_bv_value(bvsizeB, int(kappa * (2**F_prec)))
    for q_cur, q_nex in product(q_set, repeat=2):
        V_cur = bCAV_NRF(bw_obj, nnparam[q_cur], clparam[q_cur], curr_vars, scale, F_prec, bits, gap)
        V_nex = bCAV_NRF(bw_obj, nnparam[q_nex], clparam[q_nex], next_vars, scale, F_prec, bits, gap)

        for trans_ in spec_automata(ctx, q_cur, curr_vars, None, q_nex, next_vars, None, non_state_vars, 1):
            cex += check_tran(q_cur, q_nex, trans_, V_cur, V_nex, nnparam, clparam, curr_vars, next_vars, non_state_vars, scale,  ctx, is_acc, F_prec, bw_obj, bits, gap, kappa_quant)
                
    return cex


def check_init(nnparam, clparam, curr_vars, init_state_q,  F_prec, bw_obj, scale, bits, gap, q_set, ctx, kappa):
    invar_cex = []
    tm, opt, parser, bvsizeB = bw_obj
    kappa_quant = tm.mk_bv_value(bvsizeB, int(kappa * (2**F_prec)))    
    for q0 in q_set:
        if(init_state_q[q0] == 0):
            continue
        V_cur = bCAV_NRF(bw_obj, nnparam[q0], clparam[q0], curr_vars, scale, F_prec, bits, gap)
        parser.bitwuzla().push()
        cnd = BLessThan(kappa_quant, V_cur, ctx)
        parser.bitwuzla().assert_formula(cnd)
        res = parser.bitwuzla().check_sat()
        if (res == bw.Result.SAT):
            c_cur = np.array(bv2int(curr_vars, bw_obj, bits)) 
            invar_cex.append((q0, c_cur))
            print(f"{Fore.CYAN}q = {q0} [InVar] is SAT {(q0, c_cur)} {Style.RESET_ALL}")
 
            V_eval_q = nnGurobi.evalQuantNN(nnparam[q0], *clparam[q0], c_cur, scale, gap, F_prec)
            V_eval = nnGurobi.evalFunkyNN(nnparam[q0], *clparam[q0], c_cur)
            print(f"{Fore.WHITE}\tBitwuzla Rank [{todecimal(V_cur, bw_obj, bits)/2**F_prec}]; Numpy Rank [{V_eval}]; ; Numpy RankQ [{V_eval_q/2**F_prec}]  {Style.RESET_ALL}")
        else:
            print(f"{Fore.BLUE}q = {q0} [InVar] is UNSAT{Style.RESET_ALL}")    
        parser.bitwuzla().pop()
    return invar_cex

