# If we are sampling consecutive states using SMT sampling, we will get a lot of state_1 that are not reachable from 
# the initial state.
import bitwuzla as bw
import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as optim2
import torch.nn.functional as F
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

seed = 42

torch.manual_seed(seed)

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
        tmp = tm.mk_term(bw.Kind.BV_ASHR, [ tm.mk_term(bw.Kind.BV_MUL, [cnst, var]) , F_btor_larger])
        tmp = tm.mk_term(bw.Kind.BV_EXTRACT, [tmp], [bits-1, 0])
        return tmp
    part = ln // 2
    return tm.mk_term(bw.Kind.BV_ADD, [bDotnew(arrVar[:part], arrNum[:part], bw_obj, F_prec, bits), bDotnew(arrVar[part:], arrNum[part:], bw_obj, F_prec, bits)])


def bDot(arr1, arr2, bw_obj):
    tm, opt, parser, bvsizeB = bw_obj
    ln = len(arr1)
    if(ln == 1):
        return tm.mk_term(bw.Kind.BV_MUL, [arr1[0], arr2[0]])
    part = ln // 2
    return tm.mk_term(bw.Kind.BV_ADD, [bDot(arr1[:part], arr2[:part], bw_obj), bDot(arr1[part:], arr2[part:], bw_obj)])

def btorDot2(arr1, arr2):
    res = (arr1[0] * arr2[0])
    for i in range(1, len(arr1)):
        res += (arr1[i] * arr2[i])
    return res

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

def todecimal(x, bits):
    s = 1 << (bits - 1)
    return (x & s - 1) - (x & s)

def bPrint(arr, bw_obj, bits, F_prec):
    tm, opt, parser, bvsizeB = bw_obj
    for ar in arr:
        print(f" {ar.symbol()} --> {todecimal(int(parser.bitwuzla().get_value(ar).value(10)),bits)/ 2**F_prec} ")

def encode(nrf, input, F_prec, bw_obj, bits):
    tm, opt, parser, bvsizeB = bw_obj
    gm = torch.fx.symbolic_trace(nrf)
    #gm.graph.print_tabular()
    modules = dict(nrf.named_modules())
    nodes = {}
    F_btor = tm.mk_bv_value(bvsizeB, F_prec)
    def load_arg(a):
        return torch.fx.graph.map_arg(a, lambda n: nodes[n.name])
    for node in gm.graph.nodes:
        #print(f"OP: {node.op} NAME: {node.name} Target: {node.target}")  
        assert node.name not in nodes
        if node.op == 'placeholder':
            assert node.name == 'state'
            scaled_inp = np.array(bShiftInc(input, F_btor, bw_obj))
            nodes[node.name] = np.transpose(np.array(scaled_inp))
        elif node.op == 'call_method' and node.target == 'clamp':
            inp, = load_arg(node.args)
            lb = tm.mk_bv_value(bvsizeB, int(node.kwargs['min']*2**F_prec))
            ub = tm.mk_bv_value(bvsizeB, int(node.kwargs['max']*2**F_prec))
            clampBT = np.vectorize(lambda x: tm.mk_term(bw.Kind.ITE, [tm.mk_term(bw.Kind.BV_SGT, [x, lb]), tm.mk_term(bw.Kind.ITE, [tm.mk_term(bw.Kind.BV_SGT, [ub, x]), x, ub]), lb]))   # SGT => Signed greater than clip(x,0,ub)
            nodes[node.name] = clampBT(inp)
        elif node.op == 'call_module':
            module = modules[node.name]
            if(isinstance(module, nn.Linear)):      
                inp, = load_arg(node.args)
                assert not load_arg(node.kwargs)
                #inp = [tm.mk_term(bw.Kind.BV_ASHR, [inp_, F_btor]) for inp_ in inp]
                scaled_weight1 = (module.weight.detach().numpy() * (2**F_prec)).astype(int).tolist()
                scaled_weight = bMat(scaled_weight1, bw_obj)
                out = []
                for i in range(len(scaled_weight)):
                    tmp = bDotnew(inp, scaled_weight1[i], bw_obj, F_prec, bits)
                    out.append(tmp)
                if not module.bias is None:
                    scaled_bias = (module.bias.detach().numpy() * (2**F_prec)).astype(int).tolist()
                    scaled_bias = bVec(scaled_bias, bw_obj)
                    for i in range(len(scaled_bias)):
                        out[i] = tm.mk_term(bw.Kind.BV_ADD, [out[i], scaled_bias[i]])
                    
                nodes[node.name] = out
            else:
                raise Exception(f"Node operator {node.op} / target {node.target} is not supported")
        elif node.op == 'call_function':
            mat1, mat2 = load_arg(node.args)
            if(isinstance(mat2, torch.nn.parameter.Parameter)):
                mat2 = np.array(mat2.data)
            assert not load_arg(node.kwargs)
            out = []
            if str(node.target) == "<built-in function truediv>":
                scaled_weight = (1/mat2 * (2**F_prec)).astype(int).tolist()
            elif str(node.target) == "<built-in function mul>":
                scaled_weight = (mat2 * (2**F_prec)).astype(int).tolist()
            else:
                raise Exception(f"Node operator {node.op} / target {node.target} is not supported")
            for i in range(len(mat1)):
                bvsizeB_larger = tm.mk_bv_sort(bits + F_prec)
                F_btor_larger = tm.mk_bv_value(bvsizeB_larger, F_prec)
                cnst = tm.mk_bv_value(bvsizeB_larger, scaled_weight[i])
                var = tm.mk_term(bw.Kind.BV_SIGN_EXTEND, [mat1[i]], [F_prec])
                tmp = tm.mk_term(bw.Kind.BV_ASHR, [ tm.mk_term(bw.Kind.BV_MUL, [cnst, var]) , F_btor_larger])
                tmp = tm.mk_term(bw.Kind.BV_EXTRACT, [tmp], [bits-1, 0])
                out.append(tmp)
            nodes[node.name] = out
        elif node.op == 'get_attr':
            assert hasattr(nrf, node.target)
            val = getattr(nrf, node.target)
            nodes[node.name] = val
        elif node.op == 'output':
            inp,= load_arg(node.args)
            return inp
        else:
            raise Exception(f"Node operator {node.op} / target {node.target} is not supported")
        #print(bPrint(nodes[node.name]))
