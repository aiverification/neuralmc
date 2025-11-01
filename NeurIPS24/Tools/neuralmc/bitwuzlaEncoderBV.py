"""
Author:
--------
Abhinandan Pal  
University of Birmingham

Copyright:
-----------
© 2024 University of Birmingham. All rights reserved.

For Theoratical Details refer to [1].

[1] Mirco Giacobbe, Daniel Kroening, Abhinandan Pal, and Michael Tautschnig (alphabetical). “Neural Model Checking”.
Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS’24), December 9-15, 2024, Vancouver, Canada.
"""

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

"""
                                   ----------------------
                                    Bitwuzla Dot Product
                                   ----------------------

Overview:
----------
These functions are used for bit-vector arithmetic operations using Bitwuzla.
They implement multiplication and dot product calculations using SMT (Satisfiability Modulo Theories) terms for handling large bit-vectors efficiently.

1. **bMul**:
    Recursively performs a multiplication of a bit-vector variable by a number using bitwise shifts and additions to reduce complexity for the Bitwuzla SMT solver.

2. **bDotPositive**:
    The `bDotPositive` function computes the dot product of bit-vector variables (`arrVar`) and corresponding numerical values (`arrNum`), while separately handling positive and negative values.
    It calculates the dot products for positive and negative numbers individually by using their absolute values, then subtracts the negative result from the positive one.
    This approach avoids the need to encode multiplication with negative numbers in the SMT encoding.

3. **bDotnew**:
    It encodes a balanced dot product, where an expression like 2x_1 + 4x_2 + 3x_3 + 7x_4 is grouped as ((2x_1 + 4x_2) + (3x_3 + 7x_4)), instead of the more straightforward but unbalanced encoding ((((2x_1 + 4x_2) + 3x_3) + 7x_4)).

Parameters:
------------
- `bMul(bvar, number, bw_obj, bvsizeB_larger, dpt)`:
    - `bvar`: The bit-vector variable.
    - `number`: The constant to multiply the bit-vector by.
    - `bw_obj`: Tuple containing SMT-related objects (term manager, optimizer, parser, and bit-vector size).
    - `bvsizeB_larger`: Larger bit-vector size for storing the result of the multiplication.
    - `dpt`: Recursion depth for recursive calls.
    
- `bDotPositive(arrVar, arrNum, bw_obj, F_prec, bits)`:
    - `arrVar`: Array of bit-vector variables.
    - `arrNum`: Array of corresponding integers for the dot product.
    - `bw_obj`: Tuple containing SMT-related objects (term manager, optimizer, parser, and bit-vector size).
    - `F_prec`: Fixed-point precision for the operation.
    - `bits`: Total bit width for the result.
    
- `bDotnew(arrVar, arrNum, bw_obj, F_prec, bits)`:
    - `arrVar`: Array of bit-vector variables.
    - `arrNum`: Array of corresponding integers for the dot product.
    - `bw_obj`: Tuple containing SMT-related objects (term manager, optimizer, parser, and bit-vector size).
    - `F_prec`: Fixed-point precision for the operation.
    - `bits`: Total bit width for the result.

Returns:
---------
- `bMul`: Returns the result of the bit-vector multiplication using shifts and additions.
- `bDotPositive`: Returns the result of the dot product after handling positive and negative values.
- `bDotnew`: Returns the computed dot product, handling different array lengths recursively.

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
        #tmp = tm.mk_term(bw.Kind.BV_ASHR, [ tm.mk_term(bw.Kind.BV_MUL, [cnst, var]) , F_btor_larger])
        tmp = tm.mk_term(bw.Kind.BV_ASHR, [ bMul(var, arrNum[0], bw_obj, bvsizeB_larger, 0) , F_btor_larger])
        tmp = tm.mk_term(bw.Kind.BV_EXTRACT, [tmp], [bits-1, 0])
        return tmp
    part = ln // 2
    return tm.mk_term(bw.Kind.BV_ADD, [bDotnew(arrVar[:part], arrNum[:part], bw_obj, F_prec, bits), bDotnew(arrVar[part:], arrNum[part:], bw_obj, F_prec, bits)])


"""
                               -----------------------------------
                                Bitwuzla Dot Product (DEPRECATED)
                               -----------------------------------

Overview:
----------
Deprecated implementation of the dot product, provided for reference purposes.

1. **bDot (Balanced Encoding)**:
    Encodes the dot product in a balanced manner, grouping terms like:
    (2x_1 + 4x_2 + 3x_3 + 7x_4) as ((2x_1 + 4x_2) + (3x_3 + 7x_4)).

2. **btorDot2 (Unbalanced Encoding)**:
    Implements a more straightforward but unbalanced encoding, where terms are grouped as:
    ((((2x_1 + 4x_2) + 3x_3) + 7x_4)).
"""


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

"""
                               -----------------------------------
                                Miscellaneous Bitwuzla Functions
                               -----------------------------------
Overview:
----------
These module provides several utility functions for handling bit-vector operations using Bitwula SMT solver. 

Functions:
-----------

1. **bShiftInc(arr, F_prec, bw_obj)**:
    Applies a bitwise left shift to each element in the given array of bit-vectors.

2. **bMat(mat, bw_obj)**:
    Converts a 2D matrix of integers into a 2D matrix of bit-vector terms.

3. **bVec(vec, bw_obj)**:
    Converts a vector of integers into a vector of bit-vector terms.

4. **todecimal(x, bits)**:
    Converts a bit-vector integer to its decimal representation, accounting for signed integers.

5. **bPrint(arr, bw_obj, bits, F_prec)**:
    Prints the bit-vector values from `arr` in a human-readable decimal format, adjusted by Fixed-point precision factor (`F_prec`).
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

def todecimal(x, bits):
    s = 1 << (bits - 1)
    return (x & s - 1) - (x & s)

def bPrint(arr, bw_obj, bits, F_prec):
    tm, opt, parser, bvsizeB = bw_obj
    for ar in arr:
        print(f" {ar.symbol()} --> {todecimal(int(parser.bitwuzla().get_value(ar).value(10)),bits)/ 2**F_prec} ")

"""
                               --------------------------------
                                Neural Network to SMT Encoding
                               --------------------------------

Overview:
----------

The `encode` function converts a PyTorch neural network model into SMT terms using the Bitwuzla SMT solver.
It traverses the computational graph of the network to generate corresponding SMT representations.
Since the SMT encoding is based on bit-vectors, the neural network is quantized according to a fixed-point precision factor (`F_prec`).

Parameters:
------------
- **nrf** (`nn.Module`):  
  The neural network model to be encoded.

- **input** (`list` of bit-vector terms):  
  The input data to the neural network, represented as bit-vector terms.

- **F_prec** (`int`):  
  Fixed-point precision factor used for quantizing the neural network.

- **bw_obj** (`tuple`):  
  A tuple containing Bitwuzla-related objects:

- **bits** (`int`):  
  Total bit width for the bit-vector terms.

Process:
---------
1. **Symbolic Tracing**:  
   - Uses `torch.fx.symbolic_trace` to generate a symbolic graph (`gm`) of the neural network.
   - Maps the network's modules for easy reference.

2. **Node Processing**:  
   Processes each node in the graph based on `node.op`:

   - **Placeholder (Input)**:  
     Scales the input node `'state'` using fixed-point precision and stores it in the `nodes` dictionary.

   - **Call Method (Clamp)**:  
     Encodes the `clamp` operation using SMT `ITE` constructs for bounded values.

   - **Call Module (Linear Layer)**:  
     Handles modules like `nn.Linear`, scales weights/biases, computes bit-vector dot products, and stores results.

   - **Call Function (Elementwise Multiplication)**:  
     Supports basic arithmetic (e.g., `mul`, `truediv`), scales constants, performs bit-vector arithmetic, and stores results.

   - **Get Attribute (Weights & Biases)**:  
     Retrieves weights from the neural network model.

   - **Output**:  
     Returns the final SMT term from the processed nodes.

   - **Unsupported Operations**:  
     Raises an exception for unsupported node operations.

3. **Helper Functions**:
   - **`load_arg(a)`**:  
     Maps node arguments to corresponding SMT terms in `nodes`.

Notes:
-------
- The function assumes that the neural network uses supported layers and operations (e.g., linear layers, clamp).
- Fixed-point scaling is crucial for maintaining numerical precision during the SMT encoding.
- Error handling is implemented to ensure unsupported operations are flagged during the encoding process.

Returns:
--------- 
  The SMT term representing the encoded output of the neural network model.

"""


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
                    #tmp = bDotnew(inp, scaled_weight1[i], bw_obj, F_prec, bits)#bDotPositive(inp, scaled_weight1[i], bw_obj, F_btor) # bDot(inp, scaled_weight[i], bw_obj)
                    tmp = bDotPositive(inp, scaled_weight1[i], bw_obj, F_prec, bits)
                    #tmp = bDot(inp, scaled_weight[i], bw_obj)
                    #tmp = tm.mk_term(bw.Kind.BV_ASHR, [tmp, F_btor]) # signed right shift
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
                #tmp = tm.mk_term(bw.Kind.BV_ASHR, [ tm.mk_term(bw.Kind.BV_MUL, [cnst, var]) , F_btor_larger])
                tmp = tm.mk_term(bw.Kind.BV_ASHR, [ bMul(var, scaled_weight[i], bw_obj, bvsizeB_larger, 0) , F_btor_larger])
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

def old_encode(nrf, input, F_prec, bw_obj, bits):
    print("\t\t\t---OLD SMT ENCODING (NOT RECOMMENDED)---")
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


"""
                               --------------------------------------
                                Neural Network to SystemVerilog Code
                               --------------------------------------
Identical to "Neural Network to SMT Encoding," but instead of translating each component to SMT, it is converted into lines of SystemVerilog code.
"""

def encodeAsSTR(nrf, input, F_prec, bw_obj, bits):
    tm, opt, parser, bvsizeB = bw_obj
    gm = torch.fx.symbolic_trace(nrf)
    #gm.graph.print_tabular()
    modules = dict(nrf.named_modules())
    nodes = {}
    F_btor = tm.mk_bv_value(bvsizeB, F_prec)
    string_def = []
    string_exp = []
    ln = 0
    def load_arg(a):
        return torch.fx.graph.map_arg(a, lambda n: nodes[n.name])
    for node in gm.graph.nodes:
        #print(f"OP: {node.op} NAME: {node.name} Target: {node.target}")
        assert node.name not in nodes
        if node.op == 'placeholder':
            assert node.name == 'state'
            scaled_inp = [f"(v{i} << {F_prec})" for i in range(len(input))] #[inp.symbol()+"<<"+ str(F_prec) for inp in input]
            string_exp += [f"l{ln}v{i} = {input[i].symbol()} << {F_prec}" for i in range(len(input))]
            string_def += [f"l{ln}v{i}" for i in range(len(input))]
            ln += 1
            nodes[node.name] =  scaled_inp # Transpose
        elif node.op == 'call_method' and node.target == 'clamp':
            inp, = load_arg(node.args)
            lb = int(node.kwargs['min']*2**F_prec)
            ub = int(node.kwargs['max']*2**F_prec)
            #nodes[node.name] = [f"({inp[i]} > {lb})?(({inp[i]} < {ub})?{inp[i]}:{ub}):{lb}" for i in range(len(inp))]
            nodes[node.name] = [f"(({inp[i]} if ({inp[i]} < {ub}) else {ub}) if ({inp[i]} > {lb}) else {lb})" for i in range(len(inp))]
            #string_exp += [f"l{ln}v{i} = (l{ln-1}v{i} if (l{ln-1}v{i} < {ub}) else {ub}) if (l{ln-1}v{i} > {lb}) else {lb}" for i in range(len(inp))]
            string_exp += [f"l{ln}v{i} = (l{ln-1}v{i} > {lb})?((l{ln-1}v{i} < {ub})?l{ln-1}v{i}:{ub}):{lb}" for i in range(len(inp))]
            string_def += [f"l{ln}v{i}" for i in range(len(inp))]
            ln += 1
        elif node.op == 'call_module':
            module = modules[node.name]
            if(isinstance(module, nn.Linear)):   
                inp, = load_arg(node.args)
                assert not load_arg(node.kwargs)
                scaled_weight = (module.weight.detach().numpy() * (2**F_prec)).astype(int).tolist()
                scaled_bias = (module.bias.detach().numpy() * (2**F_prec)).astype(int).tolist()
                out = []
                for i in range(len(scaled_weight)):
                    tmp = ""
                    for j in range(len(scaled_weight[i])):
                        tmp += f"+ ({inp[j]} * {scaled_weight[i][j]})"
                    tmp = f"((({tmp[1:]}) >>> {F_prec}) + {scaled_bias[i]} )"
                    out.append(tmp)
                    tmp = ""
                    for j in range(len(scaled_weight[i])):
                        tmp += f"+ (l{ln-1}v{j} * {scaled_weight[i][j]})"
                    string_exp += [f"l{ln}v{i} = ((({tmp[1:]}) >>> {F_prec}) + {scaled_bias[i]} )"]
                    string_def += [f"l{ln}v{i}"]
                ln += 1
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
                for i in range(len(mat1)):
                    out.append( f"(({mat1[i]} * {scaled_weight[i]}) >> {F_prec})")   # CHANGE THIS IF NOT PYTHON
                    #out.append( f"(({mat1[i]} * {scaled_weight[i]}) >>> {F_prec})") # normalizing to 0 to N requires dividing a number
                    string_exp += [f"l{ln}v{i} = (l{ln-1}v{i} * {scaled_weight[i]}) >>> {F_prec}"]
                    string_def += [f"l{ln}v{i}"]
                ln += 1
            elif str(node.target) == "<built-in function mul>":
                scaled_weight = (mat2 * (2**F_prec)).astype(int).tolist()
                for i in range(len(mat1)):
                    out.append( f"(({mat1[i]} * {scaled_weight[i]}) >> {F_prec})")   # CHANGE THIS IF NOT PYTHON
                    #out.append(f"(({mat1[i]} * {scaled_weight[i]}) >>> {F_prec}))")
                    string_exp += [f"l{ln}v{i} = (l{ln-1}v{i} * {scaled_weight[i]}) >>> {F_prec}"]
                    string_def += [f"l{ln}v{i}"]
                ln += 1
            else:
                raise Exception(f"Node operator {node.op} / target {node.target} is not supported")
            nodes[node.name] = out
        elif node.op == 'get_attr':
            assert hasattr(nrf, node.target)
            val = getattr(nrf, node.target)
            nodes[node.name] = val
        elif node.op == 'output':
            inp,= load_arg(node.args)
            string_def = f"reg [{bits-1}: 0] " + ", ".join(string_def)
            string_exp = ";\n".join(string_exp) + ""
            return [string_def, string_exp]
        else:
            raise Exception(f"Node operator {node.op} / target {node.target} is not supported")
        #print(bPrint(nodes[node.name]))
