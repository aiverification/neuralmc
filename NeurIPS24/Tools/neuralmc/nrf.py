"""
Author:
--------
Abhinandan Pal  
University of Birmingham

Copyright:
-----------
© 2024 University of Birmingham. All rights reserved.

For Theoratical Details refer to [1], specifically section 3.

[1] Mirco Giacobbe, Daniel Kroening, Abhinandan Pal, and Michael Tautschnig (alphabetical). “Neural Model Checking”.
Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS’24), December 9-15, 2024, Vancouver, Canada.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as optim2
import torch.nn.functional as F
import numpy as np
import torch.fx
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR
from colorama import init, Fore, Back, Style
colours = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTCYAN_EX]
col_num = len(colours)


"""
                                   -----------------
                                    NRF_Clamp Class
                                   -----------------
The NRF_Clamp class implements a neural ranking function with Clamp activations and fully connected layers with configurable number of neurons. 

Constructor Parameters:
------------------------
- n_vars (int): Number of input variables.
- scale (float): Scaling factor for input normalization.
- clamp_bits (int): Defines the maximum clamp range as `2^clamp_bits`.
- nnP[:2] (str): Determines layer sizes. Options are "A1", "A2", "A3", "A4", "A5". Invalid values trigger a breakpoint.
- nnP[3:] (str): Determines whether the second layer should be a trainable element-wise multiplication ("Default") or a fully connected layer ("ExtraHidden").

Trainable Parameter:
--------
- weightedL (nn.Parameter): Scaling inputs on importance.
- fc_1, fc_2, fc_3 (nn.Linear): Fully connected layers based on `nnP`.

Clamping Range:
---------------
- clamp_min: 0
- clamp_max: `2^clamp_bits`

Forward Pass:
-------------
1. Normalize input by `scale`.
2. Apply element-wise scaling with `weightedL`.
3. Pass through `fc_1`, `fc_2`, `fc_3`, with clamping on `fc_1` and `fc_2`.
4. Return final output.
"""


class NRF_Clamp(nn.Module):
    def __init__(self, n_vars, scale, clamp_bits, nnP):
        super().__init__()

        # Layer configurations based on nnP
        layer_config = {
            "A1": (3, 2),
            "A2": (5, 3),
            "A3": (8, 5),
            "A4": (15, 8),
            "A5": (30, 15)
        }
        self.nnP1 = nnP[:2]
        self.nnP2 = nnP[3:]
        if self.nnP1 not in layer_config:
            breakpoint()
        l1_c, l2_c = layer_config[self.nnP1]

        # Element wise layer parameters
        self.scale = nn.Parameter(torch.Tensor(scale), requires_grad=False)
        if self.nnP2 == "Default":
            self.weightedL = nn.Parameter(torch.rand(n_vars), requires_grad=True)
        elif self.nnP2 == "ExtraHidden":
            self.fc_0 = nn.Linear(n_vars, n_vars, bias=True)
        else:
            breakpoint()

        # Fully connected layers
        self.fc_1 = nn.Linear(n_vars, l1_c, bias=True)
        self.fc_2 = nn.Linear(l1_c, l2_c, bias=True)
        self.fc_3 = nn.Linear(l2_c, 1, bias=True)

        # Clamping range
        self.clamp_min = 0
        self.clamp_max = 2 ** clamp_bits

    def forward(self, state):
        state = state / self.scale

        if self.nnP2 == "Default":
            state = state * self.weightedL
        elif self.nnP2 == "ExtraHidden":
            state = self.fc_0(state)
            state = state.clamp( min = self.clamp_min, max = self.clamp_max)
        

        state = self.fc_1(state)
        state = state.clamp( min = self.clamp_min, max = self.clamp_max)
        
        state = self.fc_2(state)
        state = state.clamp( min = self.clamp_min, max = self.clamp_max)

        state = self.fc_3(state)
        return state


"""
                                   ---------------
                                     NRF Training
                                   ---------------
The `training` function handles the iterative training process for a neural ranking function (NRF) using gradient descent.
It computes loss values, updates model parameters, and clips gradients to prevent exploding gradients.

Functions:
-----------
- `clip_gradients(model, max_norm)`: 
    Clips the gradients of the model's parameters to a maximum norm.
    
- `p_nrf(pnrf, inps)`: 
    Computes the output of the neural ranking functions for a state on synchronous
    composition M ∥∥ A¬Φ, picking the trained network based on the the A¬Φ state component.
    
- `loss_term(epoch, isAcc, trans, pnrf, delta, q)`: 
    Computes the loss for a particular transition pair.

Parameters:
------------
- pnrf (dict): 
    Dictionary of neural networks, keyed by  A¬Φ state.
    
- optimiserPNRF (dict): 
    Dictionary of optimizers, one for each neural networks in pnrf, keyed by A¬Φ state.
    
- all_trans (dict): 
    The dataset consists of transition pairs. The elements are organized into a dictionary
    keyed by the next A¬Φ state for debugging convenience.

- all_states (dict): 
    All states used during training. Used to compute max and min output for each A¬Φ state, post training.
    
- stt_acc (list): 
    List of accepting/fair A¬Φ state.
    
- delta (float): 
    Margin for decrease for accepting/fair transitions.

Training Process:
------------------
1. Display the individual loss for each neural network in the NRF, along with their sum, before starting training.
2. Iterate for `n_training_iters` by:
   a) computing the total loss,
   b) backpropagating through the q-th neural network, where 'q' loops over A¬Φ states,
   c) applying gradient clipping to prevent exploding gradients.
3. Print the loss (as in step 1) every 10 iterations and check for termination if the total loss reaches zero. 

Returns:
---------
- The final total loss after training (`float`). If non-zero, it means the training failed.

"""


max_gradient_norm = 0.01

def clip_gradients(model, max_norm):
    parameters = [p for p in model.parameters() if p.grad is not None]
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)

def p_nrf(pnrf, inps):
    out = None
    for key, nrf in pnrf.items():
        if key == 0:
            out = pnrf[key](inps)[:, 0] * (inps[:, -1] == key)
        else:
            out += pnrf[key](inps)[:, 0] * (inps[:, -1] == key)
    return out

def loss_term(epoch, isAcc, trans, pnrf, delta, q):
    if len(trans) == 0:
        return torch.tensor([0.0])
    if isAcc:
        return F.relu(p_nrf(pnrf, trans[:, 1]) - p_nrf(pnrf, trans[:, 0]) + delta)
    return F.relu(p_nrf(pnrf, trans[:, 1]) - p_nrf(pnrf, trans[:, 0]))

def training(pnrf, optimiserPNRF, all_trans, all_states, stt_acc, pnum, delta):
    n_training_iters = 10000

    losses_main = {
        q: loss_term(0, (q in stt_acc), all_trans[q], pnrf, delta, q).mean() for q in pnrf
    }
    losses_main_print = {f"l{q}": losses_main[q].item() for q in losses_main}
    loss_total = sum(losses_main.values())

    print(f"{colours[pnum % col_num]} [{pnum}] loss before : {loss_total} --> \n\t\t\t{losses_main_print} LRate: {optimiserPNRF[0].param_groups[0]['lr']} {Style.RESET_ALL}")
    
    torch.autograd.set_detect_anomaly(True)
    for i in range(n_training_iters):
        if False: #i % 990 == 0:
            #torch.nonzero(loss2, as_tuple=True)
            for q in all_trans.keys():
                if len(all_trans[q]) != 0:
                    print(f" T->{q}: {q in stt_acc} {p_nrf(pnrf, all_trans[q][:3,0]).tolist()} ---> {p_nrf(pnrf, all_trans[q][:3,1]).tolist()}")
            for q in all_trans.keys():
                if len(all_states[all_states[:,-1] == q]) != 0:
                    print(f" Q{q}: ({torch.min(pnrf[q](all_states[all_states[:,-1] == q]))}, {torch.max(pnrf[q](all_states[all_states[:,-1] == q]))})")
            breakpoint()

        if i % 10 == 0:
            print(f"{colours[pnum % col_num]} [{pnum}] loss [{i}] : {loss_total} --> \n\t\t\t{losses_main_print} LRate: {optimiserPNRF[0].param_groups[0]['lr']} {Style.RESET_ALL}")

        q = i % (max(pnrf.keys()) + 1)

        if losses_main[q].item() == 0.0:
            continue

        optimiserPNRF[q].zero_grad()
        loss_total.backward()
        clip_gradients(pnrf[q], max_gradient_norm)
        optimiserPNRF[q].step()

        losses_main = {
            q: loss_term(i, (q in stt_acc), all_trans[q], pnrf, delta, q).mean() for q in all_trans
        }
        losses_main_print = {f"l{q}": losses_main[q].item() for q in losses_main}
        loss_total = sum(losses_main.values())

        if sum(losses_main.values()).item() == 0.0:
            print(f"{colours[pnum % col_num]} [{pnum}] loss [{i}] : {sum(losses_main.values())} --> \n\t\t\t{losses_main_print}  LRate: {optimiserPNRF[0].param_groups[0]['lr']} {Style.RESET_ALL}")
            for q in all_trans.keys():
                if len(all_trans[q]) != 0:
                    print(f" T->{q}: {q in stt_acc} {p_nrf(pnrf, all_trans[q][:3,0]).tolist()} ---> {p_nrf(pnrf, all_trans[q][:3,1]).tolist()}")
            for q in all_trans.keys():
                if len(all_states[all_states[:,-1] == q]) != 0:
                    print(f" Q{q}: ({torch.min(pnrf[q](all_states[all_states[:,-1] == q]))}, {torch.max(pnrf[q](all_states[all_states[:,-1] == q]))})")
            return sum(losses_main.values()).item()

    return loss_total.item()
