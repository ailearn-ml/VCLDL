import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torch.optim import Optimizer


class SparseLinear(nn.Module):
    """
    Sparse Linear linear with sparse gradients
    Parameters
    ----------
    input_size: int, input size of transformation
    output_size: int, output size of transformation
    padding_idx: int, index for dummy label; embedding is not updated
    bias: boolean, default=True, whether to use bias or not
    """

    def __init__(self, input_size, output_size, padding_idx=None, bias=True):
        super(SparseLinear, self).__init__()
        if padding_idx is None:
            self.padding_idx = output_size
        self.input_size = input_size
        self.output_size = output_size + 1
        self.weight = Parameter(torch.Tensor(self.output_size, self.input_size))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            if self.bias is not None:
                self.bias.data[self.padding_idx].fill_(0)

    def forward(self, embed, shortlist):
        """
        Parameters
        ----------
        embed: torch.Tensor, input to the layer
        shortlist: torch.LongTensor evaluate these labels only
        Returns
        -------
        out: torch.Tensor, logits for each label in provided shortlist
        """
        short_weights = F.embedding(shortlist, self.weight, sparse=True, padding_idx=self.padding_idx)
        out = torch.matmul(embed.unsqueeze(1), short_weights.permute(0, 2, 1))
        if self.bias is not None:
            short_bias = F.embedding(shortlist, self.bias, sparse=True, padding_idx=self.padding_idx)
            out = out + short_bias.permute(0, 2, 1)
        return out.squeeze()

    def direct_forward(self, embed):
        return F.linear(embed, self.weight[:-1], self.bias.squeeze()[:-1])


class DenseSparseAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DenseSparseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : ``callable``, optional.
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                if 'exp_avg' not in state:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                if 'exp_avg_sq' not in state:
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']
                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    # Decay the first and second moment running average coefficient
                    #      other <- b * other + (1 - b) * new
                    # <==> other += (1 - b) * (new - other)
                    old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
                    exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
                    exp_avg.add_(make_sparse(exp_avg_update_values))
                    old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
                    exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                    exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))
                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = exp_avg_update_values.add_(old_exp_avg_values)
                    exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                    denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
                    del exp_avg_update_values, exp_avg_sq_update_values
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                    p.data.add_(make_sparse(-step_size * numer.div_(denom)))
                    if weight_decay > 0.0:
                        p.data.add_(p.data.sparse_mask(grad), alpha=-group['lr'] * weight_decay)
                else:
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    if weight_decay > 0.0:
                        p.data.add_(p.data, alpha=-group['lr'] * weight_decay)
        return loss
