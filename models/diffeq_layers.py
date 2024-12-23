import torch
import torch.nn as nn
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)


class IgnoreLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_hot):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, context, x):
        return self._layer(x)


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_hot):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1 + dim_c, dim_out)

    def forward(self, context, x):
        if x.dim() == 3:
            context = context.unsqueeze(1).expand(-1, x.size(1), -1)
        x_context = torch.cat((x, context), dim=2)
        return self._layer(x_context)


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_hot):
        super(ConcatLinear_v2, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)

    def forward(self, context, x):
        bias = self._hyper_bias(context)
        if x.dim() == 3:
            bias = bias.unsqueeze(1)
        return self._layer(x) + bias


class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_hot):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = torch.sigmoid(self._hyper(context))
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
        return self._layer(x) * gate


class ScaleLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_hot):
        super(ScaleLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = self._hyper(context)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
        return self._layer(x) * gate


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_hot):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = torch.sigmoid(self._hyper_gate(context))
        bias = self._hyper_bias(context)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class ConcatScaleLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_hot):
        super(ConcatScaleLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = self._hyper_gate(context)
        bias = self._hyper_bias(context)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret

class ConcatHotSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_hot):
        super(ConcatHotSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + (dim_c - dim_hot), dim_out, bias=False)
        self._hyper_gate = nn.Linear(1 + dim_hot, dim_out) #pair class

        self.dim_c = dim_c
        self.dim_hot = dim_hot

    def forward(self, context, x):
        hot_idx = np.concatenate(([0], np.arange(self.dim_c-self.dim_hot, self.dim_c)))
        gate = torch.sigmoid(self._hyper_gate(context[:, hot_idx]))
        bias = self._hyper_bias(context[:, :(1+self.dim_c-self.dim_hot)])
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret

class ConcatHotScaleLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, dim_hot):
        super(ConcatHotScaleLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + (dim_c - dim_hot), dim_out, bias=False)
        self._hyper_gate = nn.Linear(1 + dim_hot, dim_out)

        self.dim_c = dim_c
        self.dim_hot = dim_hot

    def forward(self, context, x):
        hot_idx = np.concatenate(([0], np.arange(self.dim_c-self.dim_hot, self.dim_c)))
        gate = self._hyper_gate(context[:, hot_idx])
        bias = self._hyper_bias(context[:, :(1+self.dim_c-self.dim_hot)])
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret