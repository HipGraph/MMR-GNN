import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import os
import sys
import math


comparator_fn_map = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b, 
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b, 
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b, 
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b, 
}


def copy_dict(a):
    return {key: value for key, value in a.items()}


def merge_dicts(a, b, overwrite=True):
    if not (isinstance(a, dict) or isinstance(b, dict)):
        raise ValueError("Expected two dictionaries")
    if overwrite:
        return {**a, **b}
    c = copy_dict(a)
    for key, value in b.items():
        if not key in a:
            c[key] = value
    return c


def format_memory(n_bytes):
    if n_bytes == 0:
        return "0B"
    mem_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(n_bytes, 1024)))
    p = math.pow(1024, i)
    s = int(round(n_bytes / p, 2))
    return "%s%s" % (s, mem_name[i])


def memory_of(item):
    if isinstance(item, torch.Tensor):
        n_bytes = item.element_size() * item.nelement()
    else:
        n_bytes = sys.getsizeof(item)
    return format_memory(n_bytes)


def make_msg_block(msg, block_char="#"):
    msg_line = 3*block_char + " " + msg + " " + 3*block_char
    msg_line_len = len(msg_line)
    msg_block = "%s\n%s\n%s" % (
        msg_line_len*block_char,
        msg_line,
        msg_line_len*block_char
    )
    return msg_block


def get_device(use_gpu):
    return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def unravel_index(indices, shape):
    """Converts flat indices into unraveled coordinates in a target shape. This is a `torch` implementation of `numpy.unravel_index`.
    
    Arguments
    ---------
    indices : LongTensor with shape=(?, N)
    shape : tuple with shape=(D,)

    Returns
    -------
    coord : LongTensor with shape=(?, N, D)

    Source
    ------
    author : francois-rozet @ https://github.com/pytorch/pytorch/issues/35674#issuecomment-739492875

    """
    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices
    coord = torch.zeros(indices.size() + shape.size(), dtype=indices.dtype, device=indices.device)
    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = torch.div(indices, dim, rounding_mode="trunc")
        return coord.flip(-1)


def align_dims(x, y, x_dim, y_dim):
    if len(x.shape) > len(y.shape):
        raise ValueError(
            "Input x must have fewer dimensions than y to be aligned. Received x.shape=%s and y.shape=%s" % (
                x.shape, y.shape
            )
        )
    elif len(x.shape) == len(y.shape):
        return x
    if x_dim < 0:
        x_dim = len(x.shape) + x_dim
    if y_dim < 0:
        y_dim = len(y.shape) + y_dim
    new_shape = [1 for _ in y.shape]
    start = y_dim - x_dim
    end = start + len(x.shape)
    new_shape[start:end] = x.shape
    return x.view(new_shape)


def align(inputs, dims):
    if not isinstance(inputs, (tuple, list)) or not all([isinstance(inp, torch.Tensor) for inp in inputs]):
        raise ValueError("Argumet inputs must be tuple or list of tensors")
    if len(inputs) < 2:
        return inputs
    if isinstance(dims, int):
        dims = tuple(dims for inp in inputs)
    elif not isinstance(dim, tuple):
        raise ValueError("Argument dim must be int or tuple of ints")
    input_dims = [inp.dim() for inp in inputs]
    idx = input_dims.index(max(input_dims))
    y = inputs[idx]
    y_dim = dims[idx]
    return [align_dims(inp, y, dim, y_dim) for inp, dim in zip(inputs, dims)]


class __Linear__(torch.nn.Module):

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)


class kLinear(__Linear__):

    debug = 0

    def __init__(self, in_size, out_size, bias=True, k=1):
        super(kLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty((k, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((out_size,)))
        else:
            self.bias = self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.k = k

    def forward(self, x):
        w, b = self.weight, self.bias
        # x.shape=(?, K, |V|, I)
        # w.shape=(K, O, I)
        # b.shape=(O,)
        if self.debug:
            print("x =", x.shape)
            print("w =", w.shape)
            if not b is None:
                print("b =", b.shape)
        x = torch.einsum("...KVI,KOI->...KVO", x, w) # shape=(?, K, |V|, O)
        if not b is None:
            x = x + b # shape=(?, |V|, O)
        return x


class mLinear(__Linear__):

    debug = 0

    def __init__(self, in_size, out_size, bias=True, M=1):
        super(mLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty((M, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((M, out_size)))
        else:
            self.bias = self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.M = M

    def forward(self, x, modality_index):
        w = torch.index_select(self.weight, 0, modality_index)
        b = None if self.bias is None else torch.index_select(self.bias, 0, modality_index)
        # x.shape=(?, |V|, I)
        # modality_index.shape=(?, |V|) E {0, 1, ..., M-1}
        # w.shape=(|V|, O, I)
        # b.shape=(|V|, O)
        if self.debug:
            print("x =", x.shape)
            print("modality_index =", modality_index.shape)
            print("w =", w.shape)
            if not b is None:
                print("b =", b.shape)
        x = torch.einsum("...VI,VOI->...VO", x, w)
        if not b is None:
            x = x + b
        return x


class kmLinear(__Linear__):

    debug = 0

    def __init__(self, in_size, out_size, bias=True, k=1, M=1):
        super(kmLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty((k, M, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((M, out_size)))
        else:
            self.bias = self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.k = k
        self.M = M

    def forward(self, x, modality_index):
        w = torch.index_select(self.weight, 1, modality_index)
        b = None if self.bias is None else torch.index_select(self.bias, 0, modality_index)
        # x.shape=(?, K, |V|, I)
        # modality_index.shape=(?, |V|) E {0, 1, ..., M-1}
        # w.shape=(K, |V|, O, I)
        # b.shape=(|V|, O)
        if self.debug:
            print(make_msg_block("kmLinear Forward"))
            print("x =", x.shape)
            print("w =", w.shape)
            if not b is None:
                print("b =", b.shape)
        x = torch.einsum("...KVI,KVOI->...KVO", x, w) # shape=(?, |V|, O)
        if not b is None:
            x = x + b # shape=(?, |V|, O)
        return x


class vwLinear(__Linear__):

    debug = 0

    def __init__(self, in_size, out_size, bias=True, embed_size=10):
        super(vwLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty((embed_size, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((embed_size, out_size)))
        else:
            self.bias = self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.embed_size = embed_size

    def forward(self, x, embedding):
        w = torch.einsum("...D,DOI->...OI", embedding, self.weight)
        b = None if self.bias is None else torch.einsum("...D,DO->...O", embedding, self.bias)
        # x.shape=(?, |V|, I)
        # embedding.shape=(?, |V|, D)
        # w.shape=(|V|, O, I)
        # b.shape=(|V|, O)
        if self.debug:
            print("x =", x.shape)
            print("embedding =", embedding.shape)
            print("w =", w.shape)
            if not b is None:
                print("b =", b.shape)
        x = torch.einsum("...VI,VOI->...VO", x, w) # shape=(?, |V|, O)
        if not b is None:
            x = x + b # shape=(?, |V|, O)
        return x


class kvwLinear(__Linear__):

    debug = 0

    def __init__(self, in_size, out_size, bias=True, k=1, embed_size=10):
        super(kvwLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty((k, embed_size, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((embed_size, out_size)))
        else:
            self.bias = self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.k = k
        self.embed_size = embed_size

    def forward(self, x, embedding):
        if self.debug:
            print("w =", self.weight.shape)
            if not self.bias is None:
                print("b =", self.bias.shape)
        w = torch.einsum("...VD,KDOI->...KVOI", embedding, self.weight)
        b = None if self.bias is None else torch.einsum("...D,DO->...O", embedding, self.bias)
        # x.shape=(?, K, |V|, I)
        # embedding.shape=(?, |V|, D)
        # w.shape=(K, |V|, O, I)
        # b.shape=(|V|, O)
        if self.debug:
            print("x =", x.shape)
            print("embedding =", embedding.shape)
            print("w =", w.shape)
            if not b is None:
                print("b =", b.shape)
        x = torch.einsum("...KVI,KVOI->...KVO", x, w) # shape=(?, |V|, O)
        if not b is None:
            x = x + b # shape=(?, |V|, O)
        return x


class gcLinear(torch_geometric.nn.conv.MessagePassing):

    debug = 0

    def __init__(self, in_size, out_size, conv="std", layer="Linear", bias=True, order=1, n_hops=1, **kwargs):
        super(gcLinear, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.conv = conv
        self.layer = layer
        self.bias = bias
        self.order = order
        self.n_hops = n_hops
        self.kwargs = kwargs
        self.conv_forward = getattr(self, "%s_conv" % (conv))
        getattr(self, "%s_init" % (layer))(in_size, out_size, bias, order, n_hops, **kwargs)
        self.layer_forward = getattr(self, "%s_forward" % (layer))

    def Linear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if order & 1:
            self.lin_i = torch.nn.Linear(in_size, out_size, bias)
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = torch.nn.Linear(in_size, out_size, bias)

    def kLinear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if order & 1:
            self.lin_i = kLinear(in_size, out_size, bias, n_hops)
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = kLinear(in_size, out_size, bias, n_hops)

    def mLinear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if order & 1:
            self.lin_i = mLinear(in_size, out_size, bias, kwargs["M"])
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = mLinear(in_size, out_size, bias, kwargs["M"])

    def kmLinear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if order & 1:
            self.lin_i = kmLinear(in_size, out_size, bias, n_hops, kwargs["M"])
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = kmLinear(in_size, out_size, bias, n_hops, kwargs["M"])

    def vwLinear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if order & 1:
            self.lin_i = vwLinear(in_size, out_size, bias, kwargs["embed_size"])
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = vwLinear(in_size, out_size, bias, kwargs["embed_size"])

    def kvwLinear_init(self, in_size, out_size, bias, order, n_hops, **kwargs):
        if self.conv == "cheb": n_hops = max(n_hops, 1)
        if order & 1:
            self.lin_i = kvwLinear(in_size, out_size, bias, n_hops, kwargs["embed_size"])
        if order & 2:
            in_size = out_size if order & 1 else in_size
            self.lin_o = kvwLinear(in_size, out_size, bias, n_hops, kwargs["embed_size"])

    def forward(self, x, edge_index, edge_weight=None, frmt="?", **kwargs):
        if self.debug:
            print(make_msg_block("gcLinear Forward"))
            print("x =", x.shape)
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if hasattr(self, "lin_i"):
                self.lin_i.debug = self.debug
            if hasattr(self, "lin_o"):
                self.lin_o.debug = self.debug
        # Start
        x, edge_index, edge_weight = self.forward_prehook(x, edge_index, edge_weight, **kwargs)
        V = x.shape[-2]
        if not edge_index is None:
            if frmt == "?":
                if edge_index.shape[-2] == V and edge_index.shape[-1] == V:
                    frmt = "adj"
                elif edge_index.shape[-2] == 2:
                    frmt = "coo"
                else:
                    raise ValueError(
                        "Cannot determine format of edge_index with shape=%s" % (str(edge_index.shape))
                    )
        #   Apply linear layer before graph conv
        if self.order & 1:
            x = self.layer_forward(self.lin_i, x, **kwargs)
        #    Apply graph conv
        x = self.conv_forward(x, edge_index, edge_weight, frmt, **kwargs)
        if self.order & 2:
            x = self.layer_forward(self.lin_o, x, **kwargs)
        #    Apply linear layer after graph conv
        if self.debug:
            print(make_msg_block("gcLinear Output"))
            print("x =", x.shape)
        #   Cleanup
        x, edge_index, edge_weight = self.forward_hook(x, edge_index, edge_weight, **kwargs)
        return x

    def forward_prehook(self, x, edge_index, edge_weight, **kwargs):
        self.__prefrwd_xshape__ = x.shape
        return x, edge_index, edge_weight

    def forward_hook(self, x, edge_index, edge_weight, **kwargs):
        self.__postfrwd_xshape__ = x.shape
        if self.debug > 1:
            print("prefrwd_xshape =", self.__prefrwd_xshape__)
            print("postfrwd_xshape =", self.__postfrwd_xshape__)
        if x.dim() > len(self.__prefrwd_xshape__):
            if self.conv in ["cheb"]: # may require reduction
                x = torch.sum(x, -3)
            else:
                raise NotImplementedError(self.conv)
        return x, edge_index, edge_weight

    def identity_conv(self, x, edge_index, edge_weight, frmt, **kwargs):
        return x

    def std_conv(self, x, edge_index, edge_weight, frmt, **kwargs):
        if not edge_index is None:
            if frmt == "coo":
                x, edge_index = align((x, edge_index), -1)
                for i in range(self.n_hops):
                    x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
                x = torch.squeeze(x, 0)
            elif frmt == "adj":
                for i in range(self.n_hops):
                    x = torch.einsum("...NM,...MI->...NI", edge_index, x)
        return x

    def cheb_conv(self, x, edge_index, edge_weight, frmt, **kwargs):
        V = x.shape[-2]
        if edge_index is None:
            x = torch.stack([x for _ in range(self.n_hops)], -3)
        elif frmt == "coo":
            raise NotImplementedError(frmt)
        else:
            supports = [torch.eye(V, device=x.device)]
            if self.n_hops > 0:
                supports.append(edge_index)
            for k in range(2, self.n_hops):
                supports.append(torch.matmul(2 * edge_index, supports[-1]) - supports[-2])
            supports = torch.stack(supports, -3)
            x = torch.einsum("...KNM,...MI->...KNI", supports, x)
        return x

    def Linear_forward(self, layer, x, **kwargs):
        return layer(x)

    def kLinear_forward(self, layer, x, **kwargs):
        return layer(x)

    def mLinear_forward(self, layer, x, **kwargs):
        return layer(x, kwargs["modality_index"])

    def kmLinear_forward(self, layer, x, **kwargs):
        return layer(x, kwargs["modality_index"])

    def vwLinear_forward(self, layer, x, **kwargs):
        return layer(x, kwargs["embedding"])

    def kvwLinear_forward(self, layer, x, **kwargs):
        return layer(x, kwargs["embedding"])

    def message(self, x_j, edge_weight):
        """
        Arguments
        ---------
        x_j : FloatTensor with shape=(?, |E|, F)
        edge_weight : (FloatTensor, optional) with shape=(?, |E|)

        Returns
        -------
        x : FloatTensor with shape=(?, |E|, F)

        """
        if edge_weight is None:
            return x_j
        if self.debug:
            print("x_j =", x_j.shape, "=")
            if self.debug > 1:
                print(x_j)
            print("edge_weight =", edge_weight.shape, "=")
            if self.debug > 1:
                print(edge_weight)
        edge_weight = torch.unsqueeze(edge_weight, -1)
        if self.debug:
            print("x_j =", x_j.shape, "=")
            if self.debug > 1:
                print(x_j)
            print("edge_weight =", edge_weight.shape, "=")
            if self.debug > 1:
                print(edge_weight)
        return edge_weight * x_j

    def reset_parameters(self):
        if self.order & 1:
            self.lin_i.reset_parameters()
        if self.order & 2:
            self.lin_o.reset_parameters()


class stGRUCell(torch.nn.Module):

    debug = 0

    def __init__(self, in_size, out_size, xs_size=0, xt_size=0, conv="std", layer="Linear", bias=True, order=1, n_hops=1, shared=True, **kwargs):
        super(stGRUCell, self).__init__()
        print("stGRUCell :", in_size, out_size, xs_size, xt_size, layer, bias, order, n_hops, shared, kwargs)
        # Layers
        if shared:
            self.lin_rz = gcLinear(
                in_size + xs_size + xt_size + out_size, 
                2 * out_size, 
                conv, layer, bias, order, n_hops, **kwargs
            )
            self.lin_n = gcLinear(
                in_size + xs_size + xt_size + out_size, 
                out_size, conv, layer, bias, order, n_hops, **kwargs
            )
        else:
            # layers for x
            self.lin_ir = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_iz = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_in = gcLinear(in_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            # layers for h
            self.lin_hr = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_hz = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            self.lin_hn = gcLinear(out_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            # layers for xs
            if xs_size > 0:
                self.lin_sr = gcLinear(xs_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
                self.lin_sz = gcLinear(xs_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
                self.lin_sn = gcLinear(xs_size, out_size, conv, layer, bias, order, n_hops, **kwargs)
            # layers for xt
            if xt_size > 0:
                self.lin_tr = torch.Linear(xt_size, out_size, bias)
                self.lin_tz = torch.Linear(xt_size, out_size, bias)
                self.lin_tn = torch.Linear(xt_size, out_size, bias)
        # Save vars
        self.in_size = in_size
        self.out_size = out_size
        self.xs_size = xs_size
        self.xt_size = xt_size
        self.shared = shared
        self.order = order

    def forward(self, x, edge_index=None, edge_weight=None, prev_state=None, **kwargs):
        if self.shared:
            self.lin_rz.debug = self.debug
        else:
            self.lin_ir.debug = self.debug
        xs, xt = kwargs.get("xs", None), kwargs.get("xt", None)
        # Setup
        if prev_state is None:
            prev_state = torch.zeros((x.size(0), x.size(-2), self.out_size), dtype=x.dtype, device=x.device)
        h = prev_state
        if self.debug:
            print("x =", x.shape)
            if not xs is None:
                print("xs =", xs.shape)
            if not xt is None:
                print("xt =", xt.shape)
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
            print("prev_state =", h.shape)
        # Start
        if self.shared:
            if (not xs is None and self.xs_size > 0) and (not xt is None and self.xt_size > 0):
                xs = torch.unsqueeze(xs, 0).expand(x.shape[:-1]+(-1,))
                xt = torch.unsqueeze(xt, 1).expand(x.shape[:-1]+(-1,))
                x = torch.cat((xs, xt, x), -1)
            elif not xs is None and self.xs_size > 0:
                xs = torch.unsqueeze(xs, 0).expand(x.shape[:-1]+(-1,))
                x = torch.cat((xs, x), -1)
            elif not xt is None and self.xt_size > 0:
                xt = torch.unsqueeze(xt, 1).expand(x.shape[:-1]+(-1,))
                x = torch.cat((xt, x), -1)
            rz = torch.sigmoid(self.lin_rz(torch.cat((x, h), -1), edge_index, edge_weight, **kwargs))
            r, z = torch.split(rz, self.out_size, -1)
            n = torch.tanh(self.lin_n(torch.cat((x, r * h), -1), edge_index, edge_weight, **kwargs))
        else:
            if (not xs is None and self.xs_size > 0) and (not xt is None and self.xt_size > 0):
                r = torch.sigmoid(
                    self.lin_sr(xs, edge_index, edge_weight, **kwargs) + \
                    self.lin_tr(xt) + \
                    self.lin_ir(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hr(h, edge_index, edge_weight, **kwargs)
                )
                z = torch.sigmoid(
                    self.lin_sz(xs, edge_index, edge_weight, **kwargs) + \
                    self.lin_tz(xt) + \
                    self.lin_iz(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hz(h, edge_index, edge_weight, **kwargs)
                )
                n = torch.tanh(
                    self.lin_sn(xs, edge_index, edge_weight, **kwargs) + \
                    self.lin_tn(xt) + \
                    self.lin_in(x, edge_index, edge_weight, **kwargs) + \
                    r * self.lin_hn(h, edge_index, edge_weight, **kwargs)
                )
            elif not xs is None and self.xs_size > 0:
                r = torch.sigmoid(
                    self.lin_sr(xs, edge_index, edge_weight, **kwargs) + \
                    self.lin_ir(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hr(h, edge_index, edge_weight, **kwargs)
                )
                z = torch.sigmoid(
                    self.lin_sz(xs, edge_index, edge_weight, **kwargs) + \
                    self.lin_iz(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hz(h, edge_index, edge_weight, **kwargs)
                )
                n = torch.tanh(
                    self.lin_in(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_sn(xs, edge_index, edge_weight, **kwargs) + \
                    r * self.lin_hn(h, edge_index, edge_weight, **kwargs)
                )
            elif not xt is None and self.xt_size > 0:
                r = torch.sigmoid(
                    self.lin_tr(xt) + \
                    self.lin_ir(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hr(h, edge_index, edge_weight, **kwargs)
                )
                z = torch.sigmoid(
                    self.lin_tz(xt) + \
                    self.lin_iz(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hz(h, edge_index, edge_weight, **kwargs)
                )
                n = torch.tanh(
                    self.lin_tn(xt) + \
                    self.lin_in(x, edge_index, edge_weight, **kwargs) + \
                    r * self.lin_hn(h, edge_index, edge_weight, **kwargs)
                )
            else:
                r = torch.sigmoid(
                    self.lin_ir(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hr(h, edge_index, edge_weight, **kwargs)
                )
                z = torch.sigmoid(
                    self.lin_iz(x, edge_index, edge_weight, **kwargs) + \
                    self.lin_hz(h, edge_index, edge_weight, **kwargs)
                )
                n = torch.tanh(
                    self.lin_in(x, edge_index, edge_weight, **kwargs) + \
                    r * self.lin_hn(h, edge_index, edge_weight, **kwargs)
                )
        _h = (1 - z) * n + z * h
        if self.debug:
            print(_h.shape)
        return _h


class RNNCell(torch.nn.Module):

    debug = 0

    def __init__(self, in_size, out_size, n_rnn_layers=1, rnn_layer="LSTM", rnn_kwargs={}, dropout=0.0):
        super(RNNCell, self).__init__()
        print("RNNCell :", in_size, out_size, n_rnn_layers, rnn_layer, rnn_kwargs, dropout)
        # Instantiate model layers
        def init_cell(in_size, out_size, rnn_layer, **rnn_kwargs):
            cell = {
                "RNN": torch.nn.RNNCell, 
                "GRU": torch.nn.GRUCell, 
                "LSTM": torch.nn.LSTMCell, 
                "stRNN": None, 
                "stGRU": stGRUCell, 
                "stLSTM": None, 
            }[rnn_layer](in_size, out_size, **rnn_kwargs)
            return cell
        self.cell_forward = getattr(self, "%s_forward" % (rnn_layer))
        self.cells = [init_cell(in_size, out_size, rnn_layer, **rnn_kwargs)]
        self.cells += [init_cell(out_size, out_size, rnn_layer, **rnn_kwargs) for i in range(1, n_rnn_layers)]
        self.drops = [torch.nn.Dropout(dropout) for i in range(n_rnn_layers)]
        self.cells = torch.nn.ModuleList(self.cells)
        self.drops = torch.nn.ModuleList(self.drops)
        # Save all vars
        self.in_size = in_size
        self.out_size = out_size
        self.n_layers = n_rnn_layers
        self.rnn_layer = rnn_layer
        self.rnn_kwargs = rnn_kwargs
        self.dropout = dropout

    def forward(self, **inputs):
        for i in range(self.n_layers):
            self.cells[i].debug = self.debug
        x, xs, xt = inputs["x"], inputs.get("xs", None), inputs.get("xt", None)
        temporal_dim = inputs.get("temporal_dim", -2)
        init_state, n_steps = inputs.get("init_state", None), inputs.get("n_steps", x.shape[temporal_dim])
        if self.debug:
            print(make_msg_block("RNN Forward"))
        if self.debug:
            print("x =", x.shape)
            print(memory_of(x))
            if self.debug > 1:
                print(x)
            if not xs is None:
                print("xs =", xs.shape)
                print(memory_of(xs))
            if not xt is None:
                print("xt =", xt.shape)
                print(memory_of(xt))
        autoregress = False
        if n_steps != x.shape[temporal_dim]: # encode autoregressively
            assert x.shape[temporal_dim] == 1, "Encoding a sequence from %d to %d time-steps is ambiguous" % (
                x.shape[temporal_dim], n_steps
            )
            autoregress = True
        outputs = {}
        get_idx_fn = self.index_select
        A = [None] * n_steps
        a = get_idx_fn(x, 0, temporal_dim)
        xt_t = None
        for i in range(self.n_layers):
            prev_state = init_state
            for t in range(n_steps):
                x_t = (a if autoregress else get_idx_fn(x, t, temporal_dim))
                if not xt is None:
                    xt_t = get_idx_fn(xt, t, temporal_dim)
                inputs_t = merge_dicts(inputs, {"x": x_t, "xt": xt_t, "prev_state": prev_state})
                a, prev_state = self.cell_forward(self.cells[i], **inputs_t)
                if self.debug:
                    print("Step-%d Embedding =" % (t+1), a.shape)
                    print(memory_of(a))
                A[t] = a
            a = torch.stack(A, temporal_dim)
            a = self.drops[i](a)
            x = a
        outputs["yhat"] = a
        outputs["final_state"] = prev_state
        return outputs

    def RNN_forward(self, cell, **inputs):
        hidden_state = cell(inputs["x"], inputs["prev_state"])
        return hidden_state, hidden_state

    def GRU_forward(self, cell, **inputs):
        hidden_state = cell(inputs["x"], inputs["prev_state"])
        return hidden_state, hidden_state

    def LSTM_forward(self, cell, **inputs):
        hidden_state, cell_state = cell(inputs["x"], inputs["prev_state"])
        return hidden_state, (hidden_state, cell_state)

    def stGRU_forward(self, cell, **inputs):
        x, prev_state = inputs["x"], inputs.get("prev_state", None)
        xs, xt = inputs.get("xs", None), inputs.get("xt", None)
        edge_index, edge_weight = inputs.get("edge_index", None), inputs.get("edge_weight", None)
        if self.debug:
            print(make_msg_block("stGRUCell Forward"))
            print("x =", x.shape)
            if not xs is None:
                print("xs =", xs.shape)
            if not xt is None:
                print("xt =", xt.shape)
            if not prev_state is None:
                print("prev_state =", (prev_state[0].shape, prev_state[1].shape))
            if not edge_index is None:
                print("edge_index =", edge_index.shape)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
        if not "embedding" in inputs:
            inputs["embedding"] = inputs.get("xs", None)
        hidden_state = cell(**inputs)
        return hidden_state, hidden_state

    def transpose_select(self, x, idx, dim):
        return torch.transpose(x, 0, dim)[idx,:,:]

    def index_select(self, x, idx, dim):
        return torch.index_select(x, dim, torch.tensor(idx, device=x.device)).squeeze(dim)

    def reset_parameters(self):
        for cell in self.cells:
            if hasattr(cell, "reset_parameters"):
                cell.reset_parameters()


class GraphConstructor(torch.nn.Module):

    debug = 0

    def __init__(self, graph_construction_method=["top-k", "Minkowski-2", 1.0]):
        super(GraphConstructor, self).__init__()
        if graph_construction_method is None:
            pass
        elif graph_construction_method[0] == "random":
            graph_construction_method.insert(None, 1)
        self.method = graph_construction_method

    def forward(self, **inputs):
        if self.debug:
            print(make_msg_block("GraphConstruction Forward"))
        outputs = {}
        if not self.method is None:
            sims = self.compute_similarities(**inputs)["sims"]
            _inputs = merge_dicts(inputs, {"sims": sims})
            outputs = self.sims_to_edges(**_inputs)
        return outputs

    def _compute_similarities(self, A, B):
        if self.debug:
            print("A =", A.shape)
            print(memory_of(A))
            if self.debug > 1:
                print(A)
            print("B =", B.shape)
            print(memory_of(B))
            if self.debug > 1:
                print(B)
        # Handle args
        if len(A.shape) < 2:
            raise NotImplementedError(A.shape)
        if len(B.shape) < 2:
            raise NotImplementedError(A.shape)
        # Setup
        sims = None
        sim_fns = self.method[1]
        if isinstance(sim_fns, str):
            sim_fns = [sim_fns]
        elif not isinstance(sim_fns, list):
            raise ValueError(sim_fns)
        # Start
        if "Minkowski" in sim_fns[0]:
            p = float(sim_fns[0].split("-")[-1])
            if len(A.shape) == 2:
                sims = torch.exp(
                    -torch.cdist(torch.unsqueeze(A, 0), torch.unsqueeze(B, 0), p) / A.shape[-1]
                )
                sims = torch.squeeze(sims, 0)
            else:
                sims = torch.exp(-torch.cdist(A, B, p) / A.shape[-1])
        elif sim_fns[0] == "cosine":
            numer = torch.einsum("...ij,...kj->...ik", A, B)
            norm = torch.sqrt(torch.outer(torch.sum(torch.pow(A, 2), -1), torch.sum(torch.pow(B, 2), -1)))
            # Two options:
            #   1. sims = torch.abs(cosims) : so negative similarity (sims=[-1,0) is NOT discounted
            #   2. sims = torch.clamp(cosims) : so negative similarity is completely discounted
            cosims = numer / torch.max(norm, torch.tensor(1e-8))
        elif sim_fns[0] == "dot":
            sims = torch.einsum("...ij,...kj->...ik", A, B)
        else:
            raise NotImplementedError(sim_fns[0])
        for fn in sim_fns[1:]:
            if fn == "ReLU":
                sims = F.relu(sims)
            elif fn == "Softmax":
                sims = F.softmax(sims, -1)
            else:
                raise NotImplementedError(fn)
        return sims

    def compute_similarities(self, **inputs):
        # Compute a similarity metric between all spatial element pairs
        #   unpack the inputs
        x, m, b = inputs["x"], inputs.get("m", None), inputs.get("b", None)
        reduce_fn, reduce_dim = inputs.get("reduce_fn", torch.mean), inputs.get("reduce_dim", None)
        agg_fn, agg_weight = inputs.get("agg_fn", sum), inputs.get("agg_weight", 1.0)
        outputs = {}
        if not isinstance(x, list):
            x = [x]
        if not isinstance(reduce_fn, list):
            reduce_fn = [reduce_fn for _ in x]
        if not isinstance(reduce_dim, list):
            reduce_dim = [reduce_dim for _ in x]
        if not isinstance(agg_weight, list) and not isinstance(agg_weight, torch.Tensor):
            agg_weight = [agg_weight for _ in x]
        sims = []
        for _x, _reduce_fn, _reduce_dim in zip(x, reduce_fn, reduce_dim):
            sim = self._compute_similarities(_x, _x)
            if not _reduce_dim is None:
                sim = _reduce_fn(sim, _reduce_dim)
            sims.append(sim)
            if self.debug:
                print("Similarity =", sim.shape)
                print(memory_of(sim))
                if self.debug > 1:
                    print(sim)
        sims = align(sims, -1)
        for i in range(len(sims)):
            sims[i] = agg_weight[i] * sims[i]
        sim = agg_fn(sims)
        if not m is None:
            sim = sim * m
        if not b is None:
            sim = sim + b
        if self.debug:
            print("Similarity =", sim.shape)
            print(memory_of(sim))
            if self.debug > 1:
                print(sim)
        outputs["sims"] = sim
        return outputs

    def _sims_to_edges(self, sims):
        n_spatial, n_spatial = sims.shape[-2:]
        edge_index, edge_weight, W = None, None, None
        if self.method[0] =="top-k":
            k = self.method[-1]
            if isinstance(k, float):
                k = int(k * n_spatial**2 + 0.5)
            values, indices = torch.topk(torch.reshape(sims, sims.shape[:-2] + (-1,)), k)
            edge_index = torch.transpose(unravel_index(indices, sims.shape[-2:]), -2, -1)
            edge_weight = values
        elif self.method[0] == "k-nn":
            n_spatial, n_spatial = sims.shape[-2:]
            k = self.method[-1]
            if isinstance(k, float):
                k = int(k * n_spatial**2 + 0.5)
            values, indices = torch.topk(sims, k)
            edge_index = torch.stack(
                [
                    torch.arange(n_spatial, device=sims.device).repeat_interleave(k), 
                    torch.reshape(indices, sims.shape[:-2] + (-1,))
                ]
            )
            edge_weight = torch.reshape(values, sims.shape[:-2] + (-1,))
        elif self.method[0] == "threshold":
            comparator, threshold = self.method[-2:]
            indices = torch.where(comparator_fn_map[comparator](sims, threshold))
            edge_index = torch.stack(indices, 0)
            edge_weight = sims[indices]
        elif self.method[0] == "range":
            lower, upper = self.method[-2:]
            indices = torch.where(torch.logical_and(sims >= lower, sims <= upper))
            edge_index = torch.stack(indices, 0)
            edge_weight = sims[indices]
        elif self.method[0] == "random":
            raise NotImplementedError(self.method)
        else:
            raise NotImplementedError("Unknown edge construction heuristic \"%s\"" % (self.method[0]))
        if edge_index.shape[-1] < n_spatial**2:
            M = torch.zeros((n_spatial, n_spatial), device=sims.device)
            M[...,edge_index[0],edge_index[1]] = 1
            W = M * sims
        else:
            W = sims
        return edge_index, edge_weight, W

    def sims_to_edges(self, **inputs):
        sims = inputs["sims"]
        orig_edge_index = inputs.get("orig_edge_index", None)
        excl_edge_index = inputs.get("excl_edge_index", None)
        ignore_self_loops = inputs.get("ignore_self_loops", True)
        prune = inputs.get("prune", None)
        # Start
        orig_edge_weight = None
        outputs = {}
        if len(sims.shape) == 2: # Construct one set of edges
            n_spatial, n_spatial = sims.shape
            if not orig_edge_index is None: # Save similarities on original edges
                orig_edge_weight = sims[orig_edge_index[0],orig_edge_index[1]]
                if self.debug:
                    print("Original Edge Weight =", orig_edge_weight.shape)
                    print(memory_of(orig_edge_weight))
                    if self.debug > 1:
                        print(orig_edge_weight)
            if ignore_self_loops:
                sims.fill_diagonal_(0) # Ignore self-similarity to avoid self-loops
            if not excl_edge_index is None: # Ignore excluded edge similarities
                excl_edge_weight = sims[excl_edge_index[0],excl_edge_index[1]]
                sims[excl_edge_index[0],excl_edge_index[1]] = 0
            edge_index, edge_weight, W = self._sims_to_edges(sims)
        elif len(sims.shape) == 3: # Construct a set of edges for each time-step
            n_temporal, n_spatial, n_spatial = sims.shape
            if not orig_edge_index is None: # Save similarities on original edges
                orig_edge_weight = sims[:,orig_edge_index[0],orig_edge_index[1]]
                outputs["orig_edge_weight"] = orig_edge_weight
                if self.debug:
                    print("Original Edge Weight =", orig_edge_weight.shape)
                    print(memory_of(orig_edge_weight))
                    if self.debug > 1:
                        print(orig_edge_weight)
            if ignore_self_loops:
                for i in range(n_temporal): # Ignore self-similarity to avoid self-loops
                    sims[i].fill_diagonal_(0)
            if not excl_edge_index is None: # Ignore excluded edge similarities
                excl_edge_weight = sims[:excl_edge_index[0],excl_edge_index[1]]
                sims[:,excl_edge_index[0],excl_edge_index[1]] = 0
            edge_index, edge_weight, W = self._sims_to_edges(sims)
        elif len(sims.shape) == 4: # Construct a set of edges for each time-step
            n_sample, n_temporal, n_spatial, n_spatial = sims.shape
            if not orig_edge_index is None: # Save similarities on original edges
                orig_edge_weight = sims[:,:,orig_edge_index[0],orig_edge_index[1]]
                outputs["orig_edge_weight"] = orig_edge_weight
                if self.debug:
                    print("Original Edge Weight =", orig_edge_weight.shape)
                    print(memory_of(orig_edge_weight))
                    if self.debug > 1:
                        print(orig_edge_weight)
            if ignore_self_loops:
                sims = (1 - torch.eye(n_spatial, device=sims.device)) * sims
            if not excl_edge_index is None: # Ignore excluded edge similarities
                excl_edge_weight = sims[:,:excl_edge_index[0],excl_edge_index[1]]
                sims[:,:,excl_edge_index[0],excl_edge_index[1]] = 0
            edge_index, edge_weight, W = self._sims_to_edges(sims)
        if not prune is None:
            if prune[0] == "weight":
                comparator, value = prune[1:]
                keep = torch.where(torch.logical_not(comparator_fn_map[comparator](edge_weight, value)))
                edge_index = torch.index_select(edge_index, -1, keep[0])
                edge_weight = torch.index_select(edge_weight, -1, keep[0])
            else:
                raise NotImplementedError(prune)
        if self.debug:
            print("Edge Index =", edge_index.shape)
            print(memory_of(edge_index))
            if self.debug > 1:
                print(edge_index)
            print("Edge Weight =", edge_weight.shape)
            print(memory_of(edge_weight))
            if self.debug > 1:
                print(edge_weight)
            if not W is None:
                print("W =", W.shape)
                if self.debug > 1:
                    print(W)
        outputs["edge_index"] = edge_index
        outputs["edge_weight"] = edge_weight
        outputs["orig_edge_weight"] = orig_edge_weight
        outputs["W"] = W
        return outputs

    def reset_parameters(self):
        pass


class GraphAugr(torch.nn.Module):

    debug = 0

    def __init__(self, n_nodes, gc_kwargs={}):
        super(GraphAugr, self).__init__()
        # Init
        self.gc = GraphConstructor(**gc_kwargs)
        self.augment = False
        if not self.gc.method is None:
            if self.gc.method[0] == "threshold" or self.gc.method[-1] > 0:
                self.augment = True
        # Save vars
        self.n_nodes = n_nodes
        self.gc_method = self.gc.method

    def forward(self, **inputs):
        self.gc.debug = self.debug
        edge_index, edge_weight = inputs.get("edge_index", None), inputs.get("edge_weight", None)
        if self.debug:
            print(make_msg_block("GraphAugr Forward"))
        outputs = {}
        if self.augment: # perform augmentation
            if self.debug:
                print(make_msg_block("Augmenting Edges", "-"))
            # Call the GraphConstructor
            gc_outputs = self.gc(**inputs)
            added_edge_index, added_edge_weight = gc_outputs["edge_index"], gc_outputs["edge_weight"]
            if "reduce_dim" in inputs:
                added_edge_index = torch.unsqueeze(added_edge_index, inputs["reduce_dim"])
                added_edge_weight = torch.unsqueeze(added_edge_weight, inputs["reduce_dim"])
            outputs["added_edge_index"] = added_edge_index
            outputs["added_edge_weight"] = added_edge_weight
            # Augment the edges
            if edge_index is None: # no existing edges - simply use constructed edges
                aug_edge_index, aug_edge_weight = added_edge_index, added_edge_weight
                W = gc_outputs.get("W", None)
            else: # existing edges - concatentate constructed edges to them
                aug_edge_index, aug_edge_weight = self.augment_edges(
                    edge_index, added_edge_index, edge_weight, added_edge_weight
                )
                W = torch.zeros((self.n_nodes, self.n_nodes), device=get_device(True))
                if edge_weight is None:
                    W[aug_edge_index[1],aug_edge_index[0]] = 1
                else:
                    W[aug_edge_index[1],aug_edge_index[0]] = aug_edge_weight
            if self.debug:
                print("aug_edge_index =", aug_edge_index.shape)
                if self.debug > 1:
                    print(aug_edge_index)
            if self.debug:
                print("aug_edge_weight =", aug_edge_weight.shape)
                if self.debug > 1:
                    print(aug_edge_weight)
        else: # DO NOT perform augmentation - simply return existing (or non-existing) edges/weights
            if self.debug:
                print(make_msg_block("No Augmentation", "-"))
            aug_edge_index, aug_edge_weight = edge_index, edge_weight
            W = torch.zeros((self.n_nodes, self.n_nodes), device=get_device(True))
            if not edge_index is None:
                if edge_weight is None:
                    W[edge_index[1],edge_index[0]] = 1
                else:
                    W[edge_index[1],edge_index[0]] = edge_weight
        return {"edge_index": aug_edge_index, "edge_weight": aug_edge_weight, "W": W}

    def augment_edges(self, orig_edge_index, added_edge_index, orig_edge_weight=None, added_edge_weight=None):
        if orig_edge_index is None:
            return added_edge_index, added_edge_weight
        if self.debug:
            print("orig_edge_index =", orig_edge_index.shape)
            if self.debug > 1:
                print(orig_edge_index)
        if self.debug:
            print("added_edge_index =", added_edge_index.shape)
            if self.debug > 1:
                print(added_edge_index)
        edge_index = maybe_expand_then_cat((orig_edge_index, added_edge_index), -1)
        if not (orig_edge_weight is None or added_edge_weight is None):
            if 0:
                print("orig_edge_weight =", orig_edge_weight.shape)
                print(orig_edge_weight)
                print("added_edge_weight =", added_edge_weight.shape)
                print(added_edge_weight)
            edge_weight = maybe_expand_then_cat((orig_edge_weight, added_edge_weight), -1)
        else:
            edge_weight = None
        if 0:
            print("edge_index =", edge_index.shape)
            print(edge_index)
            if not edge_weight is None:
                print("edge_weight =", edge_weight.shape)
                print(edge_weight)
        if self.debug:
            print("Adapted Edge Index =", edge_index.shape)
            print(memory_of(edge_index))
            if self.debug > 1:
                print(edge_index)
        if self.debug and not edge_weight is None:
            print("Adapted Edge Weight =", edge_weight.shape)
            print(memory_of(edge_weight))
            if self.debug > 1:
                print(edge_weight)
        return edge_index, edge_weight

    def reset_parameters(self):
        self.gc.reset_parameters()


class TemporalMapper(torch.nn.Module):

    def __init__(self, in_size, out_size, temporal_mapper="last", temporal_mapper_kwargs={}):
        super(TemporalMapper, self).__init__()
        # Setup
        self.supported_methods = ["last", "last_n", "attention"]
        assert temporal_mapper in self.supported_methods, "Temporal mapping method \"%s\" not supported" % (method)
        self.method_init_map = {}
        for method in self.supported_methods:
            self.method_init_map[method] = getattr(self, "%s_init" % (method))
        self.method_mapper_map = {}
        for method in self.supported_methods:
            self.method_mapper_map[method] = getattr(self, "%s_mapper" % (method))
        # Instantiate method
        self.method_init_map[temporal_mapper](in_size, out_size, temporal_mapper_kwargs)
        self.mapper = self.method_mapper_map[temporal_mapper]

    def last_init(self, in_size, out_size, kwargs={}):
        pass

    def last_n_init(self, in_size, out_size, kwargs={}):
        pass

    def attention_init(self, in_size, out_size, kwargs={}):
        attention_kwargs = merge_dicts(
            {"num_heads": 1, "dropout": 0.0, "kdim": in_size, "vdim": in_size},
            kwargs
        )
        self.attn = self.layer_fn_map["MultiheadAttention"](out_size, **attention_kwargs)

    def forward(self, **inputs):
        if self.debug:
            print(make_msg_block("TemporalMapper Forward"))
        return self.mapper(**inputs)

    def last_mapper(self, **inputs):
        if self.debug:
            print(make_msg_block("last_mapper() forward"))
        x, temporal_dim = inputs["x"], inputs.get("temporal_dim", -2)
        if self.debug:
            print("x =", x.shape)
            print("temporal_dim =", temporal_dim)
        return {
            "yhat": torch.index_select(x, temporal_dim, torch.tensor(x.shape[temporal_dim]-1, device=x.device))
        }

    def last_n_mapper(self, **inputs):
        x, n_temporal_out, temporal_dim = inputs["x"], inputs["n_temporal_out"], inputs.get("temporal_dim", -2)
        if self.debug:
            print("x =", x.shape)
            print("temporal_dim =", temporal_dim)
        t = x.shape[temporal_dim]
        return {
            "yhat": torch.index_select(
                x, temporal_dim, torch.tensor(range(t-n_temporal_out, t), device=x.device)
            )
        }

    def attention_mapper(self, **inputs):
        n_temporal_out, temporal_dim = inputs["n_temporal_out"], inputs.get("temporal_dim", -2)
        if "Q" in inputs:
            Q = inputs["Q"]
        else:
            Q = torch.transpose(inputs["x"], 0, temporal_dim)[-n_temporal_out:]
        if "K" in inputs:
            K = inputs["K"]
        else:
            K = torch.transpose(inputs["x"], 0, temporal_dim)
        if "V" in inputs:
            V = inputs["V"]
        else:
            V = torch.transpose(inputs["x"], 0, temporal_dim)
        if self.debug:
            print("Q =", Q.shape)
            print("K =", K.shape)
            print("V =", V.shape)
        a, w = self.attn(Q, K, V)
        return {"yhat": torch.transpose(a, 0, temporal_dim), "attn_weights": w}

    def reset_parameters(self):
        if hasattr(self, "attn"):
            self.attn._reset_parameters()


class MMRGNN(torch.nn.Module):

    debug = 0

    def __init__(
        self,
        Fs, 
        Ft, 
        Fst, 
        N, 
        Fst_out=1, 
        embed_size=10, 
        M=8, 
        H=16, 
        augr_kwargs={}, 
        enc_kwargs={}, 
        mapper_kwargs={}, 
        dec_kwargs={}, 
        out_layer="mLinear", 
    ):
        super(MMRGNN, self).__init__()
        print(
            "MMR-GNN :", 
            Fs, 
            Ft, 
            Fst, 
            N, 
            Fst_out, 
            embed_size, 
            M, 
            H, 
            augr_kwargs, 
            enc_kwargs, 
            mapper_kwargs, 
            dec_kwargs, 
            out_layer, 
        )
        # Instantiate model layers
        if embed_size < 0:
            raise ValueError(embed_size)
        self.node_embs = None
        if embed_size > 0:
            self.node_embs = torch.nn.Parameter(torch.randn(N, embed_size), requires_grad=True)
        #   Graph Augmenation Layer
        self.ga = GraphAugr(N, augr_kwargs)
        #   Encoding layer
        self.enc = RNNCell(Fst, H, **enc_kwargs)
        #   Temporal mapping layer (T_in->?)
        self.tmp_map = TemporalMapper(H, H, **mapper_kwargs)
        #   Temporal decoding layer
        if out_layer != "Conv2d": # Conv2d produces T_out time-steps in this case
            self.dec = RNNCell(H, H, **dec_kwargs)
        #   Output projection layer
        if out_layer == "Linear":
            self.out_proj = torch.nn.Linear(H, Fst_out)
        elif out_layer == "mLinear":
            self.out_proj = mLinear(H, Fst_out, M=M)
        elif out_layer == "Conv2d":
            self.out_proj = torch.nn.Conv2d(
                1, T * Fst_out, kernel_size=(1, H), bias=True
            )
        elif out_layer == "gcLinear":
            self.out_proj = gcLinear(
                H, Fst_out, 
                "cheb", "kmLinear", 
                order=rnn_kwargs["rnn_kwargs"].get("order", 2), 
                n_hops=rnn_kwargs["rnn_kwargs"].get("n_hops", 2), 
                M=kwargs.get("M", 3)
            )
        elif out_layer == "vwLinear":
            self.out_proj = vwLinear(H, Fst_out, embed_size=M)
        else:
            raise ValueError(self.out_layer)
        self.out_proj_act = torch.nn.Identity()
        # Save all vars
        self.Fs = Fs
        self.Ft = Ft
        self.Fst = Fst
        self.Fst_out = Fst_out
        self.N = N
        self.embed_size = embed_size
        self.H = H
        self.augr_kwargs = augr_kwargs
        self.enc_kwargs = enc_kwargs
        self.mapper_kwargs = mapper_kwargs
        self.dec_kwargs = dec_kwargs
        self.out_layer = out_layer

    def forward(self, **inputs):
        self.ga.debug = self.debug
        self.enc.debug = self.debug
        self.tmp_map.debug = self.debug
        self.dec.debug = self.debug
        self.out_proj.debug = self.debug
        xs, xt, xst = inputs.get("xs", None), inputs.get("xt", None), inputs["xst"]
        yt = inputs.get("yt", None)
        edge_index, edge_weight = inputs.get("edge_index", None), inputs.get("edge_weight", None)
        modality_index = inputs.get("modality_index", None)
        T = inputs["T"]
        if self.debug:
            print(make_msg_block("MMR-GNN Forward"))
        if self.debug and not xs is None:
            print("xs =", xs.shape)
            print(memory_of(xs))
            if self.debug > 1:
                print(xs)
        if self.debug and not xt is None:
            print("xt =", xt.shape)
            print(memory_of(xt))
            if self.debug > 1:
                print(xt)
        if self.debug:
            print("xst =", xst.shape)
            print(memory_of(xst))
            if self.debug > 1:
                print(xst)
        if self.debug and not edge_index is None:
            print("edge_index =", edge_index.shape)
            print(memory_of(edge_index))
            if self.debug > 1:
                print(edge_index)
        if self.debug and not edge_weight is None:
            print("edge_weight =", edge_weight.shape)
            print(memory_of(edge_weight))
            if self.debug > 1:
                print(edge_weight)
        if self.debug and not modality_index is None:
            print("modality_index =", modality_index.shape)
            print(memory_of(modality_index))
            if self.debug > 1:
                print(modality_index)
        # START
        n_sample, n_temporal_in, n_spatial, in_size = xst.shape # shape=(N, T, |V|, F)
        outputs = {}
        # GraphAugr layer forward
        ga_outputs = self.ga(
            x=self.node_embs, 
            excl_edge_index=edge_index, 
            orig_edge_index=edge_index, 
            ignore_self_loops=False, 
        )
        W = ga_outputs.get("W", None)
        outputs["added_edge_index"] = ga_outputs.get("added_edge_index", None)
        outputs["added_edge_weight"] = ga_outputs.get("added_edge_weight", None)
        # Encoding layer forward
        enc_outputs = self.enc(
            x=xst, 
            xs=self.node_embs, 
            xt=xt, 
            edge_index=W, 
            edge_weight=edge_weight, 
            modality_index=modality_index, 
            temporal_dim=1, 
        )
        a = enc_outputs["yhat"] # shape=(N, T, |V|, H_0)
        if self.debug:
            print("Temporal Encoding =", a.shape)
        # Temporal mapper layer forward
        a = self.tmp_map(x=a, n_temporal_out=T, temporal_dim=1)["yhat"] # shape=(N, ?, V, H_0)
        if self.debug:
            print("Temporally Remapped Encoding =", a.shape)
        # Decoding layer forward
        if self.out_layer != "Conv2d":
            a = self.dec(
                x=a, 
                xs=self.node_embs, 
                xt=yt, 
                edge_index=W, 
                edge_weight=edge_weight, 
                modality_index=modality_index, 
                n_steps=T, 
                temporal_dim=1, 
            )["yhat"] # shape=(N*|V|, T', H_1)
            if self.debug:
                print("Temporal Decoding =", a.shape)
            # Output projection layer forward
            if self.out_layer == "mLinear":
                z = self.out_proj(a, modality_index) # shape=(N, T', |V|, F')
            elif self.out_layer == "gcLinear":
                z = self.out_proj(a, edge_index=W, modality_index=modality_index) # shape=(N, T', |V|, F')
            elif self.out_layer == "vwLinear": # shape=(N, T', |V|, F')
                z = self.out_proj(a, self.node_embs)
            elif self.out_layer == "Linear":
                z = self.out_proj(a) # shape=(N, T', |V|, F')
            else:
                raise NotImplementedError(self.out_layer)
        else: # Conv2d performs decoding and projection to output dimension
            z = self.out_proj((a)) # shape=(N, T'*C, |V|, 1)
            z = z.squeeze(-1).reshape(-1, T, self.out_size, self.n_nodes)
            z = z.permute(0, 1, 3, 2)  
        a = self.out_proj_act(z)
        if self.debug:
            print("Output Projection =", a.shape)
        if self.debug:
            sys.exit(1)
        outputs["yhat"] = a
        return outputs

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            print(name, parameter.shape)
            torch.nn.init.xavier_uniform_(parameter)
