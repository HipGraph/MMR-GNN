import torch
import math
import numpy as np


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


def msg_block(msg, block_char="#"):
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


def n_sliding_windows(timesteps, length, stride=1, offset=0):
    return (timesteps - length + stride - offset) // stride


def sliding_window_indices(timesteps, length, stride=1, offset=0):
    """ Creates indices that index a set of windows

    Arguments
    ---------
    timesteps: int 
        total time-steps in source data
    length: int 
        time-steps per window
    stride: int
        time-steps between consecutive windows (typically 1)
    offset: int
        time-step where first window begins

    """
    n = n_sliding_windows(timesteps, length, stride, offset)
    return np.tile(np.arange(length), (n, 1)) + stride * np.reshape(np.arange(n), (-1, 1)) + offset


def input_output_window_indices(timesteps, in_length, out_length, horizon=1, stride=1, offset=0):
    """ Creates indices that index a set of input and output windows

    Arguments
    ---------
    timesteps: int
        number of time-steps in source
    in_length: int
        number of time-steps per input window
    out_length: int
        number of time-steps per output window
    horizon: int
        offset (in time-steps) of output windows relative to last time-step of each input window
        see notes below
    stride: int
        number of time-steps between window origins
    offset: int
        time-step where first window begins

    Notes
    -----
    horizon :
        at horizon=1  => input/output window indices are [0, 1, 2]/[3, 4, 5]
        at horizon=3  => input/output window indices are [0, 1, 2]/[5, 6, 7]
        at horizon=-2 => input/output window indices are [0, 1, 2]/[0, 1, 2]

    """
    indices = sliding_window_indices(timesteps, in_length + (horizon - 1) + out_length, stride, offset)
    return indices[:,:in_length], indices[:,-out_length:]
