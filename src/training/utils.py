import random
import torch
import torch.nn.functional as F


def repeat_tensor(tensor):
    tensor = tensor.unsqueeze(0)  # add channel dimension
    return tensor.repeat(3, 1, 1)  # repeat along channel dimension to get 3 channels


def truncate_or_pad(tensor, max_length: int):
    _, w = tensor.shape

    if w > max_length:
        return tensor[:, :max_length]
    elif w < max_length:
        padding = (0, max_length - w)
        return F.pad(tensor, padding, "constant", 0)  # pad with zeros 
    else:
        return tensor


def get_random_chunk(tensor, chunk_length: int):
    _, w = tensor.shape
    upper_idx = w - chunk_length
    
    if upper_idx < 0:
        return truncate_or_pad(tensor, max_length=chunk_length)  # this will pad tensor to `chunk_length`
    else:
        chunk_start_idx = random.randint(0, upper_idx)
        return tensor[:, chunk_start_idx:chunk_start_idx+chunk_length]
