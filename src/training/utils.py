import random
import torch
import torch.nn.functional as F


def repeat_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.unsqueeze(0)  # add channel dimension
    return tensor.repeat(3, 1, 1)  # repeat along channel dimension to get 3 channels


def truncate_or_pad(tensor: torch.Tensor, max_length: int) -> torch.Tensor:
    _, w = tensor.shape

    if w > max_length:
        return tensor[:, :max_length]
    if w < max_length:
        padding = (0, max_length - w)
        return F.pad(tensor, padding, "constant", 0)  # pad with zeros
    return tensor


def get_random_chunk(tensor: torch.Tensor, chunk_length: int) -> torch.Tensor:
    _, w = tensor.shape
    upper_idx = w - chunk_length

    if upper_idx < 0:
        # this will pad tensor to `chunk_length`
        return truncate_or_pad(tensor, max_length=chunk_length)

    chunk_start_idx = random.randint(0, upper_idx)
    return tensor[:, chunk_start_idx:chunk_start_idx+chunk_length]
