import torch
from torch import Tensor


def dice_coeff(tensor: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert tensor.size() == target.size()
    assert tensor.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if tensor.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (tensor * target).sum(dim=sum_dim)
    sets_sum = tensor.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(tensor: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(tensor.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(tensor: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(tensor, target, reduce_batch_first=True)
