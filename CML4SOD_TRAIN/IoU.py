from torch import Tensor



def iou(input: Tensor, target: Tensor, epsilon: float = 1e-6):

    assert input.size() == target.size()
    assert input.dim() == 3

    sum_dim = (-1, -2)

    input = input.int()
    target = target.int()
    intersection = (input & target).float().sum(dim=sum_dim)  # |A∩B|
    union = (input | target).float().sum(dim=sum_dim)  # |A∪B|

    iou = (intersection + epsilon) / (union + epsilon)

    smooth=1.
    dice = (2. * intersection + smooth) / (intersection+union+ smooth)

    # return iou.mean(), dice.mean()
    return iou, dice


