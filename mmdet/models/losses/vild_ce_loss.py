# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


@mmcv.jit(derivate=True, coderize=True)
def knowledge_distillation_ce_loss(pred,
                                   label,
                                   T,
                                   reduction='mean',
                                   avg_factor=None,
                                   weight=None):
    r"""Loss function for knowledge distilling using cross entropy.

    Args:
        pred (Tensor): Predicted logits with shape (N, C), C is the number
            of classes.
        label (Tensor): The learning label of the prediction.
        T (int): Temperature for distillation.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        weight (Tensor, optional): The weight of loss for each prediction. 
            Defaults to None.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    kd_loss = F.cross_entropy(
        F.softmax(pred / T, dim=-1),
        label,
        reduction='none',
        ignore_index=-100)
    
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    kd_loss = weight_reduce_loss(
        kd_loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return kd_loss


@LOSSES.register_module()
class ViLDCrossEntropyLoss(nn.Module):
    """Loss function for vild-text using cross entropy.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.

        use_sigmoid, use_mask, class_weight, ignore_index: Not used.
            Just for compatibility with CrossEntropyLoss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=100,
                 use_sigmoid=False,
                 use_mask=False,
                 class_weight=None,
                 ignore_index=False):
        super(ViLDCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_ce_loss(
            pred,
            soft_label,
            T=self.T,
            reduction=reduction,
            avg_factor=avg_factor,
            weight=weight)

        return loss_kd
