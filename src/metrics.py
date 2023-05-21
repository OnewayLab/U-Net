import numpy as np
import torch

def compute_miou(pred: torch.Tensor, label: torch.Tensor, num_classes: int, ignore_index: int = None) -> float:
    """计算平均交并比（MIoU）

    Args:
        pred: 预测结果
        label: 真实标签
        num_classes: 类别数
        ignore_index: 要忽略的标签，通常为背景

    Returns:
        float: MIoU 分数
    """
    ious = []

    # 去除要忽略的标签
    mask = label != ignore_index
    pred = pred[mask]
    label = label[mask]

    # 计算每个类别的 IoU，然后取均值
    for i in range(num_classes):
        pred_inds = pred == i
        target_inds = label == i
        intersection = (pred_inds[target_inds]).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        intersection = intersection.cpu().item()
        union = union.cpu().item()
        if union != 0:
            ious.append(intersection / union)
    miou = np.mean(ious)
    return miou