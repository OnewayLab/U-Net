import os
from math import ceil
import torch
from torch.nn import functional as F
from torch.utils import tensorboard
import numpy as np
import logging
from tqdm import tqdm
from .metrics import compute_miou
from .visualization import visualize_result


def inference(model, device, image, input_size, output_size, loss_fn=None, target=None):
    """推理

    Args:
        model: 模型
        device: 设备
        inputs: 输入图像，shape 为 (3, H, W) 的张量
        input_size: 输入大小
        output_size: 输出大小
        loss_fn: 损失函数，当 loss_fn 和 target 都不是 None 时计算损失
        target: 目标图像，shape 为 (H, W) 的张量

    Returns:
        分割掩码
    """
    # 计算四周填充大小
    height, width = image.shape[-2:]    # 获取图像的高和宽
    height_patches, width_patches = ceil(height / output_size), ceil(width / output_size)   # 获取切分后的小块的个数
    min_pad = (input_size - output_size) // 2   # 为保证最终输出等于输入大小，上下左右最小填充
    bottom_pad = output_size * height_patches - height + min_pad  # 下方实际需要的填充
    right_pad = output_size * width_patches - width + min_pad  # 右侧实际需要的填充

    # 镜像填充图像
    try:
        image = F.pad(image, (min_pad, right_pad, min_pad, bottom_pad), mode="reflect")
    except AssertionError:  # Pytorch 版本为 1.8.1 时，要求对四周进行填充时 Tensor 维度必须为 4
        image = image.unsqueeze(dim=0)
        image = F.pad(image, (min_pad, right_pad, min_pad, bottom_pad), mode="reflect")
        image = image.squeeze(dim=0)

    # 把 image 切分为 input_size 的小块
    patches = image.unfold(1, input_size, output_size).unfold(2, input_size, output_size)   # 注意“叠瓦”，步长是 output_size
    patches = patches.permute(1, 2, 0, 3, 4)
    height_patches, width_patches = patches.shape[:2]   # 获取切分后的小块的个数
    patches = patches.reshape(-1, 3, input_size, input_size)
    if target is not None:  # 对目标图像进行切片，方便计算损失
        target = target.unsqueeze(dim=0).float()   # 增加一个维度并且转换为浮点型，否则会报错
        try:
            target = F.pad(target, (0, right_pad, 0, bottom_pad), mode="reflect")
        except AssertionError:  # 原因同上...
            target = target.unsqueeze(dim=0)
            target = F.pad(target, (0, right_pad, 0, bottom_pad), mode="reflect")
            target = target.squeeze()
        target = target.squeeze().long()  # 去掉增加的维度
        target = target.unfold(0, output_size, output_size).unfold(1, output_size, output_size)
        target = target.reshape(-1, output_size, output_size)

    # 推理
    loss = 0
    with torch.no_grad():
        patches = patches.to(device)
        output = model(patches)
        if loss_fn and target is not None:
            target = target.to(device)
            loss += loss_fn(output, target).item()
        prediction = output.argmax(dim=1)

    # 把 prediction 拼接回去
    prediction = prediction.reshape(height_patches, width_patches, output_size, output_size)
    prediction = torch.cat(torch.split(prediction, 1, dim=0), dim=2)
    prediction = torch.cat(torch.split(prediction, 1, dim=1), dim=3)
    prediction = prediction.reshape(height_patches * output_size, width_patches * output_size)

    # 把多余部分切除
    prediction = prediction[0:height, 0:width]
    if loss_fn and target is not None:
        return prediction, loss
    else:
        return prediction


def test(model, device, test_data, input_size, output_size, output_path):
    """测试模型

    Args:
        model: 模型
        device: 设备
        test_data: 测试集
        input_size: 输入大小
        output_size: 输出大小
        output_path: 输出路径
    """
    os.makedirs(output_path, exist_ok=True)
    logger = logging.getLogger("test")
    handler = logging.FileHandler(os.path.join(output_path, "test.log"), "w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    writer = tensorboard.SummaryWriter(
        os.path.join(output_path, "tensorboard"), filename_suffix="_test"
    )
    logger.info(f"Test set size: {len(test_data)}")

    # 加载最好的模型
    model.to(device)
    model.load_state_dict(
        torch.load(os.path.join(output_path, "best_model.pt"), map_location=device)
    )
    logger.info("Best model loaded!")

    # 测试
    logger.info("Start Testing")
    model.eval()
    miou = 0
    for i, (input, label, image) in enumerate(tqdm(test_data)):
        input = torch.tensor(input, device=device)
        prediction = inference(model, device, input, input_size, output_size)
        miou += compute_miou(prediction, label, test_data.num_classes, ignore_index=0)
        # 保存前 20 个预测结果
        if i < 20:
            visual_prediction = visualize_result(image, prediction.cpu().numpy())
            visual_label = visualize_result(image, label)
            writer.add_image(f"test/{i}/prediction", visual_prediction, 0)
            writer.add_image(f"test/{i}/ground_truth", visual_label, 0)
    test_miou = miou / len(test_data)
    logger.info(f"Test set mean IoU: {test_miou:.4f}")
    writer.add_scalar("test/miou", test_miou, 0)
    writer.close()
