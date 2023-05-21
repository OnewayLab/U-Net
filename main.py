import numpy as np
import os
import argparse
import logging
import random
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from src.datasets import SemanticSegmentationDataset
from src.trainer import train
from src.tester import test
from src.unet import UNet


# 训练设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action='store_true', default=False, help="Test only")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("-d", "--data_path", type=str, default="./data/sidewalk-semantic", help="Data path")  # 完整数据集
    parser.add_argument("-is", "--input_size", type=int, default=724, help="Size of model input")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-ts", "--total_steps", type=int, default=4000, help="Total training steps")
    parser.add_argument("-es", "--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("-p", "--patience", type=int, default=8, help="Patience")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("-op", "--output_path", type=str, default="./output", help="Output path")

    args = parser.parse_args()
    return args


def main(args):
    # 设置日志格式
    os.makedirs(args.output_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_path, "main.log"),
        filemode="w",
        level=logging.INFO,
        format="%(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 数据增强与预处理
    MEAN = np.array([132.77355813, 127.61624462, 120.28149316]) / 255
    STD = np.array([58.87826301, 56.10282104, 59.22393147]) / 255
    train_transform = A.Compose([
        A.RandomResizedCrop(width=args.input_size, height=args.input_size, scale=(0.5, 2.0), ratio=(0.75, 1.3333333333333333)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=MEAN, std=STD),
    ])
    test_transform = A.Compose([
        A.Normalize(mean=MEAN, std=STD),
    ])

    # 加载数据集
    logging.info("Loading Data")
    train_dataset = SemanticSegmentationDataset(args.data_path, "train", train_transform)
    val_dataset = SemanticSegmentationDataset(args.data_path, "val", test_transform)
    test_dataset = SemanticSegmentationDataset(args.data_path, "test", test_transform, return_original=True)
    logging.info(
        f"\tTraining dataset size: {len(train_dataset)}\n"
        f"\tValidation dataset size: {len(val_dataset)}\n"
        f"\tTest dataset size: {len(test_dataset)}\n"
        f"\tNumber of classes: {train_dataset.num_classes}"
    )

    # 定义模型并绘制模型结构
    model = UNet(n_channels=3, n_classes=train_dataset.num_classes)
    dummy_input = torch.rand(1, 3, args.input_size, args.input_size)
    output = model(dummy_input)
    output_size = output.shape[-1]
    writer = SummaryWriter(os.path.join(args.output_path, "tensorboard"))
    writer.add_graph(model, dummy_input)
    writer.close()

    # 训练模型
    if not args.test:
        train(
            model,
            DEVICE,
            train_dataset,
            val_dataset,
            args.input_size,
            output_size,
            args.batch_size,
            args.total_steps,
            args.eval_steps,
            args.patience,
            args.learning_rate,
            optimizer="AdamW",
            lr_scheduler="OneCycleLR",
            output_path=args.output_path,
        )

    # 测试模型
    test(
        model,
        DEVICE,
        test_dataset,
        args.input_size,
        output_size,
        args.output_path,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
