import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import time
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from .tester import inference
from .metrics import compute_miou


def train(
    model,
    device,
    train_data,
    val_data,
    input_size,
    output_size,
    batch_size,
    total_steps,
    eval_steps,
    patience,
    learning_rate,
    optimizer,
    lr_scheduler,
    output_path,
):
    """训练模型

    Args:
        model: 模型
        device: 训练设备
        train_data: 训练集
        val_data: 验证集
        input_size: 输入大小
        output_size: 输出大小
        batch_size: 批大小
        total_steps: 总训练步数
        eval_steps: 评估步数，每 eval_steps 步评估一次模型
        patience: 连续 patience 次评估结果不提升时停止训练
        learning_rate: 学习率
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        output_path: 输出路径
    """
    os.makedirs(output_path, exist_ok=True)
    logger = logging.getLogger("train")
    handler = logging.FileHandler(os.path.join(output_path, "train.log"), "w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(
        f"Batch size: {batch_size}, Total steps: {total_steps}, Evaluation steps: {eval_steps}, Patience: {patience}, "
        f"Learning rate: {learning_rate}, Optimizer: {optimizer}, LR scheduler: {lr_scheduler}, "
    )
    writer = SummaryWriter(os.path.join(output_path, "tensorboard"), filename_suffix="_train")

    # 定义数据加载器
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)

    # 选择优化器
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "Momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("optimizer must be one of SGD, Momentum or AdamW")

    # 选择学习率调度器
    if lr_scheduler == "LinearLR":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1 - step / total_steps)
    elif lr_scheduler == "OneCycleLR":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)
    elif lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0, last_epoch=-1)
    elif lr_scheduler == "FixedLR":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1)
    else:
        raise ValueError("lr_scheduler must be one of Linear, OneCycleLR, CosineAnnealingLR or FixedLR")

    # 定义损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    model.to(device)
    start_time = time.time()
    train_loss_list = []
    val_loss_list = []
    val_miou_list = []
    best_val_miou = 0
    patience_count = 0
    logger.info("Start Training")
    train_data_iter = iter(train_dataloader)
    for step in range(0, total_steps, eval_steps):
        logger.info(f"Step {step}/{total_steps}")
        # 训练
        model.train()
        train_loss = 0
        for _ in tqdm(range(eval_steps), desc="Training"):
            try:
                inputs, labels = next(train_data_iter)
            except StopIteration:
                train_data_iter = iter(train_dataloader)
                inputs, labels = next(train_data_iter)
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs)
            labels = transforms.CenterCrop(outputs.shape[-2])(labels)
            loss = loss_fn(outputs, labels)
            train_loss += loss.item() * len(inputs)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        train_loss /= len(train_data)
        train_loss_list.append(train_loss)
        # 验证
        model.eval()
        val_loss = 0
        val_miou = 0
        for input, label in tqdm(val_data, desc="Validation"):
            input = torch.tensor(input, device=device)
            label = torch.tensor(label, device=device)
            prediction, loss = inference(model, device, input, input_size, output_size, loss_fn, label)
            val_loss += loss
            prediction = prediction
            val_miou += compute_miou(prediction, label, val_data.num_classes, ignore_index=0)
        val_loss /= len(val_data)
        val_loss_list.append(val_loss)
        val_miou /= len(val_data)
        val_miou_list.append(val_miou)
        writer.add_scalars("loss", {"train_loss": train_loss, "val_loss": val_loss}, step)
        writer.add_scalar("val_miou", val_miou, step)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], step)
        logger.info(f"\tTraining loss: {train_loss:.4f}")
        logger.info(f"\tValidation loss: {val_loss:.4f}")
        logger.info(f"\tValidation mean IoU: {val_miou:.4f}")
        logger.info(f"\tLearning rate: {optimizer.param_groups[0]['lr']}")
        # 保存最好的模型
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), os.path.join(output_path, "best_model.pt"))
            patience_count = 0
            logger.info("\tBest model saved!")
        else:
            patience_count += 1
            if patience_count == patience:
                logger.info("\tEarly stopping!")
                break
    logger.info(f"Training finished in {time.time() - start_time}s")

    # 绘制损失曲线
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # 设置横坐标为整数
    x_axis = [i * eval_steps for i in range(len(train_loss_list))]
    plt.plot(x_axis, train_loss_list, label="Training loss")
    plt.plot(x_axis, val_loss_list, label="Validation loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_path, "loss.png"))

    writer.close()