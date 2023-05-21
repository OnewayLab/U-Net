import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SemanticSegmentationDataset(Dataset):
    """语义分割数据集"""

    def __init__(self, root, split="train", transform=None, return_original=False):
        """初始化数据集

        Args:
            root: 数据集路径
            split: 训练集 "train"，验证集 "val"，测试集 "test"
            transform: 图像的预处理
            return_original: 是否返回原始图像
        """
        self.root = root
        self.transform = transform
        self.return_original = return_original

        # 读取数据列表
        if split == "train":
            sample_list_path = os.path.join(self.root, "train.txt")
        elif split == "val":
            sample_list_path = os.path.join(self.root, "val.txt")
        elif split == "test":
            sample_list_path = os.path.join(self.root, "test.txt")
        else:
            raise ValueError(f"Invalid split: {split}")
        self.sample_list = []
        with open(sample_list_path) as f:
            for line in f.readlines():
                line = line.strip()
                self.sample_list.append(line)

        # 读取标签名
        self.label_names = []
        with open(os.path.join(self.root, "labels.txt")) as f:
            for line in f.readlines():
                line = line.strip()
                self.label_names.append(line)
        self.num_classes = len(self.label_names)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_id = self.sample_list[idx]
        image = Image.open(os.path.join(self.root, "images", f"{sample_id}.jpg"))
        mask = Image.open(os.path.join(self.root, "masks", f"{sample_id}.png"))
        image = np.array(image)
        mask = np.array(mask)
        origin_image = image
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        if self.return_original:
            origin_image = origin_image.transpose(2, 0, 1)  # HWC -> CHW
            return image, mask, origin_image
        else:
            return image, mask
