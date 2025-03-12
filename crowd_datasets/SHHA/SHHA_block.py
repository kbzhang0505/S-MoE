import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torch.nn.functional as F
import glob
import scipy.io as io


class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "shanghai_tech_part_a_train.list"
        self.eval_list = "shanghai_tech_part_a_test.list"
        self.img_list_file = self.train_lists.split(',') if train else self.eval_list.split(',')
        self.img_map = {}
        self.img_list = []

        # 加载图片-标注路径映射
        for train_list in self.img_list_file:
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                        os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        self.nSamples = len(self.img_list)
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]

        # 加载数据
        img, points = self.load_data((img_path, gt_path))

        # 数据预处理
        if self.transform:
            img = self.transform(img)  # [C,H,W]
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # 数据增强
        if self.train:
            # 随机缩放
            if random.random() < 0.5:
                scale = random.uniform(0.7, 1.3)
                if scale * min(img.shape[1:]) > 128:
                    img = F.interpolate(img.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
                    points = points * scale

            # 随机裁剪生成多patch
            if self.patch:
                img_patches, points_patches = self.random_crop(img, points)
            else:
                img_patches = img.unsqueeze(0)  # [1,C,H,W]
                points_patches = [points]

            # 随机翻转
            if self.flip and random.random() > 0.5:
                img_patches = img_patches.flip(-1)
                for pts in points_patches:
                    pts[:, 0] = img_patches.shape[-1] - pts[:, 0]
        else:
            img_patches = img.unsqueeze(0)
            points_patches = [points]

        # 生成密度图和多目标字典
        targets = []
        for i, patch_points in enumerate(points_patches):
            # 关键修复：添加括号明确条件表达式
            h, w = (img_patches.shape[2], img_patches.shape[3]) if self.patch else (img.shape[1], img.shape[2])
            density_map = self.generate_density_map(patch_points, (h, w))

            # 构建target字典
            target = {
                'point': torch.from_numpy(patch_points).float(),
                'image_id': torch.tensor([int(os.path.basename(img_path).split('_')[-1].split('.')[0])]),
                'labels': torch.ones(len(patch_points), dtype=torch.long) if len(patch_points) > 0
                else torch.zeros(0, dtype=torch.long),
                'density_map': torch.from_numpy(density_map).float().unsqueeze(0)
            }
            targets.append(target)

        return img_patches if self.patch else img, targets

    def load_data(self, img_gt_path):
        """ 加载图像和点标注，保持原始逻辑 """
        img_path, gt_path = img_gt_path
        img = Image.open(img_path).convert('RGB')
        points = []
        with open(gt_path) as f:
            for line in f:
                x, y = map(float, line.strip().split())
                points.append([x, y])
        return img, np.array(points)

    def random_crop(self, img, points, num_patch=4):
        """ 随机裁剪生成多patch，保持点标签完整性 """
        c, h, w = img.shape
        crop_imgs = []
        crop_points = []

        for _ in range(num_patch):
            # 随机裁剪坐标
            start_h = random.randint(0, h - 128)
            start_w = random.randint(0, w - 128)
            end_h = start_h + 128
            end_w = start_w + 128

            # 裁剪图像
            crop_img = img[:, start_h:end_h, start_w:end_w]

            # 筛选并调整点坐标
            valid = (points[:, 0] >= start_w) & (points[:, 0] < end_w) & \
                    (points[:, 1] >= start_h) & (points[:, 1] < end_h)
            patch_points = points[valid].copy()
            patch_points[:, 0] -= start_w
            patch_points[:, 1] -= start_h

            crop_imgs.append(crop_img)
            crop_points.append(patch_points)

        return torch.stack(crop_imgs), crop_points

    def generate_density_map(self, points, img_size, sigma=4):
        """ 生成高斯密度图 """
        h, w = img_size
        density = np.zeros((h, w), dtype=np.float32)

        # 过滤越界点
        valid = (points[:, 0] >= 0) & (points[:, 0] < w) & \
                (points[:, 1] >= 0) & (points[:, 1] < h)
        points = points[valid].astype(int)

        # 生成二值图
        for x, y in points:
            if y < h and x < w:
                density[y, x] += 1

        # 高斯平滑
        kernel_size = int(3 * sigma)  # 使用3*sigma的大小作为卷积核
        if kernel_size % 2 == 0:
            kernel_size += 1  # 保证卷积核大小是奇数

        density = cv2.GaussianBlur(density, (kernel_size, kernel_size), sigma)
        original_sum = density.sum()
        density_resized = cv2.resize(density, (density.shape[1] // 8, density.shape[0] // 8), interpolation=cv2.INTER_CUBIC)
        # 通过调整缩放后的图像像素值来保持总和不变
        scaling_factor = original_sum / density_resized.sum()  # 计算缩放因子
        density_resized *= scaling_factor  # 调整密度图的像素值，以恢复总和
        return density_resized


# 测试代码
if __name__ == '__main__':
    dataset = SHHA(data_root='/path/to/data', train=True, patch=True)
    img, targets = dataset[0]
    print("Image shape:", img.shape)  # [4,3,128,128]
    print("Number of targets:", len(targets))  # 4
    print("Keys in target:", targets[0].keys())  # ['point', 'image_id', 'labels', 'density_map']
    print("Density map shape:", targets[0]['density_map'].shape)  # [1,128,128]