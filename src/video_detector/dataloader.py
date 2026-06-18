import torch
import os
import numpy as np
import torchaudio
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as F
import PIL
import csv
import random
import io
import sys

import random
import cv2
#cv2.setNumThreads(0)  # 禁用 cv2 内部多线程，避免与 PyTorch 冲突
#cv2.setUseOptimized(False)  # 可选：禁用 OpenCV 优化，进一步减少线程使用


import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt



import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# 微调数据集dataset
# 输入: csv文件，包含两列：图片路径，标签
# 处理: 读取图片，进行数据增强，返回图片张量和标签
# 输出: 图片张量 [3, H, W]，标签张量 [1]
class FineTuneDataset(Dataset):
    def __init__(self, csv_file, data_augment=True, mean = [0.485, 0.456, 0.406],std  = [0.229, 0.224, 0.225], if_normalize=True, img_size=512):
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)   # 跳过 header
            for row in reader:
                self.data.append((row[0], int(row[1])))  # (image_path, label)

        self.data_augment = data_augment
        print(f"now using data augmentation: {self.data_augment}")

        self.if_normalize = if_normalize
        print(f"now using normalize: {self.if_normalize}")

        self.img_size = img_size

        self.transform = ImageAugment(im_res=self.img_size, visual_augment=True, if_normalize=if_normalize, mean=mean, std=std)
        self.transform_origin = ImageTransform(im_res=self.img_size, visual_augment=self.data_augment, if_normalize=if_normalize, mean=mean, std=std)

        self.random_crop = transforms.RandomResizedCrop(
            size=512,
            scale=(0.6, 1.0),      # 裁剪面积比例
            ratio=(0.75, 1.33)     # 宽高比
        )

    def __len__(self):
        return len(self.data)

    def get_real_fake_ratio(self):
        real = 0
        fake = 0

        for _, label in self.data:
            if label == 0:
                real += 1
            else:
                fake += 1

        if fake == 0:
            return 1.0

        return real / fake



    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print("bad image:", img_path)

            # 随机换一张
            new_idx = torch.randint(0, len(self.data), (1,)).item()
            return self.__getitem__(new_idx)

        # 随机比例裁剪 IJICAI only
        # img = self.random_crop(img)

        degraded_img = self.transform(img) # tensor
        img_origin = self.transform_origin(img) # tensor
        label = torch.tensor(label, dtype=torch.float32) # for BCE

        return degraded_img, img_origin, label
    def collate_fn(self, batch):
        """
        batch: list of (degraded_img, img_origin, label)
        """
        degraded_imgs, origin_imgs, labels = zip(*batch)

        patch_size = 16
        min_size = 384
        max_size = 768

        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],std  = [0.229, 0.224, 0.225])

        # ===== 1. batch-level resolution =====
        def sample_size():
            return random.randint(min_size // patch_size, max_size // patch_size) * patch_size

        H = sample_size()
        W = sample_size()

        # ===== 2. batch-level interpolation =====
        interp = random.choice([
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4
        ])

        # ===== 3. 只 resize degraded_img =====
        degraded_out = []
        for img in degraded_imgs:
            img = cv2.resize(img, (W, H), interpolation=interp)

            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            img=normalize(img)

            degraded_out.append(img)


        degraded_out = torch.stack(degraded_out)

        # ===== 4. origin 不变（只做 tensor stack）=====
        origin_out = torch.stack(origin_imgs)

        # ===== 5. labels 不变（只做 tensor stack）=====
        labels = torch.stack(labels)
        return degraded_out, origin_out, labels


class FineTuneDataset_no_pair(Dataset):
    def __init__(self, csv_file, data_augment=True, mean = [0.485, 0.456, 0.406],std  = [0.229, 0.224, 0.225], if_normalize=True, img_size=512):
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)   # 跳过 header
            for row in reader:
                self.data.append((row[0], int(row[1])))  # (image_path, label)

        self.data_augment = data_augment
        print(f"now using data augmentation: {self.data_augment}")

        self.if_normalize = if_normalize
        print(f"now using normalize: {self.if_normalize}")

        self.img_size = img_size
        self.transform = ImageAugment(im_res=self.img_size, visual_augment=self.data_augment, if_normalize=if_normalize, mean=mean, std=std)

    def __len__(self):
        return len(self.data)

    def get_real_fake_ratio(self):
        real = 0
        fake = 0

        for _, label in self.data:
            if label == 0:
                real += 1
            else:
                fake += 1

        if fake == 0:
            return 1.0

        return real / fake

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print("bad image:", img_path)

            # 随机换一张
            new_idx = torch.randint(0, len(self.data), (1,)).item()
            return self.__getitem__(new_idx)
        
        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.float32) # for BCE
        #label = torch.tensor(int(label), dtype=torch.long)  # for CE

        return img, label


class FineTuneDataset_mask(Dataset):
    def __init__(self, csv_file, data_augment=False, mean = [0.485, 0.456, 0.406],std  = [0.229, 0.224, 0.225], if_normalize=True, img_size=512):
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)   # 跳过 header
            for row in reader:
                self.data.append((row[0], row[1], int(row[2])))  # (image_path, mask_path, label)

        self.data_augment = data_augment
        print(f"now using data augmentation: {self.data_augment}")

        self.if_normalize = if_normalize
        print(f"now using normalize: {self.if_normalize}")

        self.img_size = img_size
        self.normalize = T.Normalize(mean=mean, std=std)

        self.transform = ImageAugment_mask()
    def __len__(self):
        return len(self.data)

    def get_real_fake_ratio(self):
        real = 0
        fake = 0

        for _, _, label in self.data:
            if label == 0:
                real += 1
            else:
                fake += 1

        if fake == 0:
            return 1.0

        return real / fake

    def prepare_gt_img(self, tp_img, gt_path, label):
        if label == 0 and gt_path is None:
            return np.zeros((*tp_img.shape[:2], 3))
        elif label == 1 and gt_path is None:
            return np.full((*tp_img.shape[:2], 3), 255, dtype=np.uint8)
        else:
            return np.array(gt_path.convert('RGB'))


    def process_masks(self, gt_img):
        gt_img = (np.mean(gt_img, axis=2, keepdims=True) > 127.5) * 1.0
        gt_img = gt_img.transpose(2, 0, 1)[0]
        return gt_img

    def __getitem__(self, idx):
        img_path, mask_path, label = self.data[idx]
        img = Image.open(img_path)
        img = np.array(img.convert('RGB'))

        img = self.transform(img) # tensor

        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None

        mask = self.prepare_gt_img(img, mask, label) # [H, W, 3]
        mask = self.process_masks(mask) # [H, W]
        
        #img = F.resize(img, (self.im_res, self.im_res))
        #img = F.to_tensor(img)
        #img = self.normalize(img)

        label = torch.tensor(label, dtype=torch.float32) # for BCE

        return img, mask, label


    def collate_fn(self, batch):

        imgs, masks, labels = zip(*batch)

        patch_size = 16
        min_size = 384
        max_size = 768

        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # ===== 1. batch-level resolution =====
        def sample_size():
            return random.randint(min_size // patch_size, max_size // patch_size) * patch_size

        H = sample_size()
        W = sample_size()

        # ===== 2. 数据增强（img + mask 同步）=====
        aug_imgs = []
        aug_masks = []

        for img, mask in zip(imgs, masks):

            aug_type = random.choice(["none", "hflip", "rot90", "rot180", "rot270"])

            if aug_type == "hflip":
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

            elif aug_type == "rot90":
                img = np.rot90(img, k=1).copy()
                mask = np.rot90(mask, k=1).copy()

            elif aug_type == "rot180":
                img = np.rot90(img, k=2).copy()
                mask = np.rot90(mask, k=2).copy()

            elif aug_type == "rot270":
                img = np.rot90(img, k=3).copy()
                mask = np.rot90(mask, k=3).copy()

            # none: 不变

            aug_imgs.append(img)
            aug_masks.append(mask)

        imgs = aug_imgs
        masks = aug_masks

        # ===== 3. image resize interpolation =====
        interp = random.choice([
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4
        ])

        # ===== 4. resize imgs =====
        resized_imgs = []
        for img in imgs:
            img = cv2.resize(img, (W, H), interpolation=interp)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img = normalize(img)
            resized_imgs.append(img)

        resized_imgs = torch.stack(resized_imgs)  # [B, 3, H, W]

        # ===== 5. resize masks =====
        resized_masks = []
        for mask in masks:
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy(mask)
            resized_masks.append(mask)

        resized_masks = torch.stack(resized_masks).float()  # [B, H, W]

        # ===== 6. labels =====
        labels = torch.stack(labels).float()

        return resized_imgs, resized_masks, labels # [B,3,H,W], [B,H,W], [B]

# #数据增强类
class ImageTransform:
    def __init__(
        self,
        im_res=512,
        visual_augment=True,
        if_normalize=True,
        mean=[0.532625138759613, 0.4048449993133545, 0.3708747327327728],
        std=[0.25850796699523926, 0.21054500341415405, 0.20785294473171234]
    ):
        self.im_res = im_res
        self.visual_augment = visual_augment
        self.if_normalize = if_normalize

        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, img):
        """
        img: PIL.Image
        return: Tensor [3,H,W]
        """

        # resize
        img = F.resize(img, (self.im_res, self.im_res))


        if not self.visual_augment:
            img = F.to_tensor(img)
            if self.if_normalize:
                img = self.normalize(img)
            return img
        
        if random.random() < 0.3:
            k = random.choice([1, 2, 3])  # 1:90°, 2:180°, 3:270°
            img = F.rotate(img, angle=90 * k)
            
        # -------------------
        # Flip
        # -------------------
        if random.random() < 0.5:
            img = F.hflip(img)


        # -------------------
        # RandomResizedCrop
        # -------------------
        # i, j, h, w = T.RandomResizedCrop.get_params(
        #     img,
        #     scale=(0.9, 1.0),
        #     ratio=(0.75, 1.33),
        # )
        # img = F.resized_crop(img, i, j, h, w, (self.im_res, self.im_res))

        # -------------------
        # Down-Up Resize
        # -------------------
        if random.random() < 0.3:
            scale = random.uniform(0.5, 0.9)
            small = int(self.im_res * scale)

            img = F.resize(img, (small, small))
            img = F.resize(img, (self.im_res, self.im_res))

        # # -------------------
        # # JPEG Compression
        # # -------------------
        if random.random() < 0.3:
            quality = random.randint(30, 95)
            img = self.jpeg_compress(img, quality)

        # # -------------------
        # # Gaussian Blur
        # # -------------------
        # if random.random() < 0.2:
        #     sigma = random.uniform(0.1, 2.0)
        #     img = F.gaussian_blur(img, kernel_size=13, sigma=sigma)


        # # -------------------
        # # Color Jitter
        # # -------------------
        # brightness = random.uniform(0.8, 1.2)
        # contrast = random.uniform(0.8, 1.2)
        # saturation = random.uniform(0.8, 1.2)
        # hue = random.uniform(-0.1, 0.1)

        # if random.random() < 0.5:
        #     img = F.adjust_brightness(img, brightness)
        # if random.random() < 0.5:
        #     img = F.adjust_contrast(img, contrast)
        # if random.random() < 0.5:
        #     img = F.adjust_saturation(img, saturation)
        # if random.random() < 0.5:
        #     img = F.adjust_hue(img, hue)
        # # -------------------
        # # Grayscale
        # # -------------------
        # if random.random() < 0.2:
        #     img = F.rgb_to_grayscale(img, num_output_channels=3)

        # -------------------
        # To Tensor
        # -------------------
        img = F.to_tensor(img)


        # -------------------
        # Normalize
        # -------------------
        if self.if_normalize:
            img = self.normalize(img)

        return img


    def jpeg_compress(self, img, quality):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class ImageAugment_old:
    def __init__(
        self,
        im_res=224,
        visual_augment=True,
        if_normalize=True,
        # mean=[0.4798, 0.3348, 0.2876],
        # std=[0.2003, 0.1622, 0.1588],
        mean = [0.532625138759613, 0.4048449993133545, 0.3708747327327728],
        std = [0.25850796699523926, 0.21054500341415405, 0.20785294473171234],
        p=0.5,
        s=1.0,
        min_size=32
    ):

        self.im_res = im_res
        self.visual_augment = visual_augment
        self.if_normalize = if_normalize
        self.mean = mean
        self.std = std
        self.p = p
        self.s = s
        self.min_size = min_size

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    # --------------------------------------------------
    # 数值安全
    # --------------------------------------------------

    def safe_img(self, img):

        img = np.nan_to_num(img)
        img = np.clip(img, -255, 255)

        return img


    def shift_with_black_border(self, img):

        h, w = img.shape[:2]

        # 最大平移比例（建议 5%~10%）
        max_ratio = 0.08

        max_dx = int(w * max_ratio)
        max_dy = int(h * max_ratio)

        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)

        # 仿射矩阵
        M = np.float32([
            [1, 0, dx],
            [0, 1, dy]
        ])

        border = random.randint(0,20)
        borderValue=(border,border,border)

        shifted = cv2.warpAffine(
            img,
            M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=borderValue  # 黑边
        )

        return shifted

    # --------------------------------------------------
    # grayscale
    # --------------------------------------------------

    def grayscale(self, img):

        if random.random() < 0.3:   # 30%概率灰度
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            img = np.stack([gray, gray, gray], axis=2)

        return img

    # --------------------------------------------------
    # Blur
    # --------------------------------------------------

    def blur(self, img):

        choice = random.randint(0, 2)

        # --------------------------------
        # anisotropic gaussian blur
        # --------------------------------
        if choice == 0:

            sigma_x = random.uniform(0.2, 8*self.s)
            sigma_y = random.uniform(0.2, 8*self.s)

            kx = int(6*sigma_x + 1)
            ky = int(6*sigma_y + 1)

            kx = max(3, min(31, kx | 1))
            ky = max(3, min(31, ky | 1))

            img = cv2.GaussianBlur(img, (kx, ky), sigmaX=sigma_x, sigmaY=sigma_y)

        # --------------------------------
        # isotropic gaussian blur
        # --------------------------------
        elif choice == 1:

            sigma = random.uniform(0.2, 4*self.s)

            k = int(6*sigma + 1)
            k = max(3, min(31, k | 1))

            img = cv2.GaussianBlur(img, (k, k), sigma)

        # --------------------------------
        # mean blur
        # --------------------------------
        else:

            w = random.randint(3, 11)
            w = w | 1

            img = cv2.blur(img, (w, w))

        return img
    # --------------------------------------------------
    # Resize
    # --------------------------------------------------

    def resize(self, img):

        h, w = img.shape[:2]

        r = random.random()

        if r < 0.2:
            scale = random.uniform(1, 2)

        elif r < 0.9:
            scale = random.uniform(0.25/self.s, 1)

        else:
            scale = 1

        new_w = max(self.min_size, int(w*scale))
        new_h = max(self.min_size, int(h*scale))

        method = random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA
        ])

        img = cv2.resize(img, (new_w, new_h), interpolation=method)

        return img

    # --------------------------------------------------
    # Gaussian Noise
    # --------------------------------------------------

    def gaussian_noise(self, img):

        img = img.astype(np.float32)

        option = random.random()

        # 降低强度
        if random.random() < 0.5:
            l1, l2 = 5, 20
        else:
            l1, l2 = 20, 35

        sigma = random.uniform(l1*self.s, l2*self.s)

        if option < 0.4:

            noise = np.random.normal(0, sigma, img.shape)

        elif option < 0.8:

            noise = np.random.normal(0, sigma, (img.shape[0], img.shape[1], 1))
            noise = np.repeat(noise, 3, axis=2)

        else:

            cov = np.random.rand(3,3)
            cov = cov @ cov.T

            noise = np.random.multivariate_normal(
                [0,0,0],
                cov,
                img.shape[0]*img.shape[1]
            ).reshape(img.shape)

        img = img + noise

        return img

    # --------------------------------------------------
    # Non Gaussian Noise
    # --------------------------------------------------

    def non_gaussian_noise(self, img):

        img = img.astype(np.float32)

        if random.random() < 0.5:

            # 降低乘法噪声
            l1, l2 = 0.05, 0.2
            sigma = random.uniform(l1*self.s, l2*self.s)

            noise = np.random.normal(0, sigma, img.shape)

            img = img + img * noise

        else:

            img = np.nan_to_num(img)
            img = np.clip(img, 0, None)

            vals = len(np.unique(img))
            vals = max(vals, 2)

            vals = 2 ** np.ceil(np.log2(vals))

            img = np.random.poisson(img * vals) / float(vals)

        return img

    # --------------------------------------------------
    # JPEG Compression
    # --------------------------------------------------

    def jpeg_compress(self, img):

        img = np.clip(img, 0, 255).astype(np.uint8)

        q = random.randint(10, 95)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]

        success, enc = cv2.imencode(".jpg", img, encode_param)

        if success:
            img = cv2.imdecode(enc, 1)

        return img

    # --------------------------------------------------
    # Brightness / Contrast
    # --------------------------------------------------

    def enhance(self, img):

        img = np.clip(img, 0, 255).astype(np.uint8)

        img = Image.fromarray(img)

        factor = random.uniform(0.5, 1.5)

        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(img)
        else:
            enhancer = ImageEnhance.Contrast(img)

        img = enhancer.enhance(factor)

        return np.array(img)

    # --------------------------------------------------
    # Distractor
    # --------------------------------------------------

    def distractor(self, img):

        h, w = img.shape[:2]

        if h < 20 or w < 20:
            return img

        if random.random() < 0.5:

            x = random.randint(20, min(100, w))
            y = int(random.uniform(0.8*x, 1.2*x))

            if y >= h:
                y = h - 1

            if x >= w:
                x = w - 1

            if x <= 0 or y <= 0:
                return img

        else:

            max_x = max(1, w-50)
            max_y = max(10, h-10)

            pos = (
                random.randint(0, max_x),
                random.randint(10, max_y)
            )

            text = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))

            cv2.putText(
                img,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                random.uniform(0.5,1.5),
                (
                    random.randint(0,255),
                    random.randint(0,255),
                    random.randint(0,255)
                ),
                random.randint(1,3)
            )

        return img

    # --------------------------------------------------
    # Main
    # --------------------------------------------------

    def __call__(self, img):

        if random.random() < 0.5:
            img = F.hflip(img)
            
        if isinstance(img, Image.Image):
            img = np.array(img)

        img = img.astype(np.float32)


        if self.visual_augment:
            degradations = [
                self.shift_with_black_border,  # 几何变化
                self.resize,                   # 尺度变化
                self.blur,                     # 光学模糊
                self.enhance,                  # 亮度/对比度
                self.grayscale,                # 颜色变化
                self.gaussian_noise,           # 传感器噪声
                self.non_gaussian_noise,       # 雪花噪声
                self.jpeg_compress,            # 压缩（通常最后）
                self.distractor                # 贴片干扰
            ]
        else:
            degradations = []

        num_aug = random.randint(1, 9)

        ops = random.sample(degradations, num_aug)

        for d in ops:

            if random.random() < self.p:

                img = d(img)

                img = self.safe_img(img)

        img = np.clip(img, 0, 255).astype(np.uint8)

        img = cv2.resize(img, (self.im_res, self.im_res))

        img = Image.fromarray(img)

        img = self.to_tensor(img)

        if self.if_normalize:
            img = self.normalize(img)

        return img




class ImageAugment:

    def __init__(
        self,
        im_res=512,
        visual_augment=True,
        if_normalize=True,
        mean=[0.532625138759613, 0.4048449993133545, 0.3708747327327728],
        std=[0.25850796699523926, 0.21054500341415405, 0.20785294473171234],
        min_size=32,
        p=0.5
    ):

        self.p = p
        self.im_res = im_res
        self.visual_augment = visual_augment
        self.if_normalize = if_normalize

        self.mean = mean
        self.std = std

        self.min_size = min_size

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    # --------------------------------------------------
    # utility
    # --------------------------------------------------

    def safe_img(self, img):

        img = np.nan_to_num(img)
        img = np.clip(img, -255, 255)

        return img

    # --------------------------------------------------
    # geometric
    # --------------------------------------------------
    def rotate_90(self, img, level):
        """
        level 控制旋转发生的概率
        level ∈ [1,5] → p ∈ [0.1, 0.5]
        """

        # 将 level 映射为概率（你可以调这个范围）
        p = np.interp(level, [1, 5], [0.1, 0.5])

        if random.random() > p:
            return img  # 不旋转

        k = random.choice([1, 2, 3])  # 1:90, 2:180, 3:270

        if k == 1:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif k == 2:
            img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return img
    
    def shift_with_black_border(self, img, level):

        h, w = img.shape[:2]

        ratio = np.interp(level, [1,5], [0.02, 0.10])

        dx = int(random.uniform(-ratio, ratio) * w)
        dy = int(random.uniform(-ratio, ratio) * h)

        M = np.float32([[1,0,dx],[0,1,dy]])

        # 随机黑边或白边
        if random.random() < 0.5:
            v = random.randint(0,20)        # 黑边
        else:
            v = random.randint(235,255)     # 白边

        border = v

        img = cv2.warpAffine(
            img,
            M,
            (w,h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(border,border,border)
        )

        return img

    def resize(self, img, level):

        h, w = img.shape[:2]

        scale = np.interp(level,[1,5],[0.9,0.3])

        if random.random() < 0.5:
            scale = random.uniform(scale,1)
        else:
            scale = random.uniform(1,1.8)

        new_w = max(self.min_size,int(w*scale))
        new_h = max(self.min_size,int(h*scale))

        method = random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA
        ])

        img = cv2.resize(img,(new_w,new_h),interpolation=method)

        return img

    def dynamic_resize(self, img, min_size=384, max_size=1152, patch_size=16):
        """
        动态分辨率 resize

        Args:
            img: numpy array (H, W, C)
            min_size, max_size: 分辨率范围
            patch_size: patch 大小（保证能整除）

        Returns:
            resized img
        """

        # === 1. 随机采样 H, W（保证是 patch_size 的倍数） ===
        def sample_size():
            size = random.randint(min_size // patch_size, max_size // patch_size)
            return size * patch_size

        new_h = sample_size()
        new_w = sample_size()

        # === 2. 随机选择插值方法 ===
        interpolation_methods = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4
        ]
        interp = random.choice(interpolation_methods)

        # === 3. resize ===
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

        return img

    # --------------------------------------------------
    # blur
    # --------------------------------------------------

    def blur(self, img, level):

        sigma = np.interp(level,[1,5],[0.5,6])

        k = int(6*sigma+1)
        k = max(3,min(31,k|1))

        img = cv2.GaussianBlur(img,(k,k),sigma)

        return img

    def mean_blur(self, img, level):

        k = int(np.interp(level,[1,5],[3,11]))
        k = k | 1

        img = cv2.blur(img,(k,k))

        return img

    def defocus_blur(self, img, level):

        radius = int(np.interp(level,[1,5],[2,10]))
        k = radius*2+1

        kernel = np.zeros((k,k))
        cv2.circle(kernel,(radius,radius),radius,1,-1)

        kernel /= kernel.sum()

        img = cv2.filter2D(img,-1,kernel)

        return img

    # --------------------------------------------------
    # grayscale
    # --------------------------------------------------

    def grayscale(self, img, level):

        gray = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY)

        img = np.stack([gray,gray,gray],axis=2)

        return img

    # --------------------------------------------------
    # color
    # --------------------------------------------------

    def saturation(self, img, level):

        img = img.astype(np.uint8)

        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        factor = np.interp(level,[1,5],[0.5,1.8])

        hsv[:,:,1] = np.clip(hsv[:,:,1]*factor,0,255)

        img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        return img

    def color_shift(self, img, level):

        img = img.astype(np.float32)

        shift = np.interp(level,[1,5],[5,40])

        for c in range(3):
            img[:,:,c]+=random.uniform(-shift,shift)

        return img

    def color_quantization(self, img, level):

        img = img.astype(np.uint8)

        k = int(np.interp(level,[1,5],[64,8]))

        Z = img.reshape((-1,3))
        Z = np.float32(Z)

        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0
        )

        _,label,center = cv2.kmeans(
            Z,
            k,
            None,
            criteria,
            5,
            cv2.KMEANS_RANDOM_CENTERS
        )

        center = np.uint8(center)

        res = center[label.flatten()]
        img = res.reshape(img.shape)

        return img

    # --------------------------------------------------
    # noise
    # --------------------------------------------------

    def gaussian_noise(self, img, level):

        img = img.astype(np.float32)

        sigma = np.interp(level,[1,5],[5,35])

        noise = np.random.normal(0,sigma,img.shape)

        img = img + noise

        return img

    def salt_pepper_noise(self, img, level):

        img = img.astype(np.float32)

        amount = np.interp(level,[1,5],[0.002,0.02])

        h,w,c = img.shape

        num = int(amount*h*w)

        coords = (
            np.random.randint(0,h,num),
            np.random.randint(0,w,num)
        )

        img[coords]=255

        coords = (
            np.random.randint(0,h,num),
            np.random.randint(0,w,num)
        )

        img[coords]=0

        return img

    def speckle_noise(self, img, level):

        img = img.astype(np.float32)

        sigma = np.interp(level,[1,5],[0.05,0.3])

        noise = np.random.normal(0,sigma,img.shape)

        img = img + img*noise

        return img

    def poisson_noise(self, img, level):

        img = np.clip(img,0,None)

        vals = len(np.unique(img))
        vals = max(vals,2)

        vals = 2**np.ceil(np.log2(vals))

        img = np.random.poisson(img*vals)/float(vals)

        return img

    # --------------------------------------------------
    # jpeg
    # --------------------------------------------------

    def jpeg_compress(self, img, level):

        img = np.clip(img,0,255).astype(np.uint8)

        q = int(np.interp(level,[1,5],[95,10]))

        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),q]

        success,enc=cv2.imencode(".jpg",img,encode_param)

        if success:
            img=cv2.imdecode(enc,1)

        return img

    # --------------------------------------------------
    # brightness / contrast
    # --------------------------------------------------

    def brightness_increase(self, img, level):

        img=np.clip(img,0,255).astype(np.uint8)

        img=Image.fromarray(img)

        factor=np.interp(level,[1,5],[1.1,1.8])

        enhancer=ImageEnhance.Brightness(img)

        img=enhancer.enhance(factor)

        return np.array(img)

    def brightness_decrease(self, img, level):

        img=np.clip(img,0,255).astype(np.uint8)

        img=Image.fromarray(img)

        factor=np.interp(level,[1,5],[0.9,0.4])

        enhancer=ImageEnhance.Brightness(img)

        img=enhancer.enhance(factor)

        return np.array(img)

    def contrast_adjust(self, img, level):

        img=np.clip(img,0,255).astype(np.uint8)

        img=Image.fromarray(img)

        factor=np.interp(level,[1,5],[0.6,1.6])

        enhancer=ImageEnhance.Contrast(img)

        img=enhancer.enhance(factor)

        return np.array(img)

    # --------------------------------------------------
    # occlusion
    # --------------------------------------------------

    def distractor(self, img, level):

        h, w = img.shape[:2]

        if h < 20 or w < 20:
            return img

        if random.random() < 0.5:

            # 文本大小随level变化
            font_scale = np.interp(level, [1, 5], [0.5, 1.5])
            thickness = random.randint(1, 3)

            text = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))

            # 安全随机范围
            max_x = max(1, w - 50)
            max_y = max(10, h - 10)

            pos = (
                random.randint(0, max_x),
                random.randint(10, max_y)
            )

            cv2.putText(
                img,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ),
                thickness
            )

        return img

    # --------------------------------------------------
    # mixup
    # --------------------------------------------------

    def mix_up(self, img1):

        """
        Self rotation mixup

        img1:
            numpy array (H,W,C)

        return:
            mixed image
        """

        img1 = img1.astype(np.float32)

        # -----------------------------------
        # 随机旋转
        # -----------------------------------
        k = random.choice([1, 2, 3])

        if k == 1:
            img2 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)

        elif k == 2:
            img2 = cv2.rotate(img1, cv2.ROTATE_180)

        else:
            img2 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # -----------------------------------
        # resize回原尺寸
        # （90/270会交换H/W）
        # -----------------------------------
        h, w = img1.shape[:2]

        img2 = cv2.resize(img2, (w, h))

        # -----------------------------------
        # mixup
        # -----------------------------------
        lam = random.uniform(0.85, 0.95)

        mixed = lam * img1 + (1 - lam) * img2

        return mixed

    # --------------------------------------------------
    # main
    # --------------------------------------------------

    def __call__(self, img):

        if random.random()<0.5:
            img=F.hflip(img)

        if isinstance(img,Image.Image):
            img=np.array(img)

        img=img.astype(np.float32)

        if self.visual_augment:

            degradations=[
                self.rotate_90,
                self.shift_with_black_border,
                #self.resize,

                self.blur,
                self.mean_blur,
                self.defocus_blur,

                self.brightness_increase,
                self.brightness_decrease,
                self.contrast_adjust,

                self.grayscale,
                self.saturation,
                #self.color_shift,
                self.color_quantization,

                self.gaussian_noise,
                self.salt_pepper_noise,
                self.speckle_noise,
                self.poisson_noise,

                self.jpeg_compress,

                #self.distractor
            ]

        else:
            degradations=[]

        num_aug = random.randint(2, min(4, len(degradations)))

        ops = random.sample(degradations,num_aug)


        for d in ops:

            level=random.randint(1,4)

            if random.random() < self.p:
                img=d(img,level)

            img=self.safe_img(img)


        img=np.clip(img,0,255).astype(np.uint8)

        if random.random() < 0.3:

            img = self.mix_up(img)

            img = self.safe_img(img)

            img = np.clip(img, 0, 255).astype(np.uint8)

        # img=self.dynamic_resize(img)
        # #img=cv2.resize(img,(self.im_res,self.im_res))

        # img=Image.fromarray(img)

        # img=self.to_tensor(img)

        # if self.if_normalize:
        #     img=self.normalize(img)

        return img


class ImageAugment_mask:

    def __init__(
        self,
        im_res=512,
        visual_augment=True,
        if_normalize=True,
        mean=[0.532625138759613, 0.4048449993133545, 0.3708747327327728],
        std=[0.25850796699523926, 0.21054500341415405, 0.20785294473171234],
        min_size=32,
        p=0.5
    ):

        self.p = p
        self.im_res = im_res
        self.visual_augment = visual_augment
        self.if_normalize = if_normalize

        self.mean = mean
        self.std = std

        self.min_size = min_size

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    # --------------------------------------------------
    # utility
    # --------------------------------------------------

    def safe_img(self, img):

        img = np.nan_to_num(img)
        img = np.clip(img, -255, 255)

        return img

    # --------------------------------------------------
    # geometric
    # --------------------------------------------------
    def rotate_90(self, img, level):
        """
        level 控制旋转发生的概率
        level ∈ [1,5] → p ∈ [0.1, 0.5]
        """

        # 将 level 映射为概率（你可以调这个范围）
        p = np.interp(level, [1, 5], [0.1, 0.5])

        if random.random() > p:
            return img  # 不旋转

        k = random.choice([1, 2, 3])  # 1:90, 2:180, 3:270

        if k == 1:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif k == 2:
            img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return img
    
    def shift_with_black_border(self, img, level):

        h, w = img.shape[:2]

        ratio = np.interp(level, [1,5], [0.02, 0.20])

        dx = int(random.uniform(-ratio, ratio) * w)
        dy = int(random.uniform(-ratio, ratio) * h)

        M = np.float32([[1,0,dx],[0,1,dy]])

        # 随机黑边或白边
        if random.random() < 0.5:
            v = random.randint(0,20)        # 黑边
        else:
            v = random.randint(235,255)     # 白边

        border = v

        img = cv2.warpAffine(
            img,
            M,
            (w,h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(border,border,border)
        )

        return img

    def resize(self, img, level):

        h, w = img.shape[:2]

        scale = np.interp(level,[1,5],[0.9,0.3])

        if random.random() < 0.5:
            scale = random.uniform(scale,1)
        else:
            scale = random.uniform(1,1.8)

        new_w = max(self.min_size,int(w*scale))
        new_h = max(self.min_size,int(h*scale))

        method = random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA
        ])

        img = cv2.resize(img,(new_w,new_h),interpolation=method)

        return img

    def dynamic_resize(self, img, min_size=384, max_size=1152, patch_size=16):
        """
        动态分辨率 resize

        Args:
            img: numpy array (H, W, C)
            min_size, max_size: 分辨率范围
            patch_size: patch 大小（保证能整除）

        Returns:
            resized img
        """

        # === 1. 随机采样 H, W（保证是 patch_size 的倍数） ===
        def sample_size():
            size = random.randint(min_size // patch_size, max_size // patch_size)
            return size * patch_size

        new_h = sample_size()
        new_w = sample_size()

        # === 2. 随机选择插值方法 ===
        interpolation_methods = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4
        ]
        interp = random.choice(interpolation_methods)

        # === 3. resize ===
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

        return img

    # --------------------------------------------------
    # blur
    # --------------------------------------------------

    def blur(self, img, level):

        sigma = np.interp(level,[1,5],[0.5,6])

        k = int(6*sigma+1)
        k = max(3,min(31,k|1))

        img = cv2.GaussianBlur(img,(k,k),sigma)

        return img

    def mean_blur(self, img, level):

        k = int(np.interp(level,[1,5],[3,11]))
        k = k | 1

        img = cv2.blur(img,(k,k))

        return img

    def defocus_blur(self, img, level):

        radius = int(np.interp(level,[1,5],[2,10]))
        k = radius*2+1

        kernel = np.zeros((k,k))
        cv2.circle(kernel,(radius,radius),radius,1,-1)

        kernel /= kernel.sum()

        img = cv2.filter2D(img,-1,kernel)

        return img

    # --------------------------------------------------
    # grayscale
    # --------------------------------------------------

    def grayscale(self, img, level):

        gray = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY)

        img = np.stack([gray,gray,gray],axis=2)

        return img

    # --------------------------------------------------
    # color
    # --------------------------------------------------

    def saturation(self, img, level):

        img = img.astype(np.uint8)

        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        factor = np.interp(level,[1,5],[0.5,1.8])

        hsv[:,:,1] = np.clip(hsv[:,:,1]*factor,0,255)

        img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        return img

    def color_shift(self, img, level):

        img = img.astype(np.float32)

        shift = np.interp(level,[1,5],[5,40])

        for c in range(3):
            img[:,:,c]+=random.uniform(-shift,shift)

        return img

    def color_quantization(self, img, level):

        img = img.astype(np.uint8)

        k = int(np.interp(level,[1,5],[64,8]))

        Z = img.reshape((-1,3))
        Z = np.float32(Z)

        criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0
        )

        _,label,center = cv2.kmeans(
            Z,
            k,
            None,
            criteria,
            5,
            cv2.KMEANS_RANDOM_CENTERS
        )

        center = np.uint8(center)

        res = center[label.flatten()]
        img = res.reshape(img.shape)

        return img

    # --------------------------------------------------
    # noise
    # --------------------------------------------------

    def gaussian_noise(self, img, level):

        img = img.astype(np.float32)

        sigma = np.interp(level,[1,5],[5,35])

        noise = np.random.normal(0,sigma,img.shape)

        img = img + noise

        return img

    def salt_pepper_noise(self, img, level):

        img = img.astype(np.float32)

        amount = np.interp(level,[1,5],[0.002,0.02])

        h,w,c = img.shape

        num = int(amount*h*w)

        coords = (
            np.random.randint(0,h,num),
            np.random.randint(0,w,num)
        )

        img[coords]=255

        coords = (
            np.random.randint(0,h,num),
            np.random.randint(0,w,num)
        )

        img[coords]=0

        return img

    def speckle_noise(self, img, level):

        img = img.astype(np.float32)

        sigma = np.interp(level,[1,5],[0.05,0.3])

        noise = np.random.normal(0,sigma,img.shape)

        img = img + img*noise

        return img

    def poisson_noise(self, img, level):

        img = np.clip(img,0,None)

        vals = len(np.unique(img))
        vals = max(vals,2)

        vals = 2**np.ceil(np.log2(vals))

        img = np.random.poisson(img*vals)/float(vals)

        return img

    # --------------------------------------------------
    # jpeg
    # --------------------------------------------------

    def jpeg_compress(self, img, level):

        img = np.clip(img,0,255).astype(np.uint8)

        q = int(np.interp(level,[1,5],[95,10]))

        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),q]

        success,enc=cv2.imencode(".jpg",img,encode_param)

        if success:
            img=cv2.imdecode(enc,1)

        return img

    # --------------------------------------------------
    # brightness / contrast
    # --------------------------------------------------

    def brightness_increase(self, img, level):

        img=np.clip(img,0,255).astype(np.uint8)

        img=Image.fromarray(img)

        factor=np.interp(level,[1,5],[1.1,1.8])

        enhancer=ImageEnhance.Brightness(img)

        img=enhancer.enhance(factor)

        return np.array(img)

    def brightness_decrease(self, img, level):

        img=np.clip(img,0,255).astype(np.uint8)

        img=Image.fromarray(img)

        factor=np.interp(level,[1,5],[0.9,0.4])

        enhancer=ImageEnhance.Brightness(img)

        img=enhancer.enhance(factor)

        return np.array(img)

    def contrast_adjust(self, img, level):

        img=np.clip(img,0,255).astype(np.uint8)

        img=Image.fromarray(img)

        factor=np.interp(level,[1,5],[0.6,1.6])

        enhancer=ImageEnhance.Contrast(img)

        img=enhancer.enhance(factor)

        return np.array(img)

    # --------------------------------------------------
    # occlusion
    # --------------------------------------------------

    def distractor(self, img, level):

        h, w = img.shape[:2]

        if h < 20 or w < 20:
            return img

        if random.random() < 0.5:

            # 文本大小随level变化
            font_scale = np.interp(level, [1, 5], [0.5, 1.5])
            thickness = random.randint(1, 3)

            text = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))

            # 安全随机范围
            max_x = max(1, w - 50)
            max_y = max(10, h - 10)

            pos = (
                random.randint(0, max_x),
                random.randint(10, max_y)
            )

            cv2.putText(
                img,
                text,
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ),
                thickness
            )

        return img

    # --------------------------------------------------
    # main
    # --------------------------------------------------

    def __call__(self, img):

        if isinstance(img,Image.Image):
            img=np.array(img)

        img=img.astype(np.float32)

        if self.visual_augment:

            degradations=[
                #self.rotate_90,
                #self.shift_with_black_border,
                #self.resize,

                self.blur,
                self.mean_blur,
                self.defocus_blur,

                self.brightness_increase,
                self.brightness_decrease,
                self.contrast_adjust,

                self.grayscale,
                self.saturation,
                #self.color_shift,
                self.color_quantization,

                self.gaussian_noise,
                self.salt_pepper_noise,
                self.speckle_noise,
                self.poisson_noise,

                self.jpeg_compress,

                #self.distractor
            ]

        else:
            degradations=[]

        num_aug = random.randint(2, min(4, len(degradations)))

        ops = random.sample(degradations,num_aug)


        for d in ops:

            level=random.randint(1,3)

            if random.random() < self.p:
                img=d(img,level)

            img=self.safe_img(img)


        img=np.clip(img,0,255).astype(np.uint8)

        #img=self.dynamic_resize(img)
        #img=cv2.resize(img,(self.im_res,self.im_res))

        # img=Image.fromarray(img)

        # img=self.to_tensor(img)

        # if self.if_normalize:
        #     img=self.normalize(img)

        return img



class EvalDataset(Dataset):
    def __init__(self, csv_file, mean = [0.485, 0.456, 0.406],std  = [0.229, 0.224, 0.225], img_size=512, mode='eval', if_test_time_augment=False, if_resolution_ensemble=False, img_size_2=768):

        self.img_size = img_size
        self.img_size_2 = img_size_2
        self.data = []

        self.if_test_time_augment = if_test_time_augment
        self.if_resolution_ensemble = if_resolution_ensemble
        self.mode = mode
        print(f"Initializing EvalDataset with mode: {self.mode}, use_test_time_augment: {self.if_test_time_augment}, use_resolution_ensemble: {self.if_resolution_ensemble}")
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)   # 跳过 header
            if self.mode == 'eval': # evaluation
                for row in reader:
                    self.data.append((row[0], int(row[1])))  # (image_path, label)
            else: # inference
                for row in reader:
                    self.data.append(row[0])

        self.normalize = T.Normalize(mean=mean, std=std)

    def _process_img(self, img, size):
        """处理单张图片到指定尺寸"""
        img = F.resize(img, (size, size))
        img = F.to_tensor(img)
        img = self.normalize(img)
        return img

    def _process_flip(self, img, size):
        """处理翻转后的图片到指定尺寸"""
        img = F.hflip(img)
        img = F.resize(img, (size, size))
        img = F.to_tensor(img)
        img = self.normalize(img)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'eval':
            img_path, label = self.data[idx]
        else:
            img_path = self.data[idx]

        img = Image.open(img_path).convert('RGB')

        if self.if_resolution_ensemble:
            # 分辨率 ensemble：返回两种尺寸
            img_1 = self._process_img(img, self.img_size)
            img_2 = self._process_img(img, self.img_size_2)

            if self.if_test_time_augment:
                img_flip_1 = self._process_flip(img, self.img_size)
                img_flip_2 = self._process_flip(img, self.img_size_2)

                if self.mode == 'eval':
                    label = torch.tensor(label, dtype=torch.float32)
                    return img_path, img_1, img_flip_1, img_2, img_flip_2, label
                else:
                    return img_path, img_1, img_flip_1, img_2, img_flip_2
            else:
                if self.mode == 'eval':
                    label = torch.tensor(label, dtype=torch.float32)
                    return img_path, img_1, img_2, label
                else:
                    return img_path, img_1, img_2
        else:
            # 原始逻辑
            img = F.resize(img, (self.img_size, self.img_size))

            if self.if_test_time_augment:
                img_flip = F.hflip(img)
                img_flip = F.to_tensor(img_flip)
                img_flip = self.normalize(img_flip)

            img = F.to_tensor(img)
            img = self.normalize(img)
            
            if self.mode == 'eval':        
                label = torch.tensor(label, dtype=torch.float32)
                if self.if_test_time_augment:
                    return img_path, img, img_flip, label
                else:
                    return img_path, img, label
            else:
                if self.if_test_time_augment:
                    return img_path, img, img_flip
                else:
                    return img_path, img
