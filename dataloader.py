import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import PIL.Image as pil
import glob
import random
import os
import io

class Residual_data(object):
    def __init__(self, image_dir, patch_size, gaussian_noise_level, downsampling_factor, jpeg_quality,use_fast_loader=False):
        self.image_files = sorted(glob.glob(image_dir+'/*'))
        self.patch_size = patch_size
        self.gaussian_noise_level = gaussian_noise_level
        self.downsampling_factor = downsampling_factor
        self.jpeg_quality = jpeg_quality
        self.use_fast_loader = use_fast_loader

    def __getitem__(self, idx):
        if self.use_fast_loader:
            pass
        else:
            clean_image = pil.open(self.image_files[idx]).convert('RGB')

        #crop patches from original images
        crop_x = random.randint(0, clean_image.width - self.patch_size)
        crop_y = random.randint(0, clean_image.height - self.patch_size)
        clean_image = clean_image.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))

        #temp noise image 
        noisy_image = clean_image.copy()
        gaussian_noise = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)

        #add gaussian noise to clean_image
        if self.gaussian_noise_level is not None:
            if len(self.gaussian_noise_level) == 1:
                sigma = self.gaussian_noise_level[0]
            else:
                sigma = random.randint(self.gaussian_noise_level[0], self.gaussian_noise_level[1])
            gaussian_noise += np.random.normal(0.0, sigma, (clean_image.height, clean_image.width, 3)).astype(np.float32)

        #downsample
        if self.downsampling_factor is not None:
            if len(self.downsampling_factor) == 1:
                downsampling_factor = self.downsampling_factor[0]
            else:
                downsampling_factor = random.randint(self.downsampling_factor[0], self.downsampling_factor[1])
            
            noisy_image = noisy_image.resize((self.patch_size//downsampling_factor, self.patch_size//downsampling_factor),
                                                resample=pil.BICUBIC)
            noisy_image = noisy_image.resize((self.patch_size,self.patch_size), resample=pil.BICUBIC)

        #jpeg noise jpeg compression)
        if self.jpeg_quality is not None:
            if len(self.jpeg_quality) == 1:
                quality = self.jpeg_quality[0]
            else:
                quality = random.randint(self.jpeg_quality[0],self.jpeg_quality[1])
                buffer = io.BytesIO()
                noisy_image.save(buffer, format='jpeg', quality=quality)
                noisy_image = pil.open(buffer)

        clean_image = np.array(clean_image).astype(np.float32)
        noisy_image = np.array(noisy_image).astype(np.float32)
        noisy_image += gaussian_noise

        input = np.transpose(noisy_image, axes=[2, 0, 1])
        label = np.transpose(clean_image, axes=[2, 0, 1])

        input /= 255.0
        label /= 255.0


        return input, label

    def __len__(self):
        return len(self.image_files)

if __name__ == '__main__':
    path = 'datasets/detection/BSR/BSDS500/data/images/train'
    gnl = '25'
    df = None
    jq = None
    if gnl is not None:
        gnl = list(map(lambda x: int(x), gnl.split(',')))
    if df is not None:
        df = list(map(lambda x: int(x), df.split(',')))
    if jq is not None:
        jq = list(map(lambda x: int(x), jq.split(',')))

    dataset = Residual_data(path, 50, gnl, None, None, False)
    # print(len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True,drop_last=True)
    for data in dataloader:
        input, label = data
        print(input)
        print(label)
        print("********")

