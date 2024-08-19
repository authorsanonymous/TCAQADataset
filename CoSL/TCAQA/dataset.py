import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import pickle as pkl
from opts import *
from scipy import stats


def load_image_train(image_path, hori_flip, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0, 3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if hori_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0, 3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


class VideoDataset(Dataset):

    def __init__(self, mode, args):
        super(VideoDataset, self).__init__()
        self.mode = mode  
        self.args = args
        self.annotations = pkl.load(open(os.path.join(info_dir, 'final_annotations_dict1.pkl'), 'rb'))
        self.keys = pkl.load(open(os.path.join(info_dir, f'{self.mode}_split_1.pkl'), 'rb'))

    def proc_label(self, data):
        tmp = stats.norm.pdf(np.arange(output_dim['USDL-CoSL']), loc=data['final_score'] * (output_dim['USDL-CoSL']-1) / label_max, scale=self.args.std).astype(
            np.float32)
        data['soft_label'] = tmp / tmp.sum()
        tmp = [stats.norm.pdf(np.arange(output_dim['MUSDL-CoSL']), loc=judge_score * (output_dim['MUSDL-CoSL']-1) / judge_max, scale=self.args.std).astype(np.float32)
                for judge_score in data['judge_scores']]
        tmp = np.stack(tmp)
        data['soft_judge_scores'] = tmp / tmp.sum(axis=-1, keepdims=True) 

    def get_imgs(self, key):
        transform = transforms.Compose([transforms.CenterCrop(H),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image_list = sorted((glob.glob(os.path.join(frames_dir,
                                                    str('{:02d}_{:04d}'.format(key[0], key[1])),
                                                    '*.jpg'))))
        sample_range  = np.linspace(0, len(image_list) - 1, num=num_frames,dtype=np.int)
        # spatial augmentation
        if self.mode == 'train':
            hori_flip = random.randint(0, 1)
        images = torch.zeros(num_frames, C, H, W)
        for j, i in enumerate(sample_range):
            if self.mode == 'train':
                images[j] = load_image_train(image_list[i], hori_flip, transform)
            if self.mode == 'test':
                images[j] = load_image(image_list[i], transform)
        return images

    def __getitem__(self, ix):
        key = self.keys[ix]
        data = {}
        data['video'] = self.get_imgs(key)
        data['final_score'] = self.annotations.get(key).get('final_score')
        data['difficulty'] = self.annotations.get(key).get('difficulty')
        data['judge_scores'] = self.annotations.get(key).get('judge_scores')
        self.proc_label(data)
        return data

    def __len__(self):
        sample_pool = len(self.keys)
        return sample_pool

