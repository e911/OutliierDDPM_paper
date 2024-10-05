import numpy as np
from torchvision import transforms

from diffuseNew.utils.lib import *
from torchvision.transforms import Compose, Lambda, ToPILImage, Resize, CenterCrop, ToTensor

image_size = 28

transform = Compose([
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
    Lambda(lambda t: (t * 2) - 1),

])

reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])


transform_dataset = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])


def transforms_dataset(examples):
   examples["pixel_values"] = [transform_dataset(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples

