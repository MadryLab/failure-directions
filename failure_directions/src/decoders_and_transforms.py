import ffcv.fields.decoders as decoders
from ffcv.transforms import RandomHorizontalFlip, Cutout, RandomTranslate
import torchvision
from PIL import Image
import numpy as np

# These take in the desired image size, and the beton image size
IMAGE_DECODERS = {
    'simple': lambda imgsz: decoders.SimpleRGBImageDecoder(),
    'resized_crop': lambda imgsz: decoders.ResizedCropRGBImageDecoder((imgsz, imgsz)),
    'random_resized_crop': lambda imgsz: decoders.RandomResizedCropRGBImageDecoder((imgsz, imgsz)),
    'center_crop_256': lambda imgsz: decoders.CenterCropRGBImageDecoder((imgsz, imgsz), 224/256),
    'center_crop_75': lambda imgsz: decoders.CenterCropRGBImageDecoder((imgsz, imgsz), 64/75),
    'center_crop_full': lambda imgsz: decoders.CenterCropRGBImageDecoder((imgsz, imgsz), 1),
}

def cifar_train_aug(hparams):
    mean = hparams['mean']
    return [
        RandomHorizontalFlip(),
        RandomTranslate(padding=2, fill=tuple(map(int, mean))),
        Cutout(4, tuple(map(int, mean))),
    ]

def imagenet_train_aug(hparams):
    return [RandomHorizontalFlip()]


class PyCutOut:
    def __init__(self, crop_size, fill):
        self.crop_size = crop_size
        self.fill = fill
        
    def __call__(self, sample):
        sample = np.array(sample)
        crop_size = self.crop_size
        H, W = sample.shape[0], sample.shape[1]
        coord = (
                    np.random.randint(H - crop_size + 1),
                    np.random.randint(W - crop_size + 1),
                )
        sample[coord[0]:coord[0]+crop_size, coord[1]:coord[1]+crop_size] = self.fill
        return Image.fromarray(sample.astype(np.uint8))
        
class PyTranslate:
    def __init__(self, padding):
        self.padding = padding
        
    def __call__(self, sample):
        sample = np.array(sample)
        pad = self.padding
        h, w, c = sample.shape
        dst = np.zeros((h+2*pad, w+2*pad, c)).astype(np.uint8)
        dst[pad:pad+h, pad:pad+w] = sample
        y_coord = np.random.randint(low=0, high=2*pad+1)
        x_coord = np.random.randint(low=0, high=2*pad+1)
        return dst[y_coord:y_coord+h, x_coord:x_coord+w]

IMAGE_AUGS = {
    'cifar_train_aug': cifar_train_aug,
    'imagenet_train_aug': imagenet_train_aug, 
}