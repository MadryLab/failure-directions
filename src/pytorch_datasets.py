import torch
import torchvision
import numpy as np
import os
from PIL import Image

        
class IndexedDataset():
    # Wrapper around dataset. Return the dataset and the index in the dataset
    def __init__(self, ds):
        self.ds = ds
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        return *self.ds[idx], idx
    
        
# CIFAR 100
CIFAR_100_CLASS_MAP = {
            'aquatic mammals':	['beaver', 'dolphin', 'otter', 'seal', 'whale'],
            'fish':	['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
            'flowers':	['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
            'food containers':	['bottle', 'bowl', 'can', 'cup', 'plate'],
            'fruit and vegetables':	['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
            'household electrical devices':	['clock', 'keyboard', 'lamp', 'telephone', 'television'],
            'household furniture':	['bed', 'chair', 'couch', 'table', 'wardrobe'],
            'insects':	['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
            'large carnivores':	['bear', 'leopard', 'lion', 'tiger', 'wolf'],
            'large man-made outdoor things':	['bridge', 'castle', 'house', 'road', 'skyscraper'],
            'large natural outdoor scenes':	['cloud', 'forest', 'mountain', 'plain', 'sea'],
            'large omnivores and herbivores':	['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
            'medium-sized mammals':	['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
            'non-insect invertebrates':	['crab', 'lobster', 'snail', 'spider', 'worm'],
            'people':	['baby', 'boy', 'girl', 'man', 'woman'],
            'reptiles':	['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
            'small mammals':	['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
            'trees':	['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
            'vehicles 1':	['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
            'vehicles 2':	['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}
class SuperCIFAR100(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.ds = torchvision.datasets.CIFAR100(**kwargs)
        self.classes = sorted(list(CIFAR_100_CLASS_MAP.keys()))
        self.subclasses = self.ds.classes
        self.reverse_map = {}
        for i, k in enumerate(self.classes):
            for v_ in CIFAR_100_CLASS_MAP[k]:
                self.reverse_map[self.subclasses.index(v_)] = i
        self.subclass_targets = self.ds.targets
        self.targets = [self.reverse_map[u] for u in self.ds.targets]
  
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x, self.reverse_map[y], y

#     CELEBA
    
class SpuriousAttributeCelebA(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        # spurious is 1 if blond, 2 if black hair, 0 if neither 
        self.ds = torchvision.datasets.CelebA(**kwargs)
        self.attr_names = self.ds.attr_names
        
        male_index = self.attr_names.index('Male')
        blond_hair_index = self.attr_names.index('Blond_Hair')
        black_hair_index = self.attr_names.index('Black_Hair')
        
        self.male_targets = self.ds.attr[:, male_index]
        self.blond_hair_targets = self.ds.attr[:, blond_hair_index]
        self.black_hair_targets = self.ds.attr[:, black_hair_index]
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        primary = self.male_targets[idx]
        
        is_blond_hair = self.blond_hair_targets[idx]
        is_black_hair = self.black_hair_targets[idx]
        if is_blond_hair:
            secondary = 1
        elif is_black_hair:
            secondary = 2
        else:
            secondary = 0
        return x, primary, torch.tensor(secondary)
    
class SpuriousAttributeCelebAAge(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.ds = torchvision.datasets.CelebA(**kwargs)
        self.attr_names = self.ds.attr_names
        
        young_index = self.attr_names.index('Young')
        male_index = self.attr_names.index('Male')
        
        self.young_targets = self.ds.attr[:, young_index]
        self.male_targets = self.ds.attr[:, male_index]
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        primary = self.young_targets[idx]
        secondary = self.male_targets[idx]
        return x, primary, secondary
        
        
#  MNIST
MNIST_IMG_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((32, 32))
])        

def create_val_split(train_targets, num_classes, split_amt=5):
    N = len(train_targets)
    val_indices = []
    for c in range(num_classes):
        to_add = torch.arange(N)[train_targets == c][::split_amt]
        val_indices.append(to_add)
    val_indices = torch.cat(val_indices)
    train_indices = torch.tensor([u for u in torch.arange(N) if u not in val_indices])
    indices_dict = {
        'val_indices': val_indices,
        'train_indices': train_indices,
    }
    return indices_dict

def get_unlabeled_indices(initial_train_targets, num_classes, fold, first_val_split=5):
    # write subsets
    indices_dict = create_val_split(initial_train_targets, num_classes, first_val_split)
    base_inds = indices_dict['train_indices']
    sub_indices_dict = create_val_split(initial_train_targets[base_inds], num_classes, fold)
    result_indices = {
        'train_indices': base_inds[sub_indices_dict['val_indices']],
        'unlabeled_indices': base_inds[sub_indices_dict['train_indices']],
        'val_indices': indices_dict['val_indices']
    }
    return result_indices
        
def generate_binary_mnist_ds(root='mnist_dataset', train=True):              
    def target_transform(y):
        if y < 5:
            return 0
        else:
            return 1
    ds = torchvision.datasets.MNIST(root, train=True, transform=MNIST_IMG_TRANSFORM, 
                                      target_transform=target_transform, download=True)
    ds.binary_targets = (ds.targets >= 5).int()
    return ds
            

class ColoredMNIST(torch.utils.data.Dataset):
    def __init__(self, ds, p_corr, noise=0.25, numpy_seed=0): #changed seed from -1 
        self.ds = ds
        if numpy_seed > 0:
            np.random.seed(numpy_seed)
        N = len(ds)
        vals = np.zeros(N)
        vals[self.get_fold(N, p_corr)] = 1
        print("percent agreeing", vals.mean())
        self.agree_spurious = vals
        
        vals = np.zeros(N)
        vals[self.get_fold(N, noise)] = 1
        print("percent flipped", vals.mean())
        self.flip_label = vals
            
    def get_fold(self, N, p):
        out_inds = np.arange(N)
        np.random.shuffle(out_inds)
        return out_inds[:int(N*p)]
        
    def apply_tint(self, channel_index, img):
        new_img = torch.zeros(3, *img.shape[1:])
        new_img[channel_index] = img
        return new_img
    
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, idx):
        x, y = self.ds[idx]
        digit = y
        
        agree = self.agree_spurious[idx]
        flip = self.flip_label[idx]
        if flip:
            y = 1-y
        tint = y if agree else 1-y
        img = self.apply_tint(tint, x)
        return img.half(), y, agree
    

    
class ChestXrayDataSet(torch.utils.data.Dataset):
    def __init__(self, split, root):
        if split == 'train':
            image_list_file = os.path.join(root,'labels/train_list.txt')
        elif split == 'test':
            image_list_file = os.path.join(root, 'labels/test_list.txt')
        elif split == 'val':
            image_list_file = os.path.join(root, 'labels/val_list.txt')
        else:
            assert False
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(root, 'images', image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        
    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]

        return image, np.array(label).astype(np.int32)

    def __len__(self):
        return len(self.image_names)