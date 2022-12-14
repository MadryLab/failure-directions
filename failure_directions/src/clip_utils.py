from nltk.corpus import words
from nltk.corpus import wordnet
from nltk.corpus import webtext, brown, gutenberg
import nltk
from nltk.corpus import wordnet as wn
import tqdm
import numpy as np
import torch
import pickle as pkl
import pprint
import json
import failure_directions.src.svm_utils as svm_utils
from failure_directions.src.label_maps import CLASS_DICT
import importlib.resources
import json
from failure_directions.src.imagenet_class_index import IN_CLASS_MAPPING


ADJECTIVES =  [None, 'white', 'blue', 'red', 'green', 'black', 'yellow', 'orange', 'brown']
DESCRIPTORS = ['group of', 'close-up', 'blurry', 'far away']
BACKGROUNDS =  [None, 'outside', 'inside', 'on a black background', 'on a white background', 'on a green background', 'on a blue background', 'on a brown background']

# IMAGENET
ANCESTOR_TO_PREPS = [
    [['reptile', 'amphibian'], [ 'in a tank', 'on the ground', 'on a rock', 'in the grass']],
    [['marsupial', 'swine', 'bovid', 'feline', 'canine', 'rodent'], [ 'in the grass', 'in a house', 'in the forest', 'on the ground', 'with a person']],
    [['fungus'], [ 'on the ground', 'in the grass']],
    [['food'], [ 'on a plate', 'on the ground', 'on a table', 'with a person']],
    [['geological_formation', 'train', 'mammal', 'structure', 'entity'], [ ]],
    [['wheeled_vehicle', 'bus', 'vehicle'], [ 'on the road', 'parked']],
    [['fruit', 'fruit', 'flower'], [ 'on a table', 'on the ground', 'on a tree', 'with a person', 'in the grass']],
    [['fish', 'aquatic_mammal'], [ 'in a tank', 'with a person', 'underwater']],
    [['person'], [ 'in a house', 'on a field']],
    [['equipment','instrumentality', 'appliance', 'container', 'fabric', 'covering', 'device', 'implement'], [ 'on a table', 'with a person', 'with a hand']],
    [['primate'], [ 'in a tree', 'on the ground', 'in the grass']],
    [['invertebrate'],  ['in the grass', 'on the ground', 'in a house']],
    [['bird'], [ 'in the air', 'on the ground', 'in a cage', 'in the grass', 'flying', 'perched']],
    [['aircraft'], [ 'in the air', 'on the ground']],
    [['furniture'], [ 'in a house']],
    [['vessel'], [ 'in the water', 'docked', 'in the ocean']],
]

def get_imagenet_config():
    class_indices = IN_CLASS_MAPPING
#     with open('src/imagenet_class_index.json', 'r') as f:
#         class_indices = json.load(f)
    synsets = {}
    for k, v in class_indices.items():
        ident = int(v[0][1:])
        synsets[int(k)] = wn.synset_from_pos_and_offset('n', ident)
    
    def get_hypernyms(curr_list, curr_synset):
        curr_list = curr_list + [curr_synset.name().split('.')[0]]
        for c in curr_synset.hypernyms():
            curr_list = get_hypernyms(curr_list, c)
        return curr_list
    
    ancestor_synsets = {k: get_hypernyms([], synsets[k]) for k in synsets.keys()}
    ancestor_preps = {}
    total_ancestor_list = []
    for ancestors, preps in ANCESTOR_TO_PREPS:
        for a in ancestors:
            total_ancestor_list.append(a)
            ancestor_preps[a] = preps
    
    imagenet_label_list = np.array([CLASS_DICT['ImageNet'][u].split(',')[0] for u in range(1000)])
    prepositions = {}
    nouns = {}
    for i in range(1000):
        ancestors = ancestor_synsets[i]
        ancestor = None
        for a in ancestors:
            if a in total_ancestor_list:
                ancestor = a
                break
        assert ancestor is not None
#         nouns[imagenet_label_list[i]] = [imagenet_label_list[i]]
        nouns[imagenet_label_list[i]] = [' '.join(ancestor.split('_'))]
        prepositions[imagenet_label_list[i]] = BACKGROUNDS + ancestor_preps[ancestor]
    return {
        'adjectives': ADJECTIVES + DESCRIPTORS,
        'prepositions': prepositions,
        'nouns': nouns
    }
        
        
IMAGENET_CLIP_CONFIG = get_imagenet_config()

CIFAR_CLIP_CONFIG = {
    'synset': {
        'dog': 'dog.n.01',
        'bird': 'bird.n.01',
        'automobile': 'car.n.01',
        'airplane': 'airplane.n.01',
        'truck': 'truck.n.01',
        'ship': ['boat.n.01', 'ship.n.01'],
        'frog': 'frog.n.01',
        'cat': 'cat.n.01',
        'horse': 'horse.n.01',
        'deer': ['deer.n.01', 'antelope.n.01']
    },
    'prepositions': {
        'dog': BACKGROUNDS + ['on the grass', 'in a house', 'in the snow', 'in the forest'],
        'cat': BACKGROUNDS + ['on the grass', 'in a house', 'in the snow', 'in the forest'],
        'bird': BACKGROUNDS + ['flying', 'perched', 'in the air', 'on the ground'],
        'horse': BACKGROUNDS + ['in a field', 'in the grass'],
        'airplane': BACKGROUNDS + ['flying', 'in the air', 'on the tarmac', 'on a road'],
        'truck': BACKGROUNDS + ['on the road', 'parked'],
        'automobile': BACKGROUNDS + ['on the road', 'parked'],
        'deer': BACKGROUNDS + ['in a field', 'in the grass', 'in the snow', 'in a forest'],
        'ship': BACKGROUNDS + ['in the ocean', 'docked', 'in the water', 'on the horizon'],
        'frog': BACKGROUNDS + ['in the grass', 'in a pond'],
    },
    'adjectives': ADJECTIVES,
}

CIFAR_SIMPLE_CONFIG = {
    'nouns': {
        'dog':['dog'],
        'bird': ['bird'],
        'automobile': ['automobile'],
        'airplane': ['airplane'],
        'truck': ['truck'],
        'ship': ['ship'],
        'frog': ['frog'],
        'cat': ['cat'],
        'horse': ['horse'],
        'deer':['deer'],
    },
    'adjectives': CIFAR_CLIP_CONFIG['adjectives'],
    'prepositions': CIFAR_CLIP_CONFIG['prepositions'],
}

CIFAR100_CLIP_CONFIG = {
    'nouns': {
            'aquatic mammal':	['beaver', 'dolphin', 'otter', 'seal', 'whale'],
            'fish':	['aquarium fish', 'flatfish', 'ray', 'shark', 'trout'],
            'flower':	['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
            'food container':	['bottle', 'bowl', 'can', 'cup', 'plate'],
            'fruit or vegetable':	['apple', 'mushroom', 'orange', 'pear', 'sweet pepper'],
            'household electrical device':	['clock', 'keyboard', 'lamp', 'telephone', 'television'],
            'household furniture':	['bed', 'chair', 'couch', 'table', 'wardrobe'],
            'insect':	['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
            'large carnivore':	['bear', 'leopard', 'lion', 'tiger', 'wolf'],
            'large man-made outdoor thing':	['bridge', 'castle', 'house', 'road', 'skyscraper'],
            'large natural outdoor scene':	['cloud', 'forest', 'mountain', 'plain', 'sea'],
            'large omnivores and herbivore':	['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
            'medium-sized mammal':	['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
            'non-insect invertebrate':	['crab', 'lobster', 'snail', 'spider', 'worm'],
            'person':	['baby', 'boy', 'girl', 'man', 'woman'],
            'reptile':	['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
            'small mammal':	['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
            'tree':	['maple tree', 'oak tree', 'palm tree', 'pine tree', 'willow tree'],
            'standard vehicle':	['bicycle', 'bus', 'motorcycle', 'pickup truck', 'train'],
            'specialized vehicle':	['lawn mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    },
    'prepositions': [None],
    'adjectives': [None],
}


CHESTXRAY_CONFIG = {
    'nouns': {
        "x-ray": ["x-ray"],
    },
    'adjectives': [None, "blurry", "close-up", "light", "dark"],
    'prepositions': [
        "facing forward",
        "facing sideways",
        "cropped",
    ]
}

CELEBA_CONFIG = {
    'nouns': {
        "person": ["person", "man", "woman"],
    },
    'adjectives': [None, "old", "young"],
    'prepositions': ["who has stubble",
     'who has arched eyebrows',
     'who has bags under their eyes',
     'who has bangs',
     'who has big lips',
     'who has a big nose',
     'who has black hair',
     'who has blond hair',
     'who has brown hair',
     'who has bushy eyebrows',
     'who has a double chin',
     'who is wearing eyeglasses',
     'who has a goatee',
     'who has gray hair',
     'who has heavy makeup on',
     'who has high cheekbones',
     'who has a mouth that is slightly open',
     'who has a mustache',
     'who has narrow eyes',
     'who has no beard',
     'who has an oval face',
     'who has a pointy nose',
     'who has a receding hairline',
     'who has rosy cheeks',
     'who has sideburns',
     'who has a smile',
     'who has straight hair',
     'who has wavy hair',
     'who has earrings',
     'who is wearing a hat',
     'who has a lipstick on',
     'who is wearing a necktie']
}
CONFIG_MAPPING = {
    'CIFAR10': CIFAR_CLIP_CONFIG,
    'CELEBA': CELEBA_CONFIG,
    'IMAGENET': IMAGENET_CLIP_CONFIG,
    'CIFAR100': CIFAR100_CLIP_CONFIG,
    'CIFAR10SIMPLE': CIFAR_SIMPLE_CONFIG,
    'CHESTXRAY': CHESTXRAY_CONFIG,
}


class CaptionGenerator:
    def __init__(self, label_list, clip_config):
        self.vocab = set([w.lower() for w in words.words()])
        self.clip_config = clip_config
        self.label_list = label_list
        self.master_list = self.get_nouns()

    def _get_all_hyponyms_base(self, d):
        hyponyms = d.hyponyms()
        full_set = set(hyponyms)
        full_set.add(d)
        for d2 in hyponyms:
            full_set = full_set.union(self._get_all_hyponyms_base(d2))
        return full_set

    def get_wordnet_set(self, synset_name):
        full_set = set([])
        for s in synset_name:
            full_set = full_set.union(self._get_all_hyponyms_base(wn.synset(s)))
        return full_set

    def get_all_hyponyms(self, synset_name):
        if not isinstance(synset_name, list):
            synset_name = [synset_name]
        full_set = self.get_wordnet_set(synset_name)
        output = set([])
        for s in full_set:
            lemma = s.lemmas()[0]
            name = ' '.join(lemma.name().lower().split('_'))
            if not s.name().endswith('n.01'):
                continue
            exclude = False
            if name not in self.vocab:
                for sub in name.split(' '):
                    if sub not in self.vocab:
                        print("excluding", name)
                        exclude = True
            if exclude:
                continue
            output.add(name)
        return output
    
    def get_nouns(self):
        if 'synset' in self.clip_config:
            synset_dict = self.clip_config['synset']
            master_list = {} 
            for k, v in synset_dict.items():
                master_list[k] = self.get_all_hyponyms(v)
        elif 'nouns' in self.clip_config:
            master_list = {}
            for k, v in self.clip_config['nouns'].items():
                master_list[k] = v
        else:
            master_list = {}
            for k in self.label_list:
                master_list[k] = [k]
        master_list['reference'] = self.label_list
        return master_list
    

    def _get_text_data(self, nouns, adjectives, prepositions):
        class_level_captions = []
        for c in list(nouns):
            for a in adjectives:
                for p in prepositions:
                    ending = ' '.join([u for u in [a, c, p] if u != None])
                    class_level_captions.append(f'a photo of a {ending}')
#         text_data = torch.tensor(svm_utils.evaluate_clip_text(class_level_captions))
        return class_level_captions
    
    def get_captions(self):
        adjectives, prepositions = self.clip_config['adjectives'], self.clip_config['prepositions']
        caption_maps = {}
        for k, v in tqdm.tqdm(self.master_list.items()):
            print(k)
            if k == 'reference':
                caption_maps[k] = self._get_text_data(v, [None], [None])
            else:
                if isinstance(prepositions, list):
                    preps = prepositions
                elif isinstance(prepositions, dict):
                    preps = prepositions[k]
                elif isinstance(prepositions, str):
                    preps = prepositions
                else:
                    preps = [None]
                text_datas = {
                    'all': self._get_text_data(v, adjectives, preps),
                    'noun': self._get_text_data(v, [None], [None]),
                    'adj': self._get_text_data([k], adjectives, [None]),
                    'prep': self._get_text_data([k], [None], preps),
                }
                caption_maps[k] = text_datas
        return caption_maps
    
def get_caption_set(caption_set_name):
    assert caption_set_name in ['IMAGENET', "CIFAR", 'CELEBA', 'CIFAR100', 'CIFARSIMPLE', 'CHESTXRAY']
    if caption_set_name == 'CELEBA':
        return CaptionGenerator(label_list=['person', 'person'], clip_config=CONFIG_MAPPING['CELEBA']).get_captions()
    if caption_set_name == 'CHESTXRAY':
        return CaptionGenerator(label_list=['x-ray', 'x-ray'], clip_config=CONFIG_MAPPING['CHESTXRAY']).get_captions()
    elif caption_set_name == 'CIFAR':
        cifar_label_list = np.array([CLASS_DICT['CIFAR'][u] for u in range(10)])
        return CaptionGenerator(label_list=cifar_label_list, clip_config=CONFIG_MAPPING['CIFAR10']).get_captions()
    elif caption_set_name == 'CIFARSIMPLE':
        cifar_label_list = np.array([CLASS_DICT['CIFAR'][u] for u in range(10)])
        return CaptionGenerator(label_list=cifar_label_list, clip_config=CONFIG_MAPPING['CIFAR10SIMPLE']).get_captions()
    elif caption_set_name == 'CIFAR100':
        singular_class_names = ['aquatic mammal', 'fish', 'flower', 'food container', 
                                'fruit or vegetable', 'household electrical device', 
                                'household furniture', 'insect', 'large carnivore', 
                                'large man-made outdoor thing', 'large natural outdoor scene', 
                                'large omnivores and herbivore', 'medium-sized mammal', 
                                'non-insect invertebrate', 'person', 'reptile', 'small mammal', 
                                'tree', 'standard vehicle', 'specialized vehicle']
        return CaptionGenerator(label_list=singular_class_names, clip_config=CONFIG_MAPPING['CIFAR100']).get_captions()
    elif caption_set_name == 'IMAGENET':
        imagenet_label_list = np.array([CLASS_DICT['ImageNet'][u].split(',')[0] for u in range(1000)])
        return CaptionGenerator(label_list=imagenet_label_list, clip_config=CONFIG_MAPPING['IMAGENET']).get_captions()
    else:
        assert False, "caption_set not found"
    
def normalize_embeddings(a):
    return a/torch.linalg.norm(a, dim=1, keepdims=True)

def cosine_distance(a, b):
    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return a_norm @ b_norm.T

def order_descriptions_l2(clip_data, text_data):
    clip_mean = clip_data.mean(dim=0).unsqueeze(0)
    l2 = torch.linalg.norm(clip_mean - text_data, dim=1).cpu().numpy()
    return l2


def order_descriptions_angle(mean_point, query_points):
    clip_mean = mean_point.mean(dim=0).unsqueeze(0)
    out = cosine_distance(clip_mean, query_points).squeeze(0).cpu().numpy()
    return out
