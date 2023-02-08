import sys
import os
import pickle as pkl
import torch
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import failure_directions.src.svm_utils as svm_utils
import failure_directions.src.visualization_utils as viz_utils
import failure_directions.src.ds_utils as ds_utils
from failure_directions.src.label_maps import CLASS_DICT
import failure_directions.src.clip_utils as clip_utils

if __name__ == "__main__":
    imagenet_label_list = np.array([CLASS_DICT['ImageNet'][u].split(',')[0] for u in range(1000)])
    beton_root = "/mnt/cfs/projects/correlated_errors/betons"
    experiment_root = "/mnt/cfs/projects/correlated_errors/experiments/imagenet"

    svm_name = "svm_imagenet"
    name = os.path.join(experiment_root, f"svm_checkpoints/{svm_name}.pt") # SVM output file
    svm_model_name = os.path.join(experiment_root, f"svm_checkpoints/{svm_name}_model.pkl") # SVM output file
    model_root = os.path.join(experiment_root, "models")
    model_ckpt = os.path.join(model_root, "vanilla_imagenet/version_1/checkpoints/checkpoint_last.pt")
    loss_upweight_root = os.path.join(experiment_root, "loss_vec_files")
    subset_root = os.path.join(experiment_root, "subset_index_files")

    processor = viz_utils.SVMProcessor(name, root=beton_root, checkpoint_path=model_ckpt, get_unlabeled=True)

    split = 'test'
    test_dv = processor.metrics[f'{split}_metrics']['decision_values']
    test_confs = processor.run_dict[split]['confs']
    test_class = processor.metrics[f'{split}_metrics']['classes'] 
    test_pred_correct = processor.metrics[f'{split}_metrics']['ypred']
    test_correct = processor.metrics[f'{split}_metrics']['ytrue']

    clip_analyzer = clip_utils.ClipAnalyzer(
        processor=processor, svm_model_name=svm_model_name, class_names=imagenet_label_list,
        clip_config_name='IMAGENET', do_normalize=True)

    def get_cdf(arr, K_range=None):
        out = []
        if K_range is None:
            K_range = np.arange(10, len(arr), 10)
        for K in K_range:
            out.append(arr[:K].mean())
        out = np.array(out)
        return out, K_range

    saved_caption_and_most_relevant_imgs = {}
    METHOD = 'CLASSIFY'
    all_results = {}
    for target_class in range(1000):
        print(processor.metrics['cv_scores'][target_class])
        print("performing classify captions on svm")
        all_results[target_class] = clip_analyzer.get_svm_style_top_K(target_class, 'all')

    torch.save(all_results, "imagenet_clip_results_5_20.pt")

    