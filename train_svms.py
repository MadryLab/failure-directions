import sys
import os
import yaml
import torch
import src.svm_utils as svm_utils
import argparse
import numpy as np
from src import model_utils
import src.trainer as trainer_utils
from src.config_parsing import ffcv_read_check_override_config, svm_read_check_override_config
from src.ffcv_utils import get_training_loaders
import pprint
import pickle as pkl


METHODS = ['SVM', 'MLP']

if __name__ == "__main__":
    parser = argparse.ArgumentParser('demo')
    parser.add_argument('-c', '--config', required=True, type=str, help='dataset config file')
    parser.add_argument('-s', '--svm-config', required=True, type=str, help='SVM config file')
    parser.add_argument('--indices-file', type=str)
    parser.add_argument('--root', type=str, default='/mnt/cfs/projects/correlated_errors/betons', help='path to betons')
    parser.add_argument('--model-path', required=True, type=str, help='model path')
    parser.add_argument('--out-file', required=True, type=str, help='output path')
    parser.add_argument('--unlabeled', action='store_true')
    # chest_xray args
    parser.add_argument('--bce', action='store_true', help='bce model')
    parser.add_argument('--bce-class', type=int, default=7)
    parser.add_argument('--spurious', action='store_true', help='If present, include spurious label in the pipeline (assume available in beton)')
    args = parser.parse_args()
    
    
    # --------- READ THE CONFIGS -------------
    with open(args.config, 'r') as file:
        hparams = yaml.safe_load(file)
    hparams = ffcv_read_check_override_config(hparams)
    hparams['drop_last'] = False
    hparams['shuffle'] = False
    print("=========== Current Config ==================")
    pprint.pprint(hparams, indent=4)
    set_device = ('cmnist' in hparams.keys() and hparams['cmnist'])
    
    with open(args.svm_config, 'r') as file:
        svm_hparams = yaml.safe_load(file)
    svm_hparams = svm_read_check_override_config(svm_hparams)
    print("=========== Current SVM Config ==================")
    pprint.pprint(svm_hparams, indent=4)
    
    if args.spurious:
        pipeline_subset=['image', 'label', 'spurious', 'index']
    else:
        pipeline_subset=['image', 'label', 'index']

    # --------- GET THE LOADERS -------------
    indices_dict = torch.load(args.indices_file) if args.indices_file else None
    common_args = {
        "hparams": hparams,
        "pipeline_subset": pipeline_subset,
        "get_unlabeled": args.unlabeled,
        "root": args.root,
        "indices_dict": indices_dict,
    }
    if args.unlabeled:
        train_loader, val_loader, test_loader, unlabeled_loader = get_training_loaders(**common_args)
        loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader, 'unlabeled': unlabeled_loader}
    else:
        train_loader, val_loader, test_loader = get_training_loaders(**common_args)
        loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    
    # --------- LOAD THE MODEL -------------
    build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
    model = model_utils.load_model(args.model_path, build_fn)
    device = torch.device('cuda') if  torch.cuda.is_available() else torch.device('cpu')
    model = model.eval().to(device)    
    if args.bce:
        hparams['num_classes'] = 2
        
    num_classes = hparams['num_classes']
    
    # --------- EVALUATE THE MODEL -------------
    if args.bce:
        # get the BCE threshold
        val_out, bce_threshold = svm_utils.evaluate_bce_loader(loaders['val'], 
                                                               model, 
                                                               class_index=args.bce_class,
                                                               bce_threshold=None)
        out_dict = {'val': val_out}
        for split, loader in loaders.items():
            if split not in out_dict:
                out_dict[split], _ = svm_utils.evaluate_bce_loader(loader,
                                                                  model, 
                                                                  class_index=args.bce_class, 
                                                                  bce_threshold=bce_threshold)
    else:
        out_dict = {
            split: svm_utils.evaluate_loader(loader, model, set_device=set_device) 
            for split, loader in loaders.items()
        }
    metric_dict = {'args': vars(args)}
    if args.bce:
        metric_dict['bce_threshold'] = bce_threshold

    # --------- GET THE EMBEDDINGS -------------
    if svm_hparams['embedding'] == 'inception':
        print("using inception features")
        third_party_dict = {split: svm_utils.evaluate_inception_features(loader) for split, loader in loaders.items()}
    elif svm_hparams['embedding'] == 'clip':
        print("using clip features")
        third_party_dict = {
            split: svm_utils.evaluate_clip_features(loader, hparams, set_device=set_device) for split, loader in loaders.items()}
    else:
        third_party_dict = {
            split: out[split]['latents'] for split in out.keys()
        }
    
    # --------- PRE PROCESS --------------------
    pre_process = svm_utils.SVMPreProcessing(do_normalize=svm_hparams['normalize'])
    pre_process.update_stats(third_party_dict['train'])
    metric_dict['stats'] = pre_process._export()
    with torch.no_grad():
        third_party_dict = {
            k: pre_process(v) for k, v in third_party_dict.items()
        }
    for k in out_dict.keys():
        out_dict[k]['latents'] = third_party_dict[k]
        
    # --------- Convert to numpy arrays --------
    for k in out_dict.keys():
        for k2 in out_dict[k].keys():
            if torch.is_tensor(out_dict[k][k2]):
                out_dict[k][k2] = out_dict[k][k2].numpy()

    # --------- TRAIN SVM --------------------
    val_out = out_dict['val']
    if svm_hparams['method'] == 'SVM':
        fit_args = svm_hparams['svm_args']
        clfs, cv_scores = svm_utils.train_per_class_svm(val_out, num_classes,
                                                        split_and_search=fit_args['split_and_search'], 
                                                        balanced=fit_args['balanced'])
        metric_dict['cv_scores'] = cv_scores
        for name, ds in out_dict.items():
            print("=================", name, "=====================")
            errors, metric = svm_utils.predict_per_class_svm(ds, num_classes, clfs)
            metric_dict[f"{name}_metrics"] = metric
            metric_dict[f"predicted_{name}_errors"] = 1 - errors # save 0 if correct, 1 otherwise
            
        with open(f"{args.out_file}_model.pkl", 'wb') as f:
            pkl.dump(clfs, f)
    elif svm_hparams['method'] == 'MLP':
        mlp_model = svm_utils.train_mlp(num_classes, val_out, lr=args.mlp_lr, per_class=args.per_class)
        print("=========== VAL =======")
        for name, ds in out_dict.items():
            errors, metric = svm_utils.evaluate_mlp_model(ds, mlp_model, num_classes, per_class=args.per_class)
            metric_dict[f"{name}_metrics"] = metric
            metric_dict[f"predicted_{name}_errors"] = 1 - errors # save 0 if correct, 1 otherwise
    else:
        raise NotImplemented()
    torch.save(metric_dict, f"{args.out_file}.pt")