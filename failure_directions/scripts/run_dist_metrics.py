import sys
import os
import yaml
import torch
import argparse
import numpy as np
import pprint
import pickle as pkl
from tqdm import tqdm

from failure_directions.src import model_utils
from failure_directions.src import ffcv_utils
import failure_directions.src.trainer as trainer_utils
from failure_directions.src.config_parsing import ffcv_read_check_override_config
from failure_directions.src.ffcv_utils import get_training_loaders
from failure_directions.src import ae_utils
from failure_directions.src import svm_utils
from failure_directions.src.dist_shift_utils import MMD


METHODS = ['SVM', 'MLP']

if __name__ == "__main__":
    parser = argparse.ArgumentParser('demo')
    parser.add_argument('-c', '--config', required=True, type=str, help='dataset config file')
    parser.add_argument('--model-path', required=True, type=str)
    parser.add_argument('--method', type=str, default='SVM')
    parser.add_argument('--out-file', type=str, default='dump')

    parser.add_argument('--inception', action='store_true')
    parser.add_argument('--autoencoder', type=str, default=None)
    parser.add_argument('--uae', action='store_true', help='use encodings from an untrained autoencoder')

    parser.add_argument('--dist_metric', type=str, default='MMD', help='which distance metric to use, defaults to MMD')

    parser.add_argument('--shuffles', type=int, default=10, help='how many label shufflings to try, for each dataset and test combination') 

    parser.add_argument('--spurious', action='store_true', help='run distribution test on the spurious feature, if available')
    parser.add_argument('--dist-per-class', action='store_true', help='run distribution test on each class')
    parser.add_argument('--dist-pool-classes', action='store_true', help='run distribution test on the whole val set at once')
    parser.add_argument('--dist-two-classes', action='store_true', help='run distribution test on two distinct classes (probably to be used as sanity check)')
    
    # MLP Args
    parser.add_argument('--mlp-lr', type=float, default=0.01,  help='LR for MLP')
    parser.add_argument('--per-class', action='store_true', help='use per class outputs for MLP')
    
    # SVM Args
    parser.add_argument('--split-and-search', action='store_true', help='search for right lambda for SVM')
    parser.add_argument('--balanced', action='store_true', help='balance classes for SVM')
    parser.add_argument('--lamb', default=1.0, help='C value to use for SVM')
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        hparams = yaml.safe_load(file)
    hparams = ffcv_read_check_override_config(hparams)
    hparams['drop_last'] = False
    hparams['shuffle'] = False
    print("=========== Current Config ==================")
    pprint.pprint(hparams, indent=4)
    train_loader, val_loader, test_loader = get_training_loaders(hparams)
    
    
    build_fn = model_utils.BUILD_FUNCTIONS[hparams['arch_type']]
    model = model_utils.load_model(args.model_path, build_fn)
    model = model.eval().cuda()
    
    assert args.method in METHODS
    
    num_classes = hparams['num_classes']
    
    train_out = svm_utils.evaluate_loader(train_loader, model)
    val_out = svm_utils.evaluate_loader(val_loader, model)
    test_out = svm_utils.evaluate_loader(test_loader, model)
    
    if args.inception:
        print("using inception features")
        train_inception = svm_utils.evaluate_inception_features(train_loader)
        val_inception = svm_utils.evaluate_inception_features(val_loader)
        test_inception = svm_utils.evaluate_inception_features(test_loader)
        train_out['latents'] = train_inception
        val_out['latents'] = val_inception
        test_out['latents'] = test_inception

    if args.autoencoder is not None or args.uae:
        print("using autoencoder on train and val set for features")
        invTrans = ffcv_utils.inv_norm(hparams['mean'], hparams['std'])
        ae_utils.train_autoencoder(ae_utils.itr_merge(train_loader, val_loader), autoencoder_name=args.autoencoder, invTrans=invTrans, do_train=(not args.uae))
        train_ae = ae_utils.evaluate_autoencoder_features(dl=train_loader, autoencoder_name=args.autoencoder)
        val_ae = ae_utils.evaluate_autoencoder_features(dl=val_loader, autoencoder_name=args.autoencoder)
        test_ae = ae_utils.evaluate_autoencoder_features(dl=test_loader, autoencoder_name=args.autoencoder)
        train_out['latents'] = train_ae
        val_out['latents'] = val_ae
        test_out['latents'] = test_ae


    val_correct = (val_out['preds'] == val_out['ys']).astype(int)
    val_latent = val_out['latents']
    val_spurious = val_out['spuriouses']

    if args.dist_metric == 'MMD':
        dist_func = MMD

    if args.spurious:
        label_for_dist = val_spurious
    else:
        label_for_dist = val_correct

    per_class_res = {}
    if args.dist_per_class:
        for c in tqdm(range(num_classes)):
            class_res = {}
            mask = val_out['ys'] == c
            print('val_latent', val_latent.shape, 'mask', mask[0:5], mask.shape)
            class_latents = val_latent[mask]
            #gt = val_correct[mask]

            within_mask_correct = np.array(label_for_dist[mask], dtype='bool') #np.logical_and(mask.bool(), val_correct.bool())
            within_mask_incorrect = np.logical_not(within_mask_correct) #mask_and_incorrect = np.logical_and(mask.bool(), np.logical_not(val_correct.bool()))
            
            latents1 = class_latents[within_mask_correct]
            latents2 = class_latents[within_mask_incorrect]
            dist = dist_func(latents1, latents2)
            class_res['dist'] = dist

            if args.shuffles > 0:
                shuf_dists = []
                for shuf in range(args.shuffles):
                    dist = dist_func(latents1, latents2, shuffle=True)
                    shuf_dists.append(dist)
            class_res['shuffled_dists'] = np.array(shuf_dists)
            per_class_res[str(c)] = class_res

    pooled_res = {}
    if args.dist_pool_classes:
        within_mask_correct = np.array(label_for_dist, dtype='bool') 
        within_mask_incorrect = np.logical_not(within_mask_correct)

        latents1 = val_latent[within_mask_correct]
        latents2 = val_latent[within_mask_incorrect]
        dist = dist_func(latents1, latents2)
        pooled_res['dist'] = dist

        if args.shuffles > 0:
            shuf_dists = []
            for shuf in range(args.shuffles):
                dist = dist_func(latents1, latents2, shuffle=True)
                shuf_dists.append(dist)
        pooled_res['shuffled_dists'] = np.array(shuf_dists)

    two_class_res = {}
    if args.dist_two_classes:
        latents1 = val_latent[val_out['ys']==0]
        latents2 = val_latent[val_out['ys']==1] 
        dist = dist_func(latents1, latents2)
        two_class_res['dist'] = dist

        if args.shuffles > 0:
            shuf_dists = []
            for shuf in range(args.shuffles):
                dist = dist_func(latents1, latents2, shuffle=True)
                shuf_dists.append(dist)
        two_class_res['shuffled_dists'] = np.array(shuf_dists)

    all_dist_res = {'per_class': per_class_res, 'all_classes': pooled_res, 'two_class_compare': two_class_res}
    with open(f"{args.out_file}_dist_metrics.pkl", 'wb') as f:
        pkl.dump(all_dist_res, f)


    if args.method == 'SVM':
        val2 = None
        if args.split_and_search:
            N = len(val_out['ys'])
            train_fold = np.arange(N)[::2]
            val_fold = np.arange(N)[1::2]
            val1 = {k: v[train_fold] for k,v in val_out.items() if v is not None}
            val2 = {k: v[val_fold] for k,v in val_out.items() if v is not None}
            clfs = svm_utils.train_per_class_svm(val1,
                                                 num_classes,
                                                 split_and_search=args.split_and_search, 
                                                 balanced=args.balanced, 
                                                 val2=val2)
            print("=========== VAL =======")
            _, val_metrics = svm_utils.predict_per_class_svm(val1, num_classes, clfs)
            print("=========== VAL2 =======")
            _, val2_metrics = svm_utils.predict_per_class_svm(val2, num_classes, clfs)
        else:
            clfs = svm_utils.train_per_class_svm(val_out, num_classes, C=args.lamb,
                                                 split_and_search=args.split_and_search, balanced=args.balanced)
            
            print("=========== VAL =======")
            _, val_metrics = svm_utils.predict_per_class_svm(val_out, num_classes, clfs)
            val2_metrics = None
            
        print("=========== TEST =======")
        _, test_metrics = svm_utils.predict_per_class_svm(test_out, num_classes, clfs)
        print("=========== TRAIN =======")
        predicted_train_errors, train_metrics = svm_utils.predict_per_class_svm(train_out, num_classes, clfs)
        with open(f"{args.out_file}_model.pkl", 'wb') as f:
            pkl.dump(clfs, f)
    elif args.method == 'MLP':
        mlp_model = svm_utils.train_mlp(num_classes, val_out, lr=args.mlp_lr, per_class=args.per_class)
        print("=========== VAL =======")
        _, val_metrics = svm_utils.evaluate_mlp_model(val_out, mlp_model, num_classes, per_class=args.per_class)
        print("=========== TEST =======")
        _, test_metrics = svm_utils.evaluate_mlp_model(test_out, mlp_model, num_classes, per_class=args.per_class)
        print("=========== TRAIN =======")
        predicted_train_errors, train_metrics = svm_utils.evaluate_mlp_model(train_out, mlp_model, num_classes, per_class=args.per_class)
    else:
        raise NotImplemented()
    
    predicted_train_errors = 1 - predicted_train_errors # save 0 if correct, 1 if incorrect
    out_result = {
        'args': vars(args),
        'val_metrics': val_metrics,
        'val2_metrics': val2_metrics,
        'test_metrics': test_metrics,
        'train_metrics': train_metrics,
        'predicted_train_errors': predicted_train_errors,  
    }
    torch.save(out_result, f"{args.out_file}.pt")