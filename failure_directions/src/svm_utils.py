import yaml
import sys
import torch
from tqdm import tqdm
import os
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.model_selection import cross_val_score
import torch.nn as nn
import torch.optim as optim
import torchvision
import failure_directions.src.trainer as trainer_utils
import failure_directions.src.ffcv_utils as ffcv_utils
import torch.nn as nn
from torch.cuda.amp import autocast
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
# from src.trainer import SVMTrainer
import clip
import torchvision.transforms as transforms
import torchmetrics
import sklearn.neural_network

class SVMPreProcessing(nn.Module):
    
    def __init__(self, mean=None, std=None, do_normalize=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.do_normalize = do_normalize
        
    def update_stats(self, latents):
        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)
        self.mean = latents.mean(dim=0)
        self.std = (latents - self.mean).std(dim=0)
    
    def normalize(self, latents):
        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)
        return latents/torch.linalg.norm(latents, dim=1, keepdims=True)
    
    def whiten(self, latents):
        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)
        return (latents - self.mean) / self.std
    
    def forward(self, latents):
        if not torch.is_tensor(latents):
            latents = torch.tensor(latents)
        if self.mean is not None:
            latents = self.whiten(latents)
        if self.do_normalize:
            latents = self.normalize(latents)
        return latents
    
    def _export(self):
        return {
            'mean': self.mean,
            'std': self.std,
            'normalize': self.do_normalize
        }
    
    def _import(self, args):
        self.mean = args['mean']
        self.std = args['std']
        self.do_normalize = args['normalize']
            
    

class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        #self.inception_network = torchvision.models.inception_v3(pretrained=True)
        self.inception_network = torchvision.models.inception_v3(pretrained=False)
        self.inception_network.load_state_dict(torch.load(torch.hub.get_dir() + '/checkpoints/inception_v3_google-1a9a5a14.pth'))
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        # x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations



def read_yaml(yaml_file):
    with open(yaml_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
      
    
def sweep_f1_scores(logits, y):
    thresholds = np.arange(0.01, 1, 0.01)
    f1_scores = []
    for threshold in tqdm(thresholds):
        f1 = torchmetrics.F1Score(threshold=threshold)
        f1_scores.append(f1(logits, y.int()).item())
    print(thresholds)
    print(f1_scores)
    return thresholds[np.argmax(f1_scores)]
        
    
def evaluate_bce_loader(dl, model, class_index, set_device=False, bce_threshold=None):
    latents = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    def print_out(m, x_):
        latents.append(x_[0].cpu())
    handle = getattr(model, model._last_layer_str).register_forward_pre_hook(print_out)
    
    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        with autocast():
            all_logits, ys, spuriouses, indices = [], [], [], []
            for batch in tqdm(dl):
                x, y, spurious, idx = trainer_utils.unwrap_batch(batch, set_device=set_device)
                y = y[:, class_index]
                if len(y) != len(x):
                    x = x[:len(y)] # for drop_last
                out = model(x)
                logits = sigmoid(out)[:, class_index] 
                all_logits.append(logits.cpu())
                ys.append(y.cpu())
                if spurious is not None:
                    spuriouses.append(spurious.cpu())
                else:
                    spuriouses.append(None)
                indices.append(idx.cpu())
    all_logits = torch.cat(all_logits)
    ys = torch.cat(ys)
    
    if bce_threshold is None:
        bce_threshold = sweep_f1_scores(all_logits, ys)
        print(bce_threshold)
        
    preds = (all_logits > bce_threshold).int().cpu()
    confs = torch.zeros_like(all_logits)
    confs[ys==1] = all_logits[ys==1]
    confs[ys==0] = 1-all_logits[ys==0]

    latents = torch.cat(latents)
    if spuriouses[0] is not None:
        spuriouses = torch.cat(spuriouses)
    else:
        spuriouses = None
    indices = torch.cat(indices)
    print("Accuracy", (preds == ys).float().mean().item())
    handle.remove()
    return {
        'preds': preds, 
        'ys': ys, 
        'spuriouses': spuriouses, 
        'latents': latents,
        'confs': confs,
        'indices': indices,
    }, bce_threshold

def evaluate_loader(dl, model, robust_model=False, set_device=False, save_classes=0, save_pred_probs=False):
    latents = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def print_out(m, x_):
        latents.append(x_[0].cpu())
    handle = getattr(model, model._last_layer_str).register_forward_pre_hook(print_out)
    with torch.no_grad():
        with autocast():
            preds, ys, spuriouses, confs, indices, pred_probs = [], [], [], [], [], []
            for batch in tqdm(dl):
                x, y, spurious, idx = trainer_utils.unwrap_batch(batch, set_device=set_device)
                if len(y) != len(x):
                    x = x[:len(y)] # for drop_last
                out = model(x)
                softmax_logits = nn.Softmax(dim=-1)(out)
                if save_pred_probs:
                    pred_probs.append(softmax_logits.cpu())
                confs.append(softmax_logits[torch.arange(out.shape[0]), y].cpu())
                preds.append(out.argmax(-1).cpu())
                ys.append(y.cpu())
                if spurious is not None:
                    spuriouses.append(spurious.cpu())
                else:
                    spuriouses.append(None)
                indices.append(idx.cpu())
    if save_pred_probs:
        pred_probs = torch.cat(pred_probs)
    preds = torch.cat(preds)
    ys = torch.cat(ys)
    latents = torch.cat(latents)
    confs = torch.cat(confs)
    if spuriouses[0] is not None:
        spuriouses = torch.cat(spuriouses).cpu()
    else:
        spuriouses = None
    print("Accuracy", (preds == ys).float().mean().item())
    handle.remove()
    indices = torch.cat(indices)
    return {
        'preds': preds, 
        'ys': ys, 
        'spuriouses': spuriouses, 
        'latents': latents,
        'confs': confs,
        'indices': indices,
        'pred_probs': pred_probs,
    }

def evaluate_inception_features(dl, set_device=False):
    model = PartialInceptionNetwork()
    model = model.eval().cuda()
    resize = torchvision.transforms.Resize((299, 299))
    inception_activations = []
    with torch.no_grad():
        with autocast():
            for batch in tqdm(dl):
                x, y, _, _ = trainer_utils.unwrap_batch(batch, set_device=set_device)
                if len(y) != len(x):
                    x = x[:len(y)]
                out = model(resize(x))
                inception_activations.append(out.cpu())
    return torch.cat(inception_activations)        

def get_accuracy(ytrue, ypred):
    ytrue = torch.tensor(ytrue)
    ypred = torch.tensor(ypred)
    return float((ytrue == ypred).float().mean())

def get_balanced_accuracy(ytrue, ypred):
    accs = []
    for c in range(2):
        mask = ytrue == c
        if mask.astype(int).sum() == 0:
            continue
        accs.append(get_accuracy(ytrue[mask], ypred[mask]))
    return np.mean(accs)


# =================================
# Per class SVM
# =================================

def fit_svc_svm(C, class_weight, x, gt, cv=2):
    scorer = sklearn_metrics.make_scorer(sklearn_metrics.balanced_accuracy_score)
    clf = SVC(gamma='auto', kernel='linear', C=C, class_weight=class_weight)
    cv_scores = cross_val_score(clf, x, gt, cv=cv, scoring=scorer)
    cv_scores = np.mean(cv_scores)
    clf.fit(x, gt)
    return clf, cv_scores

def fit_svc_mlp(x, gt, svm_args={}):
    #scorer = sklearn_metrics.make_scorer(sklearn_metrics.balanced_accuracy_score)
    
    # make hidden layer based on feature dimension of x
    # batch_size = first dimension
    input_dim = x.shape[1]
    hidden_layer_size = svm_args['hidden_layer_size']
    if 'first_layer' in svm_args and not svm_args['first_layer']:
        hidden_layer_sizes =(hidden_layer_size, )
    else:
        hidden_layer_sizes=(input_dim, hidden_layer_size, )
    clf = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=800)
    clf.fit(x, gt)
    return clf, 0 

def train_per_class_svm(latents, ys, preds, balanced=True, split_and_search=False, cv=2):
    # if split_and_search is true, split our dataset into 50% svm train, 50% svm test
    # Then grid search over C = array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00])
    nc = ys.max() + 1
    class_weight = 'balanced' if balanced else None        
    val_correct = (preds == ys).astype(int)
    val_latent = latents
    clfs, cv_scores = [], []
    for c in tqdm(range(nc)):
        mask = ys == c
        x, gt = val_latent[mask], val_correct[mask]
        if split_and_search:
            best_C, best_cv, best_clf = 1, -np.inf, None
            for C_ in tqdm(np.logspace(-6, 0, 7, endpoint=True)):
                clf, cv_score = fit_svc_svm(C=C_, x=x, gt=gt, class_weight=class_weight, cv=cv)
                if cv_score > best_cv:
                    best_cv = cv_score
                    best_C = C_
                    best_clf = clf
            clfs.append(best_clf)
            cv_scores.append(best_cv)
        else:
            clf, cv_score = fit_svc_svm(C=C_, x=x, gt=gt, class_weight=class_weight, cv=cv)
            clfs.append(clf)
    return clfs, cv_scores

def train_per_class_mlp(latents, ys, preds, balanced=True, split_and_search=False, cv=2, hidden_layer_size=100, svm_args={}):
    # if split_and_search is true, split our dataset into 50% svm train, 50% svm test
    # Then grid search over C = array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00])
    nc = ys.max() + 1
    #class_weight = 'balanced' if balanced else None        
    val_correct = (preds == ys).astype(int)
    val_latent = latents
    clfs, cv_scores = [], []
    for c in tqdm(range(nc)):
        mask = ys == c
        x, gt = val_latent[mask], val_correct[mask]
        clf, cv_score = fit_svc_mlp(x=x, gt=gt, svm_args=svm_args) 
        clfs.append(clf)
    return clfs, cv_scores

def train_per_class_model(latents, ys, preds, balanced=True, split_and_search=False, cv=2, method='SVM', svm_args={}):
    if method == 'SVM':
        clfs, cv_scores = train_per_class_svm(latents=latents, ys=ys, preds=preds, balanced=balanced, split_and_search=split_and_search, cv=cv)
    else:
        clfs, cv_scores = train_per_class_mlp(latents=latents, ys=ys, preds=preds, balanced=balanced, split_and_search=split_and_search, cv=cv, svm_args=svm_args)
    return clfs, cv_scores

def predict_per_class_svm(latents, ys, clfs, preds=None, aux_info={}, verbose=True, compute_metrics=True, method='SVM'):
    N = len(ys)
    out_mask, out_decision = np.zeros(N), np.zeros(N)
    skipped_classes = []
    if compute_metrics:
        assert preds is not None
        corr = preds == ys
        ytrue = corr.astype(int)
        indiv_metrics = []
    
    for c in tqdm(range(len(clfs))): #replaced nc
        mask = ys == c
        if clfs[c] is not None and (len(mask[mask]) > 0):
            clf_out = clfs[c].predict(latents[mask])
            if method == 'SVM':
                decision_out = clfs[c].decision_function(latents[mask])
            else:
                decision_out = clfs[c].predict_proba(latents[mask])[:,0]
            if compute_metrics:
                indiv_metrics.append({
                    'accuracy': get_accuracy(ytrue=ytrue[mask], ypred=clf_out),
                    'balanced_accuracy': get_balanced_accuracy(ytrue=ytrue[mask], ypred=clf_out),
                })
            out_mask[np.arange(N)[mask][clf_out == 1]] = 1
            out_decision[np.arange(N)[mask]] = decision_out
        else:
            skipped_classes.append(c)
    ypred = out_mask.astype(int)
    if compute_metrics:
        indiv_metrics = {k: np.array([u[k] for u in indiv_metrics]) for k in indiv_metrics[0].keys()}    
        metric_dict = {
            'accuracy': get_accuracy(ytrue=ytrue, ypred=ypred),
            'balanced_accuracy': get_balanced_accuracy(ytrue=ytrue, ypred=ypred), 
            'confusion_matrix': sklearn_metrics.confusion_matrix(ytrue, ypred),
            'indiv_accs': indiv_metrics,
            'ytrue': ytrue,
            'ypred': ypred,
            'decision_values': out_decision,
            'classes': ys,
            'skipped_classes': skipped_classes,
            **aux_info
        }
        if verbose:
            pprint(metric_dict, indent=4)
        return out_mask, out_decision, metric_dict
    else:
        return out_mask, out_decision


def predict_per_class_model(latents, ys, clfs, preds=None, aux_info={}, verbose=True, compute_metrics=True, method='SVM'):
    return predict_per_class_svm(latents=latents, ys=ys, clfs=clfs, preds=preds, aux_info=aux_info, verbose=verbose, compute_metrics=compute_metrics, method=method)
