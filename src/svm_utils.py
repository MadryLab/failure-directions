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
import src.trainer as trainer_utils
import src.ffcv_utils as ffcv_utils
import torch.nn as nn
from torch.cuda.amp import autocast
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
# from src.trainer import SVMTrainer
import clip
import torchvision.transforms as transforms
import torchmetrics

class SVMPreProcessing(nn.Module):
    
    def __init__(self, mean=None, std=None, do_normalize=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.do_normalize = do_normalize
        
    def update_stats(self, latents):
        self.mean = latents.mean(dim=0)
        self.std = (latents - self.mean).std(dim=0)
    
    def normalize(self, latents):
        return latents/torch.linalg.norm(latents, dim=1, keepdims=True)
    
    def whiten(self, latents):
        return (latents - self.mean) / self.std
    
    def forward(self, latents):
        if self.mean is not None:
            latents = self.whiten(latents)
        if self.do_normalize:
            latents = self.normalize(latents)
        return latents
    
    def _export(self):
        return {
            'mean': self.mean,
            'std': self.std,
            'normalize': self.normalize
        }
    
    def _import(self, args):
        self.mean = args['mean']
        self.std = args['std']
        self.normalize = args['normalize']
            
    

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
        
def evaluate_robust_loader(dl, model, set_device=False, save_classes=0):
    latents = []
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    with torch.no_grad():
        preds, ys, spuriouses, confs, some_images, some_labels = [], [], [], [], [], []
        for x, y in tqdm(dl):
            if len(y) != len(x):
                x = x[:len(y)] # for drop_last
            inds_to_save_x = (y < save_classes)
            some_images.append(x[inds_to_save_x, ...].cpu())
            some_labels.append(y[inds_to_save_x].cpu())
            out, latent = model(x.to(device), with_image=False, with_latent=True)
            latents.append(latent.cpu())

            softmax_logits = nn.Softmax(dim=-1)(out)
            confs.append(softmax_logits[torch.arange(out.shape[0]), y].cpu())
            preds.append(out.argmax(-1).cpu())
            ys.append(y.cpu())
    if save_classes > 0:
        some_images = torch.cat(some_images).numpy()
        some_labels = torch.cat(some_labels).numpy()
    preds = torch.cat(preds).numpy()
    ys = torch.cat(ys).numpy()
    latents = torch.cat(latents).numpy()
    confs = torch.cat(confs)
    print('preds', preds.shape, 'ys', ys.shape)
    print("Accuracy", (preds == ys).astype(float).mean())
    return {
        'preds': preds, 
        'ys': ys, 
        'spuriouses': None, 
        'latents': latents,
        'confs': confs,
        'some_images': some_images,
        'some_labels': some_labels
    }
    
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
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

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

def evaluate_clip_features(dl, hparams, set_device=False):
    clip_model, preprocess = clip.load("ViT-B/32", device='cuda')
    clip_model = clip_model.eval()
    clip_normalize = preprocess.transforms[-1]
    preprocess_clip = transforms.Compose(
        [
            ffcv_utils.inv_norm(hparams['mean'], hparams['std']),
            torchvision.transforms.Resize((224, 224)),
            clip_normalize,
        ]
    )
    clip_activations = []
    with torch.no_grad():
        with autocast():
            for batch in tqdm(dl):
                x, y, _, _ = trainer_utils.unwrap_batch(batch, set_device=set_device)
                if len(y) != len(x):
                    x = x[:len(y)]
                image_features = clip_model.encode_image(preprocess_clip(x))
                clip_activations.append(image_features.cpu())
    out = torch.cat(clip_activations).float()
    return out
        
def evaluate_clip_text(captions):
    clip_model, preprocess = clip.load("ViT-B/32", device='cuda')
    clip_model = clip_model.eval()
    text = clip.tokenize(captions)
    ds = torch.utils.data.TensorDataset(text)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, drop_last=False, shuffle=False)
    clip_activations = []
    with torch.no_grad():
        for batch in tqdm(dl):
            caption = batch[0].cuda()
            text_features = clip_model.encode_text(caption)
            clip_activations.append(text_features.cpu())
    return torch.cat(clip_activations).float().numpy()
        

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

def train_per_class_svm(val_out, nc, balanced=True, split_and_search=False, cv=2):
    # if split_and_search is true, split our dataset into 50% svm train, 50% svm test
    # Then grid search over C = array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00])
    class_weight = 'balanced' if balanced else None        
    val_correct = (val_out['preds'] == val_out['ys']).astype(int)
    val_latent = val_out['latents']
    clfs, cv_scores = [], []
    for c in tqdm(range(nc)):
        mask = val_out['ys'] == c
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

def predict_per_class_svm(out_dict, nc, clfs, verbose=True, doSVC=True):
    N = len(out_dict['ys'])
    corr = out_dict['preds'] == out_dict['ys']
    ytrue = corr.astype(int)
    out_mask, out_decision = np.zeros(N), np.zeros(N)
    indiv_metrics, skipped_classes = [], []
    for c in tqdm(range(len(clfs))): #replaced nc
        mask = out_dict['ys'] == c
        if clfs[c] is not None:
            clf_out = clfs[c].predict(out_dict['latents'][mask])
            decision_out = clfs[c].decision_function(out_dict['latents'][mask])
            indiv_metrics.append({
                'accuracy': get_accuracy(ytrue=ytrue[mask], ypred=clf_out),
                'balanced_accuracy': get_balanced_accuracy(ytrue=ytrue[mask], ypred=clf_out),
            })
            out_mask[np.arange(N)[mask][clf_out == 1]] = 1
            out_decision[np.arange(N)[mask]] = decision_out
        else:
            skipped_classes.append(c)
    ypred = out_mask.astype(int)
    indiv_metrics = {k: np.array([u[k] for u in indiv_metrics]) for k in indiv_metrics[0].keys()}
    print('out_dict keys within predict_per_class_svm: ', out_dict.keys())
    
    metric_dict = {
        'accuracy': get_accuracy(ytrue=ytrue, ypred=ypred),
        'balanced_accuracy': get_balanced_accuracy(ytrue=ytrue, ypred=ypred), 
        'confusion_matrix': sklearn_metrics.confusion_matrix(ytrue, ypred),
        'indiv_accs': indiv_metrics,
        'ytrue': ytrue,
        'ypred': ypred,
        'decision_values': out_decision,
        'classes': out_dict['ys'],
        'skipped_classes': skipped_classes
    }
    if 'spuriouses' in out_dict.keys():
        metric_dict['spuriouses'] = out_dict['spuriouses']
    if 'indices' in out_dict.keys():
        metric_dict['indices'] = out_dict['indices']
    if verbose:
        pprint(metric_dict, indent=4)
    return out_mask, metric_dict

# =================================
# Neural Network
# =================================
def make_mlp(nin, nout, nhidden):
    return nn.Sequential(
        nn.Linear(nin, nhidden),
        nn.LeakyReLU(),
        nn.Linear(nhidden, nout)
    )


def get_mlp_tensor_dataset(out_dict, nc, shuffle=True):
    gt = torch.ones(len(out_dict['latents'])) * -1
    for c in range(nc):
        gt[out_dict['ys'] == c] = c
    correct = out_dict['ys'] == out_dict['preds']
    gt[correct] = nc

    dataset = torch.utils.data.TensorDataset(out_dict['latents'], gt.long())
    nn_dl = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=shuffle)
    return nn_dl

def get_correct_mlp_tensor_dataset(out_dict, shuffle=True):
    correct = out_dict['ys'] == out_dict['preds']

    dataset = torch.utils.data.TensorDataset(out_dict['latents'], correct.long())
    nn_dl = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=shuffle)
    return nn_dl

def train_mlp(nc, val_out, weight_decay=5e-4, lr=0.1, per_class=False, epochs=2000):
    if per_class:
        nn_dl = get_mlp_tensor_dataset(val_out, nc, shuffle=True)
    else:
        nn_dl = get_correct_mlp_tensor_dataset(val_out, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    nin = val_out['latents'].shape[-1]
    
    mlp_func = make_mlp
    if per_class:
        mlp_model = mlp_func(nin, nc+1, 128)
    else:
        mlp_model = mlp_func(nin, 2, 128)
    mlp_model = mlp_model.cuda()
    optimizer = optim.SGD(mlp_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    for epoch in range(1000):
        avg_loss = 0
        corr = 0
        total = 0
        for x, y in nn_dl:
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            out = mlp_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            corr += (out.argmax(-1) == y).float().sum().item()
            total += len(x)
        scheduler.step()
        if epoch % 50 == 0:
            print(epoch, scheduler.get_last_lr(), avg_loss/len(nn_dl), corr/total)
    return mlp_model

def evaluate_mlp_model(out_dict, mlp_model, nc, per_class=False):
    if per_class:
        nn_dl = get_mlp_tensor_dataset(out_dict, nc, shuffle=False)
    else:
        nn_dl = get_correct_mlp_tensor_dataset(out_dict, shuffle=False)
    preds = []
    ys = []
    with torch.no_grad():
        for x, y in nn_dl:
            x = x.cuda()
            y = y.cuda()
            out = mlp_model(x).argmax(-1)
            preds.append(out.cpu())
            ys.append(y.cpu())
    preds = torch.cat(preds)
    ys = torch.cat(ys)
    metric_dict = {
        'accuracy': (preds == ys).float().mean().item(),
        'confusion_matrix': sklearn_metrics.confusion_matrix(ys.numpy(), preds.numpy()),
    }
    pprint(metric_dict, indent=4)
    if per_class:
        return (preds == nc).int().numpy(), metric_dict
    else:
        return (preds == 1).int().numpy(), metric_dict

    
    
