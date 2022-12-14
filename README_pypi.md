# Distilling Model Failures as Directions in Latent Space
Python package for automatically finding failure modes in your dataset.

The paper: **[Distilling Model Failures as Directions in Latent Space](https://arxiv.org/abs/2206.14754)** and corresponding [blog post](https://gradientscience.org/failure-directions/) 

See the [github repository](https://github.com/MadryLab/failure-directions) for documentation and examples!


## Getting Started
Install using pip, or clone our repository.
```
pip install failure-directions
```
 
Basic usage. Here, `svm_decision_values` are the alignment of the images to the identified failure modes. 
```
import failure_directions
# let hparams contain mean and std for dataset
# let loaders contain a dict of train, test, val loaders.
# let split_gts contain the ground truth labels for each split
# let split_preds contain the predictions for each split

# Load CLIP features
clip_processor = failure_directions.CLIPProcessor(ds_mean=hparams['mean'], ds_std=hparams['std'])
clip_features = {}
for split, loader in loaders.items():
    clip_features[split] = clip_processor.evaluate_clip_images(loader)
    
# Fit SVM
svm_fitter = failure_directions.SVMFitter()
svm_fitter.set_preprocess(clip_features['train'])
cv_scores = svm_fitter.fit(preds=split_preds['val'], ys=split_gts['val'], latents=clip_features['val'])

# Get SVM decision values
svm_predictions = {}
svm_decision_values = {}
for split, loader in loaders.items():
    mask, dv = svm_fitter.predict(ys=split_gts[split], latents=clip_features[split], compute_metrics=False)
    svm_predictions[split] = mask
    svm_decision_values[split] = dv
```


## Citation
If use this package, please cite the corresponding paper
```
@inproceedings{jain2022distilling,
   title = {Distilling Model Failures as Directions in Latent Space},
   author = {Saachi Jain and Hannah Lawrence and Ankur Moitra and Aleksander Madry}, 
   booktitle = {ArXiv preprint arXiv:2206.14754},
   year = {2022}
}
```

## Maintainers
[Saachi Jain](https://twitter.com/saachi_jain_)<br>
[Hannah Lawrence](https://twitter.com/HLawrenceCS)

