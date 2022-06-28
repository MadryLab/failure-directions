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
import torch.nn as nn
import torch.optim as optim
import torchvision
import src.trainer as trainer_utils
import torch.nn as nn
from torch.cuda.amp import autocast
from pprint import pprint

def pairwise_exp_kernel(pairwise_squared_dists, sigma):
	return torch.exp(pairwise_squared_dists*-1/sigma)

def get_pairwise_squared_dists(latents):
	# Brute force for simplicity. speed up later 
	numlatents = latents.shape[0]
	out = torch.zeros(numlatents, numlatents)

	all_dists = torch.cdist(latents, latents)**2
	out = all_dists.masked_select(~torch.eye(numlatents, dtype=bool)).view(numlatents, numlatents - 1).view(-1)

	"""all_dists = torch.zeros(int(numlatents*(numlatents-1)/2))
	counter = 0
	for i in range(numlatents-1):
		for j in range(i+1, numlatents):
			dist_ij = torch.sum((latents[i, :] - latents[j, :])**2)
			out[i, j] = dist_ij
			out[j, i] = dist_ij
			all_dists[counter] = dist_ij
			print('i', i, 'j', j, dist_ij)
			counter = counter+1"""
	return all_dists, out

def perm_2d_tensor(mat, perm):
	return mat[perm,:][:,perm]

def MMD(latents1, latents2, shuffle=False):
	# latents = [Batch Dimension]

	latents1 = torch.tensor(latents1).float()
	latents2 = torch.tensor(latents2).float()

	all_latents = torch.cat((latents1, latents2), dim=0)
	all_pairwise_squared_dists, APSD_list = get_pairwise_squared_dists(all_latents)

	# Set sigma to median distance between samples over both
	sigma = torch.median(torch.tensor(APSD_list))
	exp_kernel = pairwise_exp_kernel(all_pairwise_squared_dists, sigma)

	m = latents1.shape[0]
	n = latents2.shape[0]

	if shuffle:
		perm = torch.randperm(m+n)
		exp_kernel = perm_2d_tensor(exp_kernel, perm)

	dists1 = torch.sum(torch.sum(exp_kernel[0:m, 0:m])) / (2*(m**2 - m))
	dists2 = torch.sum(torch.sum(exp_kernel[(m+1):n , (m+1):n ])) / (2*(n**2 - n)) 
	cross_dists = torch.sum(torch.sum(exp_kernel[0:m, (m+1):n ])) * 2 / (m*n)
	return dists1 + dists2 - cross_dists



