#!/usr/bin/env python
# coding: utf-8

from fastai import *
from fastai.text import *

def calcPagePrior(train_df, valid_df):
    
    # get label info
    df = pd.concat([train_df, valid_df], sort=False)
    labels = df['label'].tolist()
    composers = sorted(list(set(labels)))
    composer2idx = {c:i for i, c in enumerate(composers)}
    
    # accumulate & normalize
    counts = np.zeros(len(composers))
    for l in labels:
        counts[composer2idx[l]] += 1
    priors = counts / np.sum(counts)

    # format
    priors = torch.from_numpy(priors.reshape((1,-1)))
    
    return priors

def calcAccuracy_fullpage(learner, path, train_df, valid_df, test_df, databunch = None, ensembled = False):
    
    # batch inference
    if databunch is None: # RNNLearner (AWD-LSTM)
        learner.export()
        learner = load_learner(path, test=TextList.from_df(test_df, path, cols='text'))
        probs, y = learner.get_preds(ds_type=DatasetType.Test, ordered=True) 
    else: # Generic Learner (RoBERTa, GPT-2)
        learner = Learner(databunch, learner.model)
        probs = learner.get_preds(ds_type=DatasetType.Test)[0].detach().cpu() # not sorted
        sampler = [i for i in databunch.dl(DatasetType.Test).sampler]
        reverse_sampler = np.argsort(sampler)
        probs = probs[reverse_sampler, :]
    
    # ground truth labels
    labels = list(test_df['label'])
    composers = sorted(set(labels))
    composer2idx = {c:i for i, c in enumerate(composers)}
    gt = torch.from_numpy(np.array([composer2idx[l] for l in labels]))
    
    # average if ensembled
    if ensembled:
        boundaries = getPageBoundaries(test_df)
        probs, gt = averageEnsembled(probs, gt, boundaries)
    
    # apply priors
    priors = calcPagePrior(train_df, valid_df)
    probs_with_priors = torch.mul(probs, priors)
    
    # calc accuracy
    acc = accuracy(probs, gt).item()
    acc_with_prior = accuracy(probs_with_priors, gt).item()
    
    return acc, acc_with_prior

def getPageBoundaries(df):
    queryids = list(df['id'])
    boundaries = []
    for i, qid in enumerate(queryids):
        if qid[-2:] == '_0':
            boundaries.append(i)
    return boundaries

def averageEnsembled(probs, gt, boundaries):
    gt_selected = gt[boundaries]
    accum = torch.zeros((len(boundaries), probs.shape[1]))
    for i, bnd in enumerate(boundaries):
        if i == len(boundaries) - 1:
            accum[i,:] = torch.sum(probs[bnd:,:], axis=0)
        else:
            next_bnd = boundaries[i+1]
            accum[i,:] = torch.sum(probs[bnd:next_bnd,:], axis=0)
    return accum, gt_selected
