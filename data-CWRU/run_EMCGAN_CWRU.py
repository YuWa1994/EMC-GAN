# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:22:42 2022

@author: yuwa
"""

import numpy as np
import scipy.io as sio 
import YW_utils as utils
import GAN_for_ENC as GAN
import GAN_Nets
import os
import torch
    

def get_data(datapath, is_normalization=False):
    data = sio.loadmat(datapath)
    
    trainX = data['trainX'].astype(np.float32) 
    testX = data['testX'].astype(np.float32) 
    trainY = data['trainY']-1
    testY = data['testY']-1

    if is_normalization:
        trainX, mu, sigma = utils.zscore(trainX, dim=0)
        testX = utils.normalize(testX, mu, sigma)
    return trainX, trainY, testX, testY


def test_update(model, X=None, D=None, G=None,warm_up=None,basemodels=None,
         check_hidden_representation=False):
    if not os.path.exists('./result'):
        os.makedirs('./result')
    result = {}
    
    out = model.get_scores_update(X, discriminator, generator, update=True,
                                  warm_up=warm_up, basemodels=basemodels)

    result['test_score'] = out[0]
    result['pred_label'] = out[1]
    result['prediction'] = out[2]
    
    if check_hidden_representation:
        result['DM'] = out[3]
        result['SM'] = out[4]
        result['FM'] = out[5]
                        
    return result

def test(model, X=None, D=None, state_dict=None, count=None, 
         check_hidden_representation=False, savepath='./result'):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    result = {}
    
    out = model.get_scores(X, D, state_dict)
    prediction = np.argmax(out[1], axis=1)

    result['test_score'] = out[0]
    result['prediction'] = prediction
    result['pred_label'] = out[2]
    
    if check_hidden_representation:
        result['DM'] = out[3]
        result['SM'] = out[4]
        result['FM'] = out[5]
    
    if count == 0:
        result['loss_C'] = model.returns['loss_C']
        result['loss_D'] = model.returns['loss_D']
        result['loss_G'] = model.returns['loss_G']
        result['gen_X'] = model.returns['gen_X']
        result['gen_X_label'] = model.returns['gen_X_label']
        result['gen_X_scores'] = model.returns['gen_X_scores']        
    return result


# ------------------------------
#  Multi-classification 
# ------------------------------
datasets = ['CWRU4-6-V2']

for data in datasets:
    save_history = f'./history/{data}'
    save_result = f'./results/{data}'
    
    utils.setup_seed(seed=0)
    trainX, trainY, testX, testY = get_data(data)
    
    ### Define Nets
    generator = GAN_Nets.Generator_1024V1()
    discriminator = GAN_Nets.Discriminator_1024V2(classes=len(np.unique(trainY))+1)
    
    ### Train Nets
    model = GAN.GAN_for_ENC(trainX, label=trainY, n_epoch=500, batch_size=20,
                            learn_rate=0.0001,
                            buffer_size=200,
                            savepath=save_history, sample_interval=1,
                            noise=False, is_normalization=True, check_loss=True,
                            train_method='WGAN-gp')
    
    model = model.train(G=generator, D=discriminator)
         
    ### ensemble model with update
    out = model.get_scores_update(testX, discriminator, generator, update=True,
                                  warm_up=50, basemodel_num=15, stride=1)
    result = {}
    result['test_score'] = out[0]
    result['pred_label'] = out[1]
    result['prediction'] = out[2]
    result['update_time'] = model.update_time
    result['X'] = model.X
    result['label'] = model.label
    result['loss_C'] = model.returns['loss_C']
        
    if not os.path.exists(save_result): os.makedirs(save_result)
    sio.savemat(f'{save_result}/result.mat', result)   
    
'''
    ### single model
    count = 0
    if not os.path.exists(save_result): os.makedirs(save_result)
    for _, _ ,k in os.walk(save_history):
        for n in k:
            state_dict = save_history+'/'+n
            result = test(model, testX, discriminator, state_dict, count, savepath=save_result)
            sio.savemat(f'{save_result}/epoch{n[2:]}.mat', result)  
            count = count+1    
    
''' 