'''
Class for conducting cross validation

'''
import torch
import itertools as it
import numpy as np 
import pandas as pd 
import os
from os.path import join
import datetime
import json
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
class CrossValidation:
    def __init__(self, config, parameters, model_type=None):
        self.config = config
        self.parameters = parameters
        self.model_type = model_type
        self.output_path = self.config['output_dir']

    def gridsearch(self, *data, k_folds=5, jobs=0):

        kfold = KFold(n_splits=k_folds, shuffle=True,random_state=42) #check default split size
        param_combs = list(it.product(*(self.parameters[key] for key in self.parameters)))
        self.input_dims = [np.shape(data_)[1] for data_ in data]
        param_labels_dict = {}
        for i, param_comb in enumerate(param_combs):        
            param_labels_dict['Comb {0}'.format(i)] = {key: param_comb[idx] for idx, key in enumerate(self.parameters)}
        
        if jobs>0:
            print("running array job")
            cv_results = Parallel(n_jobs=jobs)(delayed(self.kfold_cv)(kfold,  param_labels_dict['Comb {0}'.format(i)], *data) for i, param_comb in enumerate(param_combs))
        else:
            cv_results = [self.kfold_cv(kfold, param_labels_dict['Comb {0}'.format(i)], *data) for i, param_comb in enumerate(param_combs)]
        cv_dict = {}
        for i in range(len(param_combs)):        
            cv_dict['Comb {0}'.format(i)] = cv_results[i]
            
        if self.config['sparse']:
            self.model_type = 'sparse_' + self.model_type  
        if self.config['save_model']:
            outpath = self.format_folder()
            json.dump(param_labels_dict,open(join(outpath, 'parameter_labels.json'),'w'))
            json.dump(cv_dict,open(join(outpath, 'cv_results.json'),'w'))
        self.param_labels_dict = param_labels_dict 
        self.cv_dict = cv_dict

    def kfold_cv(self, kfold, parameters, *data):
        cv_dict = {}
        for fold, (train_ids, val_ids) in enumerate(kfold.split(data[0])):
            print("fold: ", fold)
            train_data, val_data = self.select_data(train_ids, val_ids, *data)
            model = self.init_model(self.input_dims, parameters)
            model.fit(*train_data)
            same_recon, cross_recon = model.MSE_results(*val_data)
            cv_dict['fold {0}'.format(fold)] = {'same view MSE': round(same_recon, 4), 'cross view MSE': round(cross_recon, 4)}
        return cv_dict
        
    def init_model(self, input_dims, parameters):
        self.input_dims = input_dims
        use_GPU = self.config['use_GPU']
        use_cuda = use_GPU and torch.cuda.is_available()
        self.kwargs_generator = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        self.device = torch.device("cuda" if use_cuda else "cpu")
        #update config file depending on parameters
        config = self.config
        for key,val in parameters.items():
            config[key] = val
        if self.model_type=='joint_VAE':
            from ..models.joint_vae import VAE
            return VAE(self.input_dims, config).to(self.device)
        elif self.model_type=='VAE':
            from ..models.vae import VAE
            return VAE(self.input_dims, config).to(self.device)
        #TO DO rest of models
        else:
            print("model type not recognised")
            exit()
    def select_data(self, train_ids,val_ids, *data):
        train_data = []
        val_data = []
        for data_ in data:
            train_data.append(data_[train_ids,:])
            val_data.append(data_[val_ids,:])
        return train_data, val_data

    def format_folder(self):
        init_path = self.config['output_dir']
        model_type = self.model_type
        date = str(datetime.date.today())
        date = date.replace('-', '_')
        output_path = init_path + '/' + model_type + '/' + date
        if not os.path.exists(output_path):
            os.makedirs(output_path)   
        return output_path