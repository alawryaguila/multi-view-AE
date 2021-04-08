'''
calc_corr: function for calculating different correlation measurements

'''
import numpy as np

def calc_corr(x1, x2, corr_type='pearson'):
    if corr_type=='pearson':
        return np.corrcoef(x1,x2)

