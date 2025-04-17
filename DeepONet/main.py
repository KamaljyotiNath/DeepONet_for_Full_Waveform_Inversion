'''
DeepONet for inverse problem in Wave propagation
@author: Kamaljyoti Nath(kamaljyoti_nath@brown.edu)
'''

import os

os.environ['CUDA_VISIBLE_DEVICES']='2'
print('GPU call', 2)

import tensorflow as tf
import numpy as np
import time
import scipy.io as io
import argparse
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print('GPU', gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

# tf.random.set_seed(1234)
# np.random.seed(1234)

from dataset import DataSet
from DeepONet import DeepONet
# from model_train import Train_Adam
from model_save import save_out
# from model_error import Error_cal

#%%
def main(Case):
    print('Case', Case)
    np_data_type = np.float32
    tf_data_type = tf.float32
    
    W_b_dict = np.load('Weight_bias/Weight_bias.npy', allow_pickle='True')
    W_b_dict = W_b_dict.item()
    
    dataset = DataSet()
    n_test_sample = dataset.load_data(np_data_type, Case)
    
    bs_sample_test = 100
    num_point_test = 70*70      
      
    #%%
    model = DeepONet(tf_data_type)   
    
    save_result = save_out(W_b_dict, model, np_data_type)
    save_result.nn_save_test(dataset, num_point_test, n_test_sample, bs_sample_test, Case)
        
    print('Done')
    print('==============================================================================')
#%%    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Case_', default=1, type=int, help='Case')
    args = parser.parse_args()
    Case = args.Case_
    
    # creat_folder
    print('==============================================================================')
    print('Tensorflow version:', tf.__version__)
    current_directory = os.getcwd()
    print('current_directory', current_directory)
    print('==============================================================================')
    
    save_dir = '/Output/Case_'+str(Case)
    save_output_to = current_directory + save_dir
    if not os.path.exists(save_output_to):
        os.makedirs(save_output_to, exist_ok=True)
        
    main(Case)
    print('Done')
    print('==============================================================================')
    
