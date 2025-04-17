import tensorflow as tf
import numpy as np
import scipy.io as io

class save_out:
    def __init__(self, W_b_dict, model, np_data_type):
        self.W_b_dict = W_b_dict
        self.model = model
        self.np_data_type = np_data_type
    #%%
    
    def nn_save_test(self, dataset, num_point_test, n_test_sample, bs_sample, Case):
        
        out_test_pred_np = np.zeros((n_test_sample, num_point_test))
        out_test_true = np.zeros((n_test_sample, num_point_test))
        
        err_test_total = 0
        bs = int(n_test_sample/bs_sample)
        for k in range(bs):
            sample = np.linspace(k*bs_sample, (k+1)*bs_sample, bs_sample, dtype=int, endpoint=False)
            br_test_sam, tr_x_test_sam, tr_y_test_sam, out_test_sam = dataset.test_mini_batch(sample)
            
            out_test_pred = self.model.out_CNN_DeepONet(self.model,
                                                       self.W_b_dict,
                                                       br_test_sam, tr_x_test_sam, tr_y_test_sam)
                
            err_test = np.mean(np.linalg.norm((out_test_pred - out_test_sam), axis=(-1), ord=2)/np.linalg.norm(out_test_sam, axis=(-1), ord=2))*100
            
            err_test_total += err_test
            
            out_test_pred_np[sample,:] = out_test_pred.numpy()
            out_test_true[sample,:] = out_test_sam
        
        out_test_pred_np = dataset.decode_out(out_test_pred_np)
        out_test_true = dataset.decode_out(out_test_true)
        
        err_test_total = err_test_total/bs
        
        print('Testing error: %0.2e' %err_test_total)
        
        save_dict = {'out_test_pred': out_test_pred_np,
                     'out_test_true': out_test_true,
                     'error_test': err_test_total}
        
        io.savemat('./Output/Case_'+str(Case)+'/result_test_data.mat', save_dict)