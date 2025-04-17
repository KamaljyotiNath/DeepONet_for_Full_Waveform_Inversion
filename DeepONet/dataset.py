import numpy as np
import scipy.io as io
import sys

class DataSet:
    def __init__(self):
        pass

    def load_data(self, np_data_type, Case):
        
        # print('loading training data')
        # for k in range(32): 
        #     print('Sample', k)
        #     data_station = np.load('../Data/seis/seis2_1_'+str(k)+'.npy')
        #     data_vel = np.load('../Data/vel/vel2_1_'+str(k)+'.npy')
            
        #     data_station = np.transpose(data_station, (0, 3, 2, 1))

        #     data_vel = data_vel[:,0,:,:]
            
        #     if k == 0:
        #         self.br_train = data_station
        #         self.out_train = data_vel
        #     else:
        #         self.br_train = np.concatenate((self.br_train, data_station), axis = 0)
        #         self.out_train = np.concatenate((self.out_train, data_vel), axis = 0)
        
        # for k in range(32): 
        #     print('Sample', k)
        #     data_station = np.load('../Data/seis/seis3_1_'+str(k)+'.npy')
        #     data_vel = np.load('../Data/vel/vel3_1_'+str(k)+'.npy')
            
        #     data_station = np.transpose(data_station, (0, 3, 2, 1))

        #     data_vel = data_vel[:,0,:,:]
            
        #     self.br_train = np.concatenate((self.br_train, data_station), axis = 0)
        #     self.out_train = np.concatenate((self.out_train, data_vel), axis = 0)
            
        # for k in range(32): 
        #     print('Sample', k)
        #     data_station = np.load('../Data/seis/seis4_1_'+str(k)+'.npy')
        #     data_vel = np.load('../Data/vel/vel4_1_'+str(k)+'.npy')
            
        #     data_station = np.transpose(data_station, (0, 3, 2, 1))

        #     data_vel = data_vel[:,0,:,:]
            
        #     self.br_train = np.concatenate((self.br_train, data_station), axis = 0)
        #     self.out_train = np.concatenate((self.out_train, data_vel), axis = 0)
        
        # self.br_train = np.log1p(np.abs(self.br_train))*np.sign(self.br_train)
        
        # train_sample = np.shape(self.br_train)[0]
        # self.out_train = np.reshape(self.out_train, (train_sample, -1))
        #%%
        print('loading testing data')
        for k in range(32, 36): 
            print('Sample', k)
            data_station = np.load('../Data/seis/seis2_1_'+str(k)+'.npy')
            data_vel = np.load('../Data/vel/vel2_1_'+str(k)+'.npy')
            
            data_station = np.transpose(data_station, (0, 3, 2, 1))

            data_vel = data_vel[:,0,:,:]
            
            if k == 32:
                self.br_test = data_station
                self.out_test = data_vel
            else:
                self.br_test = np.concatenate((self.br_test, data_station), axis = 0)
                self.out_test = np.concatenate((self.out_test, data_vel), axis = 0)
               
        for k in range(32, 36): 
            print('Sample', k)
            data_station = np.load('../Data/seis/seis3_1_'+str(k)+'.npy')
            data_vel = np.load('../Data/vel/vel3_1_'+str(k)+'.npy')
            
            data_station = np.transpose(data_station, (0, 3, 2, 1))

            data_vel = data_vel[:,0,:,:]
            
            self.br_test = np.concatenate((self.br_test, data_station), axis = 0)
            self.out_test = np.concatenate((self.out_test, data_vel), axis = 0)
        
        for k in range(32, 36): 
            print('Sample', k)
            data_station = np.load('../Data/seis/seis4_1_'+str(k)+'.npy')
            data_vel = np.load('../Data/vel/vel4_1_'+str(k)+'.npy')
            
            data_station = np.transpose(data_station, (0, 3, 2, 1))

            data_vel = data_vel[:,0,:,:]
            
            self.br_test = np.concatenate((self.br_test, data_station), axis = 0)
            self.out_test = np.concatenate((self.out_test, data_vel), axis = 0)
        
        self.br_test = np.log1p(np.abs(self.br_test))*np.sign(self.br_test)
                
        test_sample = np.shape(self.br_test)[0]
        
        self.out_test = np.reshape(self.out_test, (test_sample, -1))
        
        print('=====================================')
        # total_sample = train_sample + test_sample
        # print('[Total samples: %d] [Train samples: %d] [Test samples: %d]' %(total_sample, train_sample, test_sample))
        #%% trunk data
        self.X = np.linspace(0, 690, 70).reshape(-1,1)
        self.Y = np.linspace(0, 690, 70).reshape(-1,1)
        self.X, self.Y = np.meshgrid(self.X, self.Y)
        
        self.X = np.reshape(self.X, (-1,1))
        
        self.Y = np.reshape(self.Y, (-1,1))
        #%%
        print('============== Before normalization ============')
        
        br_scale = io.loadmat('Weight_bias/branch_input_scale.mat')
        
        self.in_max_train = br_scale['in_max_train']
        self.in_min_train = br_scale['in_min_train']
        
        # print('Train In [max: %0.2e] [min: %0.2e]' %(self.in_max_train, self.in_min_train))
        
        self.in_max_test = np.max(self.br_test)
        self.in_min_test = np.min(self.br_test)
        print('Test In [max: %0.2e] [min: %0.2e]' %(self.in_max_test, self.in_min_test))
        
        self.out_max_test = np.max(self.out_test)
        self.out_min_test = np.min(self.out_test)
        print('Test Out [max: %0.2e] [min %0.2e]' %(self.out_max_test, self.out_min_test))
        
        #%%
        # self.br_train = 2*(self.br_train - self.in_min_train)/(self.in_max_train - self.in_min_train) - 1
        self.br_test = 2*(self.br_test - self.in_min_train)/(self.in_max_train - self.in_min_train) - 1
        
        #%% Assign data type
        # self.br_train = self.br_train.astype(np_data_type)
        self.br_test = self.br_test.astype(np_data_type)
        
        # self.out_train = self.out_train.astype(np_data_type)
        self.out_test = self.out_test.astype(np_data_type)
        
        self.X = self.X.astype(np_data_type)
        self.Y = self.Y.astype(np_data_type)  
       
        print('============== After normalization ============')
        # in_max_train = np.max(self.br_train)
        # in_min_train = np.min(self.br_train)
        # print('Train In [max: %0.2e] [min: %0.2e]' %(in_max_train, in_min_train))
        
        in_max_test = np.max(self.br_test)
        in_min_test = np.min(self.br_test)
        print('Test In [max: %0.2e] [min: %0.2e]' %(in_max_test, in_min_test))
        
        # out_max_train = np.max(self.out_train)
        # out_min_train = np.min(self.out_train)
        # print('Train Out [max: %0.2e] [min: %0.2e]' %(out_max_train, out_min_train))
        
        out_max_test = np.max(self.out_test)
        out_min_test = np.min(self.out_test)
        print('Test Out [max: %0.2e] [min %0.2e]' %(out_max_test, out_min_test))
        
        return test_sample

    #%%
    def train_mini_batch(self, sample):
        
        br1_train_sam = self.br_train[sample,:,:,:]
        out_train_sam = self.out_train[sample,:]
                
        return br1_train_sam, self.X, self.Y, out_train_sam
    
    def test_mini_batch(self, sample):
        
        br1_test_sam = self.br_test[sample,:,:,:]
        out_test_sam = self.out_test[sample,:]
                
        return br1_test_sam, self.X, self.Y, out_test_sam
    
    def decode_out(self, output):
        
        return output
    