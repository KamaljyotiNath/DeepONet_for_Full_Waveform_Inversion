import tensorflow as tf
import numpy as np
import sys

class DeepONet:
    def __init__(self, tf_data_type):
        self.tf_data_type = tf_data_type
        
    def hyper_ini_W_b(self, layers):
        L = len(layers)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers[l-1]
            out_dim = layers[l]
            # std = np.sqrt(2./(in_dim + out_dim))
            std = np.sqrt(2.0/(in_dim))
            weight = tf.Variable(tf.random.truncated_normal(shape=[in_dim, out_dim], mean= 0.0, stddev=std, dtype=self.tf_data_type), dtype=self.tf_data_type)
            bias = tf.Variable(tf.zeros(shape=[1, out_dim], dtype=self.tf_data_type), dtype=self.tf_data_type)
                              
            W.append(weight)
            b.append(bias)
        
        return W, b
    
    def hyper_ini_tr(self, layers):
        in_dim = layers[0]
        out_dim = layers[1]
        std = np.sqrt(2.0/(in_dim))
        weight = tf.Variable(tf.random.truncated_normal(shape=[in_dim, out_dim],  mean= 0.0, stddev=std, dtype=self.tf_data_type), dtype=self.tf_data_type)
        bias = tf.Variable(tf.zeros(shape=[1, out_dim], dtype=self.tf_data_type), dtype=self.tf_data_type)
                              
        return weight, bias
    
    def hyper_ini_CNN(self, shape_w, shape_b):
        ch_1 = shape_w[0]
        ch_2 = shape_w[1]
        in_dim = shape_w[2]
        std = np.sqrt(2.0/(ch_1*ch_2*in_dim))
        print('CNN Dim: [%2d, %d, %d] [SD: %0.4f]' %(ch_1, ch_2, in_dim, std))
        weight = tf.Variable(tf.random.truncated_normal(shape=shape_w,  mean= 0.0, stddev=std, dtype=self.tf_data_type), dtype=self.tf_data_type)
        bias = tf.Variable(tf.zeros(shape=shape_b, dtype=self.tf_data_type), dtype=self.tf_data_type)
        
        return weight, bias
    
    def hyper_ini_CNN_transp(self, shape_w, shape_b):
        ch_1 = shape_w[0]
        ch_2 = shape_w[1]
        in_dim = shape_w[3]
        std = np.sqrt(2.0/(ch_1*ch_2*in_dim))
        print('Trans CNN Dim: [%2d, %d, %d] [SD: %0.4f]' %(ch_1, ch_2, in_dim, std))
        weight = tf.Variable(tf.random.truncated_normal(shape=shape_w,  mean= 0.0, stddev=std, dtype=self.tf_data_type), dtype=self.tf_data_type)
        bias = tf.Variable(tf.zeros(shape=shape_b, dtype=self.tf_data_type), dtype=self.tf_data_type)
        
        return weight, bias
    
    #%%
    def fnn_B1(self, W, b, X):
        A = X
        L = len(W)
        for i in range(L-1):
            A = tf.nn.relu((tf.add(tf.matmul(A, W[i]), b[i])))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y
    
    def fnn_T(self, W, b, X, Y):
        X  = 2*(X-0.0)/(690.0-0.0) -1.0
        Y = 2*(Y-0.0)/(690.0-0.0) -1.0
        A = tf.concat((X, Y), axis=-1)
        L = len(W)
        for i in range(L-1):
            A = tf.nn.relu((tf.add(tf.matmul(A, W[i]), b[i])))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y
    
    #%%
    def conv_layer(self, x, W, b, stride_x, stride_t, actn=tf.nn.relu):
        layer = tf.nn.conv2d(x, W, strides=[1, stride_x, stride_t, 1], padding='VALID')
        layer += b
        y = actn(layer)
        
        return y
    
    def avg_pool(self, x, ksize, stride):
        pool_out = tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1], \
                                  strides=[1, stride, stride, 1],\
                                  padding='SAME')    
        return pool_out
    
    def max_pool(self, x, ksize, stride):
        pool_out = tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], \
                                  strides=[1, stride, stride, 1],\
                                  padding='SAME')    
        return pool_out

    def flatten_layer(self, layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat
    
    #%%
    def trans_conv_layer(self, x, W, b, stride_x, stride_t, output_shape, actn=tf.nn.leaky_relu): 
        layer = tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, stride_x, stride_t, 1], padding='VALID')
        layer += b
        y = actn(layer)
        
        return y
    
    def upsampling(self, x, stride_upsam):
        
        y = tf.keras.layers.UpSampling2D(size=(stride_upsam, stride_upsam))(x)
        
        return y
    
    #%%
    @tf.function
    def out_CNN_DeepONet(self, model,
                                W_b_dict,
                                br_in, tr_x_in, tr_y_in):
        
        UNet_CNN_1_W, UNet_CNN_1_b = W_b_dict['UNet_CNN_1_W'], W_b_dict['UNet_CNN_1_b']
        UNet_CNN_2_W, UNet_CNN_2_b = W_b_dict['UNet_CNN_2_W'], W_b_dict['UNet_CNN_2_b']
        UNet_CNN_3_W, UNet_CNN_3_b = W_b_dict['UNet_CNN_3_W'], W_b_dict['UNet_CNN_3_b']
        UNet_CNN_4_W, UNet_CNN_4_b = W_b_dict['UNet_CNN_4_W'], W_b_dict['UNet_CNN_4_b']
        
        stride_x, stride_t = 2, 2
        UNet_conv_out_1 = model.conv_layer(br_in, UNet_CNN_1_W, UNet_CNN_1_b, stride_x, stride_t, actn=tf.nn.relu)
        # print('UNet_conv_out_1', np.shape(UNet_conv_out_1))
        
        stride_x, stride_t = 2, 2
        UNet_conv_out_2 = model.conv_layer(UNet_conv_out_1, UNet_CNN_2_W, UNet_CNN_2_b, stride_x, stride_t, actn=tf.nn.relu)
        # print('UNet_conv_out_2', np.shape(UNet_conv_out_2))
        
        stride_x, stride_t = 2, 2
        UNet_conv_out_3 = model.conv_layer(UNet_conv_out_2, UNet_CNN_3_W, UNet_CNN_3_b, stride_x, stride_t, actn=tf.nn.relu)
        # print('UNet_conv_out_3', np.shape(UNet_conv_out_3))
        
        stride_x, stride_t = 2, 2
        UNet_conv_out_4 = model.conv_layer(UNet_conv_out_3, UNet_CNN_4_W, UNet_CNN_4_b, stride_x, stride_t, actn=tf.nn.relu)
        # print('UNet_conv_out_4', np.shape(UNet_conv_out_4))
                       
        UNet_TrCNN_1_W,  UNet_TrCNN_1_b = W_b_dict['UNet_TrCNN_1_W'],  W_b_dict['UNet_TrCNN_1_b']
        UNet_TrCNN_2_W,  UNet_TrCNN_2_b = W_b_dict['UNet_TrCNN_2_W'],  W_b_dict['UNet_TrCNN_2_b']
        UNet_TrCNN_3_W,  UNet_TrCNN_3_b = W_b_dict['UNet_TrCNN_3_W'],  W_b_dict['UNet_TrCNN_3_b']
        UNet_TrCNN_4_W,  UNet_TrCNN_4_b = W_b_dict['UNet_TrCNN_4_W'],  W_b_dict['UNet_TrCNN_4_b']
        #%%
        out_dim1_1 = UNet_conv_out_3.get_shape()[0:3]
        out_dim2_1 = UNet_TrCNN_1_W.get_shape()[2:3]
        
        out_shape_1 = tf.concat((out_dim1_1, out_dim2_1), axis=-1)
        
        stride_x, stride_t = 2, 2
        UNet_TrCNN_1_in = UNet_conv_out_4 
        UNet_TrCNN_out_1 = model.trans_conv_layer(UNet_TrCNN_1_in, UNet_TrCNN_1_W,  UNet_TrCNN_1_b, stride_x, stride_t, out_shape_1, actn=tf.nn.leaky_relu)       
        # tf.print('out_shape_1', out_shape_1)
        # print('UNet_TrCNN_out_1', np.shape(UNet_TrCNN_out_1))
        
        out_dim1_2 = UNet_conv_out_2.get_shape()[0:3]
        out_dim2_2 = UNet_TrCNN_2_W.get_shape()[2:3]
        
        out_shape_2 = tf.concat((out_dim1_2, out_dim2_2), axis=-1)
        
        stride_x, stride_t = 2, 2
        UNet_TrCNN_2_in = tf.concat((UNet_TrCNN_out_1, UNet_conv_out_3), axis=-1)
        UNet_TrCNN_out_2 = model.trans_conv_layer(UNet_TrCNN_2_in, UNet_TrCNN_2_W,  UNet_TrCNN_2_b, stride_x, stride_t, out_shape_2, actn=tf.nn.leaky_relu)       
        # tf.print('out_shape_2', out_shape_2)
        # print('UNet_TrCNN_out_2', np.shape(UNet_TrCNN_out_2))
        
        out_dim1_3 = UNet_conv_out_1.get_shape()[0:3]
        out_dim2_3 = UNet_TrCNN_3_W.get_shape()[2:3]
        
        out_shape_3 = tf.concat((out_dim1_3, out_dim2_3), axis=-1)
        
        stride_x, stride_t = 2, 2
        UNet_TrCNN_3_in = tf.concat((UNet_TrCNN_out_2, UNet_conv_out_2), axis=-1)
        UNet_TrCNN_out_3 = model.trans_conv_layer(UNet_TrCNN_3_in, UNet_TrCNN_3_W,  UNet_TrCNN_3_b, stride_x, stride_t, out_shape_3, actn=tf.nn.leaky_relu)       
        # tf.print('out_shape_3', out_shape_3)
        # print('UNet_TrCNN_out_3', np.shape(UNet_TrCNN_out_3))
        
        out_dim1_4 = br_in.get_shape()[0:3]
        out_dim2_4 = UNet_TrCNN_4_W.get_shape()[2:3]
        
        out_shape_4 = tf.concat((out_dim1_4, out_dim2_4), axis=-1)
        
        stride_x, stride_t = 2, 2
        UNet_TrCNN_4_in = tf.concat((UNet_TrCNN_out_3, UNet_conv_out_1), axis=-1)
        UNet_TrCNN_out_4 = model.trans_conv_layer(UNet_TrCNN_4_in, UNet_TrCNN_4_W,  UNet_TrCNN_4_b, stride_x, stride_t, out_shape_4, actn=tf.nn.leaky_relu)
        
        #%%
        br1_CNN_1_W, br1_CNN_1_b1 = W_b_dict['br1_CNN_1_W'], W_b_dict['br1_CNN_1_b']
        br1_CNN_2_W, br1_CNN_2_b1 = W_b_dict['br1_CNN_2_W'], W_b_dict['br1_CNN_2_b']
        br1_CNN_3_W, br1_CNN_3_b1 = W_b_dict['br1_CNN_3_W'], W_b_dict['br1_CNN_3_b']
        br1_CNN_4_W, br1_CNN_4_b1 = W_b_dict['br1_CNN_4_W'], W_b_dict['br1_CNN_4_b']
        br1_CNN_5_W, br1_CNN_5_b1 = W_b_dict['br1_CNN_5_W'], W_b_dict['br1_CNN_5_b']
        br1_CNN_6_W, br1_CNN_6_b1 = W_b_dict['br1_CNN_6_W'], W_b_dict['br1_CNN_6_b']
        br1_CNN_7_W, br1_CNN_7_b1 = W_b_dict['br1_CNN_7_W'], W_b_dict['br1_CNN_7_b']
        br1_CNN_8_W, br1_CNN_8_b1 = W_b_dict['br1_CNN_8_W'], W_b_dict['br1_CNN_8_b']
        br1_CNN_9_W, br1_CNN_9_b1 = W_b_dict['br1_CNN_9_W'], W_b_dict['br1_CNN_9_b']
        
        br1_W, br1_b = W_b_dict['br1_W'], W_b_dict['br1_b']
        
        stride_x, stride_t = 1, 1
        X_in = tf.concat((UNet_TrCNN_out_4, br_in), axis=-1)
        conv_out_1_br_1 = model.conv_layer(X_in, br1_CNN_1_W, br1_CNN_1_b1, stride_x, stride_t, actn=tf.nn.relu)  
        # pool_1_br_1 = model.avg_pool(conv_out_1_br_1, ksize = 2, stride = stride_avg_poll)
        # print('conv_out_1_br_1', np.shape(conv_out_1_br_1))
        
        stride_x, stride_t = 1, 1
        conv_out_2_br_1 = model.conv_layer(conv_out_1_br_1, br1_CNN_2_W, br1_CNN_2_b1, stride_x, stride_t, actn=tf.nn.relu)
        # print('conv_out_2_br_1', np.shape(conv_out_2_br_1))
        
        stride_x, stride_t = 1, 1
        conv_out_3_br_1 = model.conv_layer(conv_out_2_br_1, br1_CNN_3_W, br1_CNN_3_b1, stride_x, stride_t, actn=tf.nn.relu)
        # print('conv_out_3_br_1', np.shape(conv_out_3_br_1))
        
        stride_x, stride_t = 1, 2
        conv_out_4_br_1 = model.conv_layer(conv_out_3_br_1, br1_CNN_4_W, br1_CNN_4_b1, stride_x, stride_t, actn=tf.nn.relu)
        # print('conv_out_4_br_1', np.shape(conv_out_4_br_1))
        
        stride_x, stride_t = 1, 2
        conv_out_5_br_1 = model.conv_layer(conv_out_4_br_1, br1_CNN_5_W, br1_CNN_5_b1, stride_x, stride_t, actn=tf.nn.relu)
        # print('conv_out_5_br_1', np.shape(conv_out_5_br_1))
        
        stride_x, stride_t = 2, 2
        conv_out_6_br_1 = model.conv_layer(conv_out_5_br_1, br1_CNN_6_W, br1_CNN_6_b1, stride_x, stride_t, actn=tf.nn.relu)
        # print('conv_out_6_br_1', np.shape(conv_out_6_br_1))
        
        stride_x, stride_t = 2, 2
        conv_out_7_br_1 = model.conv_layer(conv_out_6_br_1, br1_CNN_7_W, br1_CNN_7_b1, stride_x, stride_t, actn=tf.nn.relu)  
        #print('conv_out_7_br_1', np.shape(conv_out_7_br_1))
        
        stride_x, stride_t = 2, 2
        conv_out_8_br_1 = model.conv_layer(conv_out_7_br_1, br1_CNN_8_W, br1_CNN_8_b1, stride_x, stride_t, actn=tf.nn.relu)  
        #print('conv_out_8_br_1', np.shape(conv_out_8_br_1))
        
        stride_x, stride_t = 2, 2
        conv_out_9_br_1 = model.conv_layer(conv_out_8_br_1, br1_CNN_9_W, br1_CNN_9_b1, stride_x, stride_t, actn=tf.nn.relu)  
        #print('conv_out_9_br_1', np.shape(conv_out_9_br_1))
              
        layer_flat_br1 = model.flatten_layer(conv_out_9_br_1)
        br1_out = model.fnn_B1(br1_W, br1_b, layer_flat_br1)
        #print('layer_flat_br1', np.shape(layer_flat_br1))
        
        tr_W, tr_b = W_b_dict['tr_W'], W_b_dict['tr_b']
        tr_output = model.fnn_T(tr_W, tr_b, tr_x_in, tr_y_in)
        
        # print('tr_output', np.shape(tr_output))
        # print('br1_out', np.shape(br1_out))
            
        tr_output = tf.transpose(tr_output, (1,0))
        
        # print('tr_output', np.shape(tr_output))
        
        out = br1_out@tr_output
        out = 1490.0 + tf.nn.sigmoid(out)*(4510.0 - 1490.0)

        return out
