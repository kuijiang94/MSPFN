import tensorflow as tf
import numpy as np
import sys
sys.path.append('../utils')
from layer import *
from BasicConvLSTMCell import *

class Model:
    def __init__(self, x_rain, is_training, batch_size):
        self.batch_size = batch_size
        n,w,h,c = x_rain.get_shape().as_list()
        self.weight = w//4
        self.height = h//4
        self.rain_res = self.generator(x_rain, is_training, tf.AUTO_REUSE)
        self.imitation = x_rain -  self.rain_res
    
    def generator(self, rain, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):
            # RNN框架
            with tf.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([self.weight*4, self.height*4], [3, 3], 256)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            with tf.variable_scope('LSTM2'):
                cell2 = BasicConvLSTMCell([self.weight*2, self.height*2], [3, 3], 128)
                rnn_state2 = cell2.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            with tf.variable_scope('LSTM4'):
                cell4 = BasicConvLSTMCell([self.weight, self.height], [3, 3], 64)
                rnn_state4 = cell4.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                    
            self.down_2 = self.downscale2(rain)
            self.down_4 = self.downscale4(rain)
                    
            with tf.variable_scope('rnn1'):
                rain1 = deconv_layer(
                    rain, [3, 3, 64, 3], [self.batch_size, self.weight*4, self.height*4, 64], 1)
                rain1 = prelu(rain1)
            with tf.variable_scope('rnn2'):
                rain2 = deconv_layer(
                    self.down_2, [3, 3, 64, 3], [self.batch_size, self.weight*2, self.height*2, 64], 1)
                rain2 = prelu(rain2)
            with tf.variable_scope('rnn4'):
                rain4 = deconv_layer(
                    self.down_4, [3, 3, 64, 3], [self.batch_size, self.weight, self.height, 64], 1)
                rain4 = prelu(rain4)

            long_connection = rain1
            with tf.variable_scope('residual_memory'):
			    ## RMM
			    ############ rain4
			    ############ rain4
                with tf.variable_scope('rain4_res1'):
                    res4_lstmin = deconv_layer(
                        rain4, [3, 3, 32, 64], [self.batch_size, self.weight, self.height, 32], 1)
                    res4_lstmin = prelu(res4_lstmin)
                with tf.variable_scope('lstm_group4'):
                    y_4, rnn_state4 = cell4(res4_lstmin, rnn_state4)
                with tf.variable_scope('rain4_res2'):
                    res4_lstmout = deconv_layer(
                        y_4, [3, 3, 64, 64], [self.batch_size, self.weight, self.height, 64], 1)
                    res4_lstmout = prelu(res4_lstmout)
                res4_lstmout += rain4
			    ############ rain2
			    ############ rain2
                with tf.variable_scope('upRMM4_2'):
                    rain4to2 = deconv_layer(
                        res4_lstmout, [3, 3, 64, 64], [self.batch_size, self.weight*2, self.height*2, 64], 2)
                    rain4to2 = prelu(rain4to2)
                rain2_e = rain2 + rain4to2
                with tf.variable_scope('rain2_res1'):
                    res2_lstmin = deconv_layer(
                        rain2_e, [3, 3, 32, 64], [self.batch_size, self.weight*2, self.height*2, 32], 1)
                    res2_lstmin = prelu(res2_lstmin)
                with tf.variable_scope('lstm_group2'):
                    y_2, rnn_state2 = cell2(res2_lstmin, rnn_state2)
                with tf.variable_scope('rain2_res2'):
                    res2_lstmout = deconv_layer(
                        y_2, [3, 3, 64, 128], [self.batch_size, self.weight*2, self.height*2, 64], 1)
                    res2_lstmout = prelu(res2_lstmout)
                res2_lstmout += rain2
			    ############ rain
			    ############ rain
                with tf.variable_scope('upRMM4_1'):
                    rain4to2 = deconv_layer(
                        rain4to2, [3, 3, 64, 64], [self.batch_size, self.weight*4, self.height*4, 64], 2)
                    rain4to2 = prelu(rain4to2)
                with tf.variable_scope('upRMM2_1'):
                    rain2to1 = deconv_layer(
                        res2_lstmout, [3, 3, 64, 64], [self.batch_size, self.weight*4, self.height*4, 64], 2)
                    rain2to1 = prelu(rain2to1)
                rain1_e = rain1 + rain2to1 + rain4to2
                with tf.variable_scope('rain_res1'):
                    res_lstmin = deconv_layer(
                        rain1_e, [3, 3, 32, 64], [self.batch_size, self.weight*4, self.height*4, 32], 1)
                    res_lstmin = prelu(res_lstmin)
                with tf.variable_scope('lstm_group'):
                    y_1, rnn_state = cell(res_lstmin, rnn_state)
                with tf.variable_scope('rain_res2'):
                    res_lstmout = deconv_layer(
                        y_1, [3, 3, 64, 256], [self.batch_size, self.weight*4, self.height*4, 64], 1)
                    res_lstmout = prelu(res_lstmout)
                res_lstmout += rain1

            u4_long_connection = res4_in = res4_lstmout
            u2_long_connection = res2_in = res2_lstmout
            u_long_connection = res_in = res_lstmout
            for n in range(17):
                ## URAB
                ############ rain4
                ############ rain4
                with tf.variable_scope('URAB4_{}'.format(n)):
                    with tf.variable_scope('down4_{}'.format(n+1)):
                        x_rnn = conv_layer(res4_in, [3, 3, 64, 64], 2) 
                        x_rnn = prelu(x_rnn)
                    res_short = res_input = x_rnn
                    for m in range(1):
                        with tf.variable_scope('group_{}_RCAB{}'.format(n+1,m+1)):
                            res_input = self.RCAB(res_input, 4)
                    with tf.variable_scope('up_{}'.format(n+1)):
                        res_out4 = deconv_layer(
                            tf.concat([res_short, res_input],3), [3, 3, 64, 128], [self.batch_size, self.weight, self.height, 64], 2)#2
                        res_out4 = prelu(res_out4)
                    res4_in += res_out4

                ############ rain2
                ############ rain2
                with tf.variable_scope('URAB2_{}'.format(n)):
                    with tf.variable_scope('upURAB4_2_{}'.format(n)):
                        rain4_resto2 = deconv_layer(
                            res4_in, [3, 3, 64, 64], [self.batch_size, self.weight*2, self.height*2, 64], 2)
                        rain4_resto2 = prelu(rain4_resto2)
                    res2_in_e = res2_in + rain4_resto2
                    with tf.variable_scope('down2_{}'.format(n+1)):
                        x_rnn = conv_layer(res2_in_e, [3, 3, 64, 64], 2) 
                        x_rnn = prelu(x_rnn)
                    res_short = res_input = x_rnn
                    for m in range(3):
                        with tf.variable_scope('group_{}_RCAB{}'.format(n+1,m+1)):
                            res_input = self.RCAB(res_input, 4)
                    with tf.variable_scope('up_{}'.format(n+1)):
                        res_out2 = deconv_layer(
                            tf.concat([res_short, res_input],3), [3, 3, 64, 128], [self.batch_size, self.weight*2, self.height*2, 64], 2)#2
                        res_out2 = prelu(res_out2)
                    res2_in += res_out2

                ############ rain
                ############ rain
                with tf.variable_scope('URAB_{}'.format(n)):
                    with tf.variable_scope('upURAB4to1_{}'.format(n)):
                        rain4_resto1 = deconv_layer(
                            rain4_resto2, [3, 3, 64, 64], [self.batch_size, self.weight*4, self.height*4, 64], 2)
                        rain4_resto1 = prelu(rain4_resto1)
                    with tf.variable_scope('upURAB2to1_{}'.format(n)):
                        rain2_resto1 = deconv_layer(
                            res2_in, [3, 3, 64, 64], [self.batch_size, self.weight*4, self.height*4, 64], 2)
                        rain2_resto1 = prelu(rain2_resto1)
                    res_in_e = res_in + rain4_resto1 + rain2_resto1
                    with tf.variable_scope('down_{}'.format(n+1)):
                        x_rnn = conv_layer(res_in_e, [3, 3, 64, 64], 2) 
                        x_rnn = prelu(x_rnn)
                    res_short = res_input = x_rnn
                    for m in range(3):
                        with tf.variable_scope('group_{}_RCAB{}'.format(n+1,m+1)):
                            res_input = self.RCAB(res_input, 4)
                    with tf.variable_scope('up_{}'.format(n+1)):
                        res_out = deconv_layer(
                            tf.concat([res_short, res_input],3), [3, 3, 64, 128], [self.batch_size, self.weight*4, self.height*4, 64], 2)#2
                        res_out = prelu(res_out)
                    res_in += res_out
                    
            ################ rememory-long connections
            with tf.variable_scope('rnn_recon4'):
                res4_in = deconv_layer(
                    tf.concat([res4_in, res4_lstmout],3), [3, 3, 64, 128], [self.batch_size, self.weight, self.height, 64], 1)#2
                res4_in = prelu(res4_in)
            with tf.variable_scope('rnn_recon2'):
                res2_in = deconv_layer(
                    tf.concat([res2_in, res2_lstmout],3), [3, 3, 64, 128], [self.batch_size, self.weight*2, self.height*2, 64], 1)#2
                res2_in = prelu(res2_in)
            with tf.variable_scope('rnn_recon'):
                res_in = deconv_layer(
                    tf.concat([res_in, res_lstmout],3), [3, 3, 64, 128], [self.batch_size, self.weight*4, self.height*4, 64], 1)#2
                res_in = prelu(res_in)
                    
            #lstm_output = lstm_in + lstm_input
            with tf.variable_scope('up4_2'):
                res4_2 = deconv_layer(
                    res4_in, [3, 3, 64, 64], [self.batch_size, self.weight*2, self.height*2, 64], 2)#2
                res4_2 = prelu(res4_2)
            with tf.variable_scope('up2_1'):
                res42_1 = deconv_layer(
                    tf.concat([res4_2, res2_in],3), [3, 3, 64, 128], [self.batch_size, self.weight*4, self.height*4, 64], 2)#2
                res42_1 = prelu(res42_1)
            res_mem_con = tf.concat([res_in, res42_1],3)
            with tf.variable_scope('rnn5'):
                res_mem_con = deconv_layer(
                    res_mem_con, [3, 3, 32, 128], [self.batch_size, self.weight*4, self.height*4, 32], 1)#2
                res_mem_con = prelu(res_mem_con)
            with tf.variable_scope('rnn7'):
                res_rainimage = deconv_layer(
                    res_mem_con, [3, 3, 3, 32], [self.batch_size, self.weight*4, self.height*4, 3], 1)#2
        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return res_rainimage
		
    def RCAB(self, input, reduction):
        b, w, h, channel = input.get_shape()  # (B, W, H, C)
        f = tf.layers.conv2d(input, channel, 3, padding='same', activation=prelu)  # (B, W, H, C)tf.nn.relu
        f = tf.layers.conv2d(f, channel, 3, padding='same')  # (B, W, H, C)
        x = tf.reduce_mean(f, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
        x = tf.layers.conv2d(x, channel // reduction, 1, activation=prelu)  # (B, 1, 1, C // r)
        x = tf.layers.conv2d(x, channel, 1, activation=tf.nn.sigmoid)  # (B, 1, 1, C)
        x = tf.multiply(f, x)  # (B, W, H, C)
        x = tf.add(input, x)
        return x
        
    def downscale2(self, x):
        K = 2
        arr = np.zeros([K, K, 3, 3])
        arr[:, :, 0, 0] = 1.0 / K ** 2
        arr[:, :, 1, 1] = 1.0 / K ** 2
        arr[:, :, 2, 2] = 1.0 / K ** 2
        weight = tf.constant(arr, dtype=tf.float32)
        downscaled = tf.nn.conv2d(
            x, weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled
    def downscale4(self, x):
        K = 4
        arr = np.zeros([K, K, 3, 3])
        arr[:, :, 0, 0] = 1.0 / K ** 2
        arr[:, :, 1, 1] = 1.0 / K ** 2
        arr[:, :, 2, 2] = 1.0 / K ** 2
        weight = tf.constant(arr, dtype=tf.float32)
        downscaled = tf.nn.conv2d(
            x, weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled

