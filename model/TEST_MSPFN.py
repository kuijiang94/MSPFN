import tensorflow as tf
import numpy as np
import sys
sys.path.append('../utils')
#sys.path.append('../vgg19')
from layer import *
from BasicConvLSTMCell import *

class Model:
    def __init__(self, x_rain, is_training, batch_size):
        self.batch_size = batch_size
        n,w,h,c = x_rain.get_shape().as_list()
        self.weight = w//4
        self.height = h//4
        #self.vgg = VGG19(None, None, None)
        #self.down_2 = self.downscale2(x_rain)
        #self.down_4 = self.downscale4(x_rain)
        #self.rian = x_rain - x
        #self.input_ori = x_rain-self.rain_4-self.rain_2
        #self.label_ori = self.rian - self.rain_4 - self.rain_2 
        self.rain_res = self.generator(x_rain, is_training, tf.AUTO_REUSE)

        self.imitation = x_rain -  self.rain_res
    #image_size =  24#self.frame_hr, self.frame_sr,, self.g_frame_loss, self.d_frame_loss
    
    def generator(self, rain, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):
            # RNN框架
            with tf.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([self.weight*4, self.height*4], [3, 3], 32)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            with tf.variable_scope('LSTM2'):
                cell2 = BasicConvLSTMCell([self.weight*2, self.height*2], [3, 3], 32)
                rnn_state2 = cell2.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            with tf.variable_scope('LSTM4'):
                cell4 = BasicConvLSTMCell([self.weight, self.height], [3, 3], 32)
                rnn_state4 = cell4.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                    
            #self.down_2 = tf.image.resize_images(rain, [self.weight*2, self.height*2], method=2)
            #self.down_4 = tf.image.resize_images(rain, [self.weight, self.height], method=2)
            self.down_2 = self.downscale2(rain)
            self.down_4 = self.downscale4(rain)
            #input = tf.concat([self.bic_2, self.bic_4, rain],3)
                    
            with tf.variable_scope('rnn1'):#1.68kb
                rain1 = deconv_layer(
                    rain, [3, 3, 32, 3], [self.batch_size, self.weight*4, self.height*4, 32], 1)
                rain1 = prelu(rain1)
            with tf.variable_scope('rnn2'):#1.68kb
                rain2 = deconv_layer(
                    self.down_2, [3, 3, 32, 3], [self.batch_size, self.weight*2, self.height*2, 32], 1)
                rain2 = prelu(rain2)
            with tf.variable_scope('rnn4'):#1.68kb
                rain4 = deconv_layer(
                    self.down_4, [3, 3, 32, 3], [self.batch_size, self.weight, self.height, 32], 1)
                rain4 = prelu(rain4)

            #long_connection4 = res_in4 = rain4
            #long_connection2 = res_in2 = rain2
            long_connection = rain1
            with tf.variable_scope('residual_memory'):#1.68kb
			    ## RMM
			    ############ rain4
			    ############ rain4
                with tf.variable_scope('rain4_res1'):#1.68kb
                    res4_lstmin = deconv_layer(
                        rain4, [3, 3, 32, 32], [self.batch_size, self.weight, self.height, 32], 1)
                    res4_lstmin = prelu(res4_lstmin)
                with tf.variable_scope('lstm_group4'):
                    y_4, rnn_state4 = cell4(res4_lstmin, rnn_state4)
                with tf.variable_scope('rain4_res2'):#1.68kb
                    res4_lstmout = deconv_layer(
                        y_4, [3, 3, 32, 32], [self.batch_size, self.weight, self.height, 32], 1)
                    res4_lstmout = prelu(res4_lstmout)
                res4_lstmout += rain4
			    ############ rain2
			    ############ rain2
                with tf.variable_scope('upRMM4_2'):#1.68kb
                    rain4to2 = deconv_layer(
                        res4_lstmout, [3, 3, 32, 32], [self.batch_size, self.weight*2, self.height*2, 32], 2)
                    rain4to2 = prelu(rain4to2)
                # with tf.variable_scope('t_to2'):#1.68kb
                    # rain2 = deconv_layer(
                        # tf.concat([rain2, rain4to2],3), [1, 1, 64, 128], [self.batch_size, self.weight*2, self.height*2, 64], 1)
                    # rain2 = prelu(rain2)
                rain2_e = rain2 + rain4to2
                with tf.variable_scope('rain2_res1'):#1.68kb
                    res2_lstmin = deconv_layer(
                        rain2_e, [3, 3, 32, 32], [self.batch_size, self.weight*2, self.height*2, 32], 1)
                    res2_lstmin = prelu(res2_lstmin)
                with tf.variable_scope('lstm_group2'):
                    y_2, rnn_state2 = cell2(res2_lstmin, rnn_state2)
                with tf.variable_scope('rain2_res2'):#1.68kb
                    res2_lstmout = deconv_layer(
                        y_2, [3, 3, 32, 32], [self.batch_size, self.weight*2, self.height*2, 32], 1)
                    res2_lstmout = prelu(res2_lstmout)
                res2_lstmout += rain2
			    ############ rain
			    ############ rain
                with tf.variable_scope('upRMM4_1'):#1.68kb
                    rain4to2 = deconv_layer(
                        rain4to2, [3, 3, 32, 32], [self.batch_size, self.weight*4, self.height*4, 32], 2)
                    rain4to2 = prelu(rain4to2)
                with tf.variable_scope('upRMM2_1'):#1.68kb
                    rain2to1 = deconv_layer(
                        res2_lstmout, [3, 3, 32, 32], [self.batch_size, self.weight*4, self.height*4, 32], 2)
                    rain2to1 = prelu(rain2to1)
                # with tf.variable_scope('t_to1'):#1.68kb
                    # rain1 = deconv_layer(
                        # tf.concat([rain1, rain2to1, rain4to2],3), [1, 1, 64, 192], [self.batch_size, self.weight*4, self.height*4, 64], 1)
                    # rain1 = prelu(rain1)
                rain1_e = rain1 + rain2to1 + rain4to2
                with tf.variable_scope('rain_res1'):#1.68kb
                    res_lstmin = deconv_layer(
                        rain1_e, [3, 3, 32, 32], [self.batch_size, self.weight*4, self.height*4, 32], 1)
                    res_lstmin = prelu(res_lstmin)
                with tf.variable_scope('lstm_group'):
                    y_1, rnn_state = cell(res_lstmin, rnn_state)
                with tf.variable_scope('rain_res2'):#1.68kb
                    res_lstmout = deconv_layer(
                        y_1, [3, 3, 32, 32], [self.batch_size, self.weight*4, self.height*4, 32], 1)
                    res_lstmout = prelu(res_lstmout)
                res_lstmout += rain1

            u4_long_connection = res4_in = res4_lstmout
            u2_long_connection = res2_in = res2_lstmout
            u_long_connection = res_in = res_lstmout
            for n in range(10):# 1296+864+432kb
                ## BCM
                ############ rain4
                ############ rain4
                with tf.variable_scope('BCM4_{}'.format(n)):#1.68kb
                    with tf.variable_scope('down4_{}'.format(n+1)):
                        x_rnn = conv_layer(res4_in, [3, 3, 32, 16], 2) 
                        x_rnn = prelu(x_rnn)
                    res_short = res_input = x_rnn
                    for m in range(1):
                        with tf.variable_scope('group_{}_RCAB{}'.format(n+1,m+1)):
                            res_input = self.RCAB(res_input, 4)
                    with tf.variable_scope('up_{}'.format(n+1)):
                        res_out4 = deconv_layer(
                            tf.concat([res_short, res_input],3), [3, 3, 32, 32], [self.batch_size, self.weight, self.height, 32], 2)#2
                        res_out4 = prelu(res_out4)
                    # u4_cont = tf.concat([u4_long_connection, res_out4],3)
                    # with tf.variable_scope('rnn4_con_{}'.format(n)):#144
                        # res4_mem_out = deconv_layer(
                            # u4_cont, [3, 3, 64, 128], [self.batch_size, self.weight, self.height, 64], 1)#2
                        # res4_mem_out = prelu(res4_mem_out)
                    res4_in += res_out4

                ############ rain2
                ############ rain2
                with tf.variable_scope('BCM2_{}'.format(n)):#1.68kb
                    with tf.variable_scope('upBCM4_2_{}'.format(n)):#1.68kb
                        rain4_resto2 = deconv_layer(
                            res4_in, [3, 3, 32, 32], [self.batch_size, self.weight*2, self.height*2, 32], 2)
                        rain4_resto2 = prelu(rain4_resto2)
                    # with tf.variable_scope('t_to2_{}'.format(n)):#1.68kb
                        # res2_in = deconv_layer(
                            # tf.concat([res2_in, rain4_resto2],3), [1, 1, 64, 128], [self.batch_size, self.weight*2, self.height*2, 64], 1)
                        # res2_in = prelu(res2_in)
                    res2_in_e = res2_in + rain4_resto2
                    with tf.variable_scope('down2_{}'.format(n+1)):
                        x_rnn = conv_layer(res2_in_e, [3, 3, 32, 16], 2) 
                        x_rnn = prelu(x_rnn)
                    res_short = res_input = x_rnn
                    for m in range(1):
                        with tf.variable_scope('group_{}_RCAB{}'.format(n+1,m+1)):
                            res_input = self.RCAB(res_input, 4)
                    with tf.variable_scope('up_{}'.format(n+1)):
                        res_out2 = deconv_layer(
                            tf.concat([res_short, res_input],3), [3, 3, 32, 32], [self.batch_size, self.weight*2, self.height*2, 32], 2)#2
                        res_out2 = prelu(res_out2)
                    # u2_cont = tf.concat([u2_long_connection, res_out],3)
                    # with tf.variable_scope('rnn2_con_{}'.format(n)):#144
                        # res2_mem_out = deconv_layer(
                            # u2_cont, [3, 3, 64, 128], [self.batch_size, self.weight*2, self.height*2, 64], 1)#2
                        # res2_mem_out = prelu(res2_mem_out)
                    res2_in += res_out2

                ############ rain
                ############ rain
                with tf.variable_scope('BCM_{}'.format(n)):#1.68kb
                    with tf.variable_scope('upBCM4to1_{}'.format(n)):#1.68kb
                        rain4_resto1 = deconv_layer(
                            rain4_resto2, [3, 3, 32, 32], [self.batch_size, self.weight*4, self.height*4, 32], 2)
                        rain4_resto1 = prelu(rain4_resto1)
                    with tf.variable_scope('upBCM2to1_{}'.format(n)):#1.68kb
                        rain2_resto1 = deconv_layer(
                            res2_in, [3, 3, 32, 32], [self.batch_size, self.weight*4, self.height*4, 32], 2)
                        rain2_resto1 = prelu(rain2_resto1)
                    # with tf.variable_scope('t_to1_{}'.format(n)):#1.68kb
                        # res_in = deconv_layer(
                            # tf.concat([res_in, rain2_resto1, rain4_resto1],3), [1, 1, 64, 192], [self.batch_size, self.weight*4, self.height*4, 64], 1)
                        # res_in = prelu(res_in)
                    res_in_e = res_in + rain4_resto1 + rain2_resto1
                    with tf.variable_scope('down_{}'.format(n+1)):
                        x_rnn = conv_layer(res_in_e, [3, 3, 32, 16], 2) 
                        x_rnn = prelu(x_rnn)
                    res_short = res_input = x_rnn
                    for m in range(1):
                        with tf.variable_scope('group_{}_RCAB{}'.format(n+1,m+1)):
                            res_input = self.RCAB(res_input, 4)
                    with tf.variable_scope('up_{}'.format(n+1)):
                        res_out = deconv_layer(
                            tf.concat([res_short, res_input],3), [3, 3, 32, 32], [self.batch_size, self.weight*4, self.height*4, 32], 2)#2
                        res_out = prelu(res_out)
                    # u_cont = tf.concat([u_long_connection, res_out],3)
                    # with tf.variable_scope('rnn_con_{}'.format(n)):#144
                        # res_mem_out = deconv_layer(
                            # u_cont, [3, 3, 64, 128], [self.batch_size, self.weight*4, self.height*4, 64], 1)#2
                        # res_mem_out = prelu(res_mem_out)
                    res_in += res_out
                    
            ################ rememory
            with tf.variable_scope('rnn_recon4'):#144
                res4_in = deconv_layer(
                    tf.concat([res4_in, res4_lstmout],3), [3, 3, 32, 64], [self.batch_size, self.weight, self.height, 32], 1)#2
                res4_in = prelu(res4_in)
            with tf.variable_scope('rnn_recon2'):#144
                res2_in = deconv_layer(
                    tf.concat([res2_in, res2_lstmout],3), [3, 3, 32, 64], [self.batch_size, self.weight*2, self.height*2, 32], 1)#2
                res2_in = prelu(res2_in)
            with tf.variable_scope('rnn_recon'):#144
                res_in = deconv_layer(
                    tf.concat([res_in, res_lstmout],3), [3, 3, 32, 64], [self.batch_size, self.weight*4, self.height*4, 32], 1)#2
                res_in = prelu(res_in)
                    
            #lstm_output = lstm_in + lstm_input
            with tf.variable_scope('up4_2'):#144
                res4_2 = deconv_layer(
                    res4_in, [3, 3, 32, 32], [self.batch_size, self.weight*2, self.height*2, 32], 2)#2
                res4_2 = prelu(res4_2)
            with tf.variable_scope('up2_1'):#144
                res42_1 = deconv_layer(
                    tf.concat([res4_2, res2_in],3), [3, 3, 32, 64], [self.batch_size, self.weight*4, self.height*4, 32], 2)#2
                res42_1 = prelu(res42_1)
            res_mem_con = tf.concat([res_in, res42_1],3)#res_in + res42_1#
            with tf.variable_scope('rnn5'):#144
                res_mem_con = deconv_layer(
                    res_mem_con, [3, 3, 16, 64], [self.batch_size, self.weight*4, self.height*4, 16], 1)#2
                #x_rnn = pixel_shuffle_layerg(x_rnn, 2, 64) # n_split = 256 / 2 ** 2
                res_mem_con = prelu(res_mem_con)
            #res_longrange = tf.concat([res_mem_con, long_connection],3)
            with tf.variable_scope('rnn7'):
                res_rainimage = deconv_layer(
                    res_mem_con, [3, 3, 3, 16], [self.batch_size, self.weight*4, self.height*4, 3], 1)#2
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

