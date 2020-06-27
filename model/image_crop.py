import numpy as np
import math
import tensorflow as tf

# input feature maps is of the form: N-C-(WH)/(HW)
# ex. spatial_pyramid:
#	[[1, 1], [2, 2], [3, 3], [4, 5]]
# each row is a level of pyramid with nxm pooling
def crop(image, scale, dtype=np.float32):
    #assert image.ndim == 4
    batch_size = image.shape[0]
    w = image.shape[1]
    h = image.shape[2]  
    num_channels = image.shape[3] 
    in_img_type = image.dtype
    image.astype(np.float32)
    w_size = w//scale
    h_size = h//scale
    #if in_img_type != np.uint8:
        #image *= 255.
        #image = np.uint8(np.clip((image+1)*127.5,0,255.0))
    concat = []
    for i in range(scale):
        for j in range(scale):
            img = image[:, i*w_size:(i+1)*w_size, j*h_size:(j+1)*h_size, :]
            #concat = tf.concat([img, concat],3)#concat.append(img)
            concat.append(img)
    #concat = np.array(concat)
    img_concat = concat[0]
    for i in range(1,len(concat)):
        img_concat = np.concatenate((img_concat, concat[i]), axis=3)
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]'''
    #if in_img_type == np.uint8:
        #ima_y = ima_y.round()
    #else:
    #ima_y /= 255.
    #ima_y_1 = np.reshape(ima_y, (-1,h,w,1))
    #print(img_concat.shape)
    return img_concat.astype(dtype)

def img_crop(image, scale, dtype=tf.float32):
	return tf.py_func(crop, [image, scale], dtype)
