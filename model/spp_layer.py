import numpy as np
import math
import tensorflow as tf

# input feature maps is of the form: N-C-(WH)/(HW)
# ex. spatial_pyramid:
#	[[1, 1], [2, 2], [3, 3], [4, 5]]
# each row is a level of pyramid with nxm pooling
def np_spatial_pyramid_pooling(input_feature_maps, spatial_pyramid, dtype=np.float32):
	assert input_feature_maps.ndim == 4
	assert spatial_pyramid.ndim == 2
	assert spatial_pyramid.shape[1] == 2

	batch_size = input_feature_maps.shape[0]
	num_channels = input_feature_maps.shape[1]
	h = input_feature_maps.shape[2]
	w = input_feature_maps.shape[3]

	num_levels = spatial_pyramid.shape[0]

	# N-C-W*H
	flattened_feature_maps = np.reshape(input_feature_maps, (batch_size, num_channels, -1))
	num_px = flattened_feature_maps.shape[2]

	bins_per_level = np.prod(spatial_pyramid, axis=1)
	num_bins = np.sum(bins_per_level)
	stack = []

	# stride tricks, then max pool along one dimension 
	# then stride tricks again and max pool along the other dimension
	# but whats the length and stride?
	# ceil(w/n) for window size, floor(w/n) for stride,
	# where w is the original dim, and n is the number of bins along the dim 
	# but this implementation may leave out some pixels (consider w = 5, n = 3)

	sizeof_item = np.dtype(dtype).itemsize

	for i in range(num_levels):
		n_h = spatial_pyramid[i][0]
		n_w = spatial_pyramid[i][1]
		
		l = math.ceil(w/n_w)
		s = math.floor(w/n_w)

		ar = np.lib.stride_tricks.as_strided(flattened_feature_maps, (batch_size, num_channels, h, n_w, l), 
			(sizeof_item*num_px*num_channels, sizeof_item*num_px, sizeof_item*w, sizeof_item*s, sizeof_item))
		ar = np.transpose(np.amax(ar, axis=4), (0, 1, 3, 2)).copy()

		l = math.ceil(h/n_h)
		s = math.floor(h/n_h)

		ar = np.lib.stride_tricks.as_strided(ar, (batch_size, num_channels, n_w, n_h, l), 
			(sizeof_item*n_w*h*num_channels, sizeof_item*n_w*h, sizeof_item*h, sizeof_item*s, sizeof_item))
		ar = np.transpose(np.amax(ar, axis=4), (0, 1, 3, 2))

		# for debugging purposes, monitor "ar" here, before flattening

		stack.append(np.reshape(ar, (batch_size, num_channels, -1)))

	stack = np.concatenate(stack, axis=2)
	#print(stack.shape)

	return stack

def tf_spatial_pyramid_pooling(tf_input_feature_maps, tf_spatial_pyramid, dtype=tf.float32):
	return tf.py_func(np_spatial_pyramid_pooling, [tf_input_feature_maps, tf_spatial_pyramid], dtype)
