import tensorflow as tf

def _PS(X, r, n_out_channel):
        if n_out_channel >= 1:
            assert int(X.get_shape()[-1]) == (r ** 2) * n_out_channel, _err_log
            bsize, a, b, c = X.get_shape().as_list()
            bsize = tf.shape(X)[0] # Handling Dimension(None) type for undefined batch dim
            # X = tf.cast(X, tf.int32)
            Xs = tf.split(X, r, 3) #b*h*w*r*r dtype
            # Xs = tf.split(X, r, 3)
            Xr = tf.concat(Xs, 2) #b*h*(r*w)*r
            X = tf.reshape(Xr, (bsize, r*a, r*b, n_out_channel)) # b*(r*h)*(r*w)*c
        else:
            print(_err_log)
        return X