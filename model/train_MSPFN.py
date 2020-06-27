import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('../utils')
from layer import *
from MSPFN import MODEL
import load_rain
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

start_learning_rate = 5e-4#
batch_size = 12

def train():
    x = tf.placeholder(tf.float32, [None, 96, 96, 3])#128,96
    x_rain = tf.placeholder(tf.float32, [None, 96, 96, 3])#128,96
    is_training = tf.placeholder(tf.bool, [])

    model = MODEL(x, x_rain, is_training, batch_size)
    sess = tf.Session()
    with tf.variable_scope('MSPFN'):
        global_step = tf.Variable(0, name='global_step', trainable=False)#6000,1e-5
    opt = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(start_learning_rate,global_step, 20000, decay_rate=0.8,staircase=False)+1e-6)
    train_op = opt.minimize(model.train_loss, global_step=global_step, var_list=model.variables)
    init = tf.global_variables_initializer() 
    sess.run(init)
    
    # Restore the MSPFN network
    if tf.train.get_checkpoint_state('MSPFN/'):
        saver = tf.train.Saver()
        saver.restore(sess, 'MSPFN/epoch50')

    # Load the data
    x_train, x_test, x_train_rain, x_test_rain = load_rain.load()

    # Train the MSPFN model
    n_iter = int(len(x_train) / batch_size)
    while True:
        epoch = int(sess.run(global_step) / n_iter ) + 1#2
        print('epoch:', epoch)
        for i in tqdm(range(n_iter)):
            x_batch = normalize(x_train[i*batch_size:(i+1)*batch_size])
            x_batch_rain = normalize(x_train_rain[i*batch_size:(i+1)*batch_size])
            j = np.random.randint(0,8,size=1)
            x_batch = rotate(x_batch,j)
            x_batch_rain = rotate(x_batch_rain,j)
            train_loss, edge_loss, _ = sess.run([model.train_loss, model.edge_loss, train_op], feed_dict={x: x_batch, x_rain: x_batch_rain, is_training: True})
            format_str = ('epoch: %d, train_loss: %.5f, edge_loss: %.7f')
            print((format_str % (epoch, train_loss, edge_loss)))
                
        # Save the model
        saver = tf.train.Saver()
        save_path = 'MSPFN'
        if epoch>0:
            saver.save(sess, os.path.join(save_path, 'epoch{}'.format(epoch)), write_meta_graph=False)
            
        # Validate
        # raw = normalize(x_test[:batch_size])
        # raw_rain = normalize(x_test_rain[:batch_size])
        # # rain_4, rain_2, rain_ori, fake = sess.run(
            # # [model.rain_4, model.rain_2, model.rain_ori, model.imitation],
            # # feed_dict={x: raw, x_rain: raw_rain, is_training: False})
        # # save_img([raw_rain, rain_4, rain_2, rain_ori, fake, raw], ['rain', 'rain_4', 'rain_2', 'rain_ori', 'clean', 'Ground Truth'], epoch)
        # fake = sess.run(
            # [model.imitation],
            # feed_dict={x: raw, x_rain: raw_rain, is_training: False})
        # save_img([raw_rain, fake, raw], ['rain', 'clean', 'Ground Truth'], epoch)

def count_model_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('  + Number of params: %.2fM' % (total_parameters / 1e6))

def save_img(imgs, label, epoch):
    for i in range(batch_size):
        fig = plt.figure()
        for j, img in enumerate(imgs):
            #im = np.uint8((img[i]+1)*127.5)
            im = np.uint8(np.clip((img[i]+1)*127.5,0,255.0))
            #print('imshape:',im.shape)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            fig.add_subplot(1, len(imgs), j+1)
            plt.imshow(im)
            plt.tick_params(labelbottom='off')
            plt.tick_params(labelleft='off')
            plt.gca().get_xaxis().set_ticks_position('none')
            plt.gca().get_yaxis().set_ticks_position('none')
            plt.xlabel(label[j])
        seq_ = "{0:09d}".format(i+1)
        epoch_ = "{0:09d}".format(epoch)
        path = os.path.join('result', seq_, '{}.png'.format(epoch_))
        if os.path.exists(os.path.join('result', seq_)) == False:
            os.mkdir(os.path.join('result', seq_))
        plt.savefig(path)
        plt.close()


def normalize(images):
    return np.array([image/127.5 - 1 for image in images])


if __name__ == '__main__':
    train()
    count_model_params()
