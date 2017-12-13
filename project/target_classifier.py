import numpy as np
import tensorflow as tf
import vgg16
from scipy.misc import imread, imresize
import time
from tqdm import trange
from scipy.misc import imsave
import sqlite3

def get_annotations(filename):
    conn = sqlite3.connect(filename)
    cur = conn.cursor()
    cur.execute("SELECT * FROM bounding_boxes")
    rows = cur.fetchall()
    return rows


def init_vgg(sess):
    height = 200
    width = 300
    opt_img = tf.Variable( tf.truncated_normal( [1,height, width,3],
                                            dtype=tf.float32,
                                            stddev=1e-1), name='opt_img' )
    tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )
    vgg = vgg16.vgg16(tmp_img, '/home/dallin/deep_learning_data/vgg_weights/vgg16_weights.npz', sess)
    return vgg


sess = tf.Session()

data_file = '/home/dallin/deep_learning_data/Dataset_atrium/atrium_annotations/atrium_gt.sqlite'
data = get_annotations(data_file)

vgg = init_vgg(sess)

layers = [ 'conv1_1', 'conv1_2',
           'conv2_1', 'conv2_2',
           'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3',
           'conv5_1', 'conv5_2', 'conv5_3' ]
conv_ops = [ getattr( vgg, x ) for x in layers ]

# There are 52 target id's in the dataset.

# TODO: Bring in images, one at a time.
# TODO: Set bounding box to upper left corner and make it a fixed width and height.
# Average width is 75.6 pixels, median is 76
# Average height is 170.3 pixels, median is 194.



# TODO: Crop both the image and bgs image to the bounding box
# TODO: If the target ID is new, then query the classification network if it has a high probability of being a known class.
# TODO: If low probability, train on the image for 30 frames?
# TODO: Setup classification and training network.

print('hello')

