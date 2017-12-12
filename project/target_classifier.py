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
    vgg = vgg16.vgg16(tmp_img, 'vgg16_weights.npz', sess)
    return vgg


sess = tf.Session()

data_file = '/home/dallin/Dropbox/Deep_Learning/Project/Dataset_atrium/atrium_annotations/atrium_gt.sqlite'
data = get_annotations(data_file)

vgg = init_vgg(sess)



print('hello')

