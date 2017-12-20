import numpy as np
import tensorflow as tf
import vgg16
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import time
from tqdm import trange
from scipy.misc import imsave
import sqlite3
from scipy import ndimage


def get_annotations(filename):
    conn = sqlite3.connect(filename)
    cur = conn.cursor()
    cur.execute("SELECT * FROM bounding_boxes")
    rows = cur.fetchall()
    return rows


def init_vgg(sess, width, height):
    opt_img = tf.Variable( tf.truncated_normal( [1,width, height, 3],
                                            dtype=tf.float32,
                                            stddev=1e-1), name='opt_img' )
    tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )
    vgg = vgg16.vgg16(tmp_img, '/home/dallin/deep_learning_data/vgg_weights/vgg16_weights.npz', sess)
    return vgg

def image_grabber(image_file):
    image = imread(image_file)
    return image

def crop_image(image, data, width, height):
    x1 = int(data[2])
    y1 = int(data[3])
    x2 = int(data[4])
    y2 = int(data[5])
    x = int((x1 + x2)/2) - int(width/2)
    y = int((y1 + y2)/2) - int(height/2)
    img_cropped = image[y:y+height, x:x+width]
    return img_cropped


# There are 53 target id's in the dataset.
# Average width is 75.6 pixels, median is 76
# Average height is 170.3 pixels, median is 194.

def get_targets(i, data, width, height):
    image_file = '/home/dallin/deep_learning_data/Dataset_atrium/atrium_frames/' + '%08d' % (i+1) + '.jpg'
    bgs_file = '/home/dallin/deep_learning_data/Dataset_atrium/atrium_bgs/' + '%08d' % i + '.png'
    targets = []
    truths = []
    if i in data[:,1]:
        data_row = np.where(data[:,1] == i)
        for j in range(0,len(data_row[0])):
            data_index = data_row[0][j]
            anno_h = int(data[data_index,5] - data[data_index,3])
            anno_w = int(data[data_index,4] - data[data_index,2])
            if anno_h >= height and anno_w >= width:
                img_full = image_grabber(image_file)
                img_bgs = image_grabber(bgs_file)
                img_shape = img_full.shape
                x = int(data[data_index, 2])
                y = int(data[data_index, 3])
                if anno_w + x <= img_shape[1] and anno_h + y <= img_shape[0]:
                    img_t = crop_image(img_full, data[data_index], width, height)
                    img_t_bgs = crop_image(img_bgs, data[data_index], width, height)
                    targets.append(img_t)
                    truths.append(img_t_bgs)
    return targets, truths


def get_id(i, data):
    id_nums = []
    if i in data[:, 1]:
        data_row = np.where(data[:, 1] == i)
        for j in range(0, len(data_row[0])):
            id_nums.append(data[data_row, 0])
    return id_nums


height = 170
width = 60

sess = tf.Session()

data_file = '/home/dallin/deep_learning_data/Dataset_atrium/atrium_annotations/atrium_gt.sqlite'
data = np.array(get_annotations(data_file))
data_list = get_annotations(data_file)

vgg = init_vgg(sess, width, height)

layers = [ 'conv1_1', 'conv1_2',
           'conv2_1', 'conv2_2',
           'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3',
           'conv5_1', 'conv5_2', 'conv5_3' ]
conv_ops = [ getattr( vgg, x ) for x in layers ]

for i in range(0,4539):
    targets = get_targets(i, data, width, height)
    target_ids = get_id(i, data)



# TODO: Simply train on targets for the first 10 target or so and then test. This will be good enough.
# TODO: Setup classification and training network.
# You can flip the image and add gaussian noise to train on a batch greater than one.
# You should probably wait for at least 5 tracks or so before you start checking for classification.
# People: White shirt girl, black, red, green, blueish, black shirt white guy, white shirt guy, black shirt girl.

print('hello')

