from glob import glob

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def tf_parse_filename(filename):
    """Take batch of filenames and create point cloud and label"""

    idx_lookup = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4,
                  'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9,
                  'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14,
                  'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18,
                  'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23,
                  'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28,
                  'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33,
                  'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38,
                  'xbox': 39}

    def parse_filename(filename_batch):

        pt_clouds = []
        labels = []
        for filename in filename_batch:
            # Read in point cloud
            filename_str = filename.numpy().decode()
            pt_cloud = np.load(filename_str)

            # Add rotation and jitter to point cloud
            theta = np.random.random() * 2*3.141
            A = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
            offsets = np.random.normal(0, 0.02, size=pt_cloud.shape)
            pt_cloud = np.matmul(pt_cloud, A) + offsets

            # Create classification label
            obj_type = filename_str.split('/')[1]   # e.g., airplane, bathtub
            label = np.zeros(40, dtype=np.float32)
            label[idx_lookup[obj_type]] = 1.0

            pt_clouds.append(pt_cloud)
            labels.append(label)

        return np.stack(pt_clouds), np.stack(labels)


    x, y = tf.py_function(parse_filename, [filename], [tf.float32, tf.float32])
    return x, y


def tf_parse_filename_test(filename):
    """Take batch of filenames and create point cloud and label"""

    idx_lookup = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4,
                  'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9,
                  'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14,
                  'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18,
                  'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23,
                  'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28,
                  'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33,
                  'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38,
                  'xbox': 39}

    def parse_filename(filename_batch):

        pt_clouds = []
        labels = []
        for filename in filename_batch:
            # Read in point cloud
            filename_str = filename.numpy().decode()
            pt_cloud = np.load(filename_str)

            # Create classification label
            obj_type = filename_str.split('/')[1]   # e.g., airplane, bathtub
            label = np.zeros(40, dtype=np.float32)
            label[idx_lookup[obj_type]] = 1.0

            pt_clouds.append(pt_cloud)
            labels.append(label)

        return np.stack(pt_clouds), np.stack(labels)


    x, y = tf.py_function(parse_filename, [filename], [tf.float32, tf.float32])
    return x, y


def train_val_split(train_size=0.92):
    train, val = [], []
    for obj_type in glob('ModelNet40/*/'):
        cur_files = glob(obj_type + 'train/*.npy')
        cur_train, cur_val = \
            train_test_split(cur_files, train_size=train_size, random_state=0, shuffle=True)
        train.extend(cur_train)
        val.extend(cur_val)

    return train, val
