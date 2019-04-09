import os
import argparse
import numpy as np

from glob import glob
from subprocess import call
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Absolute path to ModelNet40', type=str)
parser.add_argument('-c', '--category', help='Specific category for conversion (e.g., )', type=str)
parser.add_argument('-n', '--n_samples', help='Number of pt cloud samples [default=2048]', default='2048', type=str)
FLAGS = parser.parse_args()


path = FLAGS.data_path
if not path.endswith('/'): path += '/'				# correct for '/'

if not FLAGS.category:                              # if no input arg, do all
    categories = glob(path + '*/')
else:
    categories = [path + FLAGS.category + '/']

n_samples = FLAGS.n_samples
leaf_size = '0.005'


def ply_to_pcd(paths):

	for cat_path in paths:						# e.g., abs/lamp/
		for direct in glob(cat_path + '/*/'):	# e.g., abs/lamp/train/

			# Create pt cloud directory
			try:
				os.mkdir(direct + 'pcd/')
			except:
				pass

			# Step through each file in ply/ and call executable
			for file in glob(direct + 'ply/*.ply'):
				call(['pcl_mesh_sampling', file, file.replace('ply', 'pcd'),
					'-n_samples', n_samples, '-leaf_size', leaf_size,
					'-no_vis_result'])


ply_to_pcd(categories)
