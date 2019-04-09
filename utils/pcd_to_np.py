import os
import argparse
import numpy as np

from glob import glob
from helpers import norm_pts
from sklearn.utils import shuffle


parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='Absolute path to ModelNet40', type=str)
parser.add_argument('-c', '--category', help='Specific category for conversion (e.g., )', type=str)
parser.add_argument('-n', '--n_samples', help='Number of pt cloud samples [default=1024]', default=1024, type=int)
FLAGS = parser.parse_args()


path = FLAGS.data_path
if not path.endswith('/'): path += '/'		# correct for '/'
path_len = len(path)

if not FLAGS.category:						# if no input arg, do all
    categories = glob(path + '*/')
else:
    categories = [path + FLAGS.category + '/']

n_samples = FLAGS.n_samples


def fn_to_np(file, sample_pts=1024):
	with open(file, 'r') as fh:

		# Skip through first part of header
		for _ in range(9): fh.readline()

		# Grab number of pt cloud points; skip line
		num_pts = fh.readline().rstrip().split(' ')[1]
		num_pts = int(num_pts)
		fh.readline()

		# Read in point cloud
		data = []
		for _ in range(num_pts):
			pts = [float(x) for x in fh.readline().rstrip().split(' ')]
			data.append(pts)
	
	# Shuffle data (needed due to PCL software), grab subset, normalize
	data = np.array(shuffle(data))
	return norm_pts(data[:sample_pts, :])	# shape "frame" normalized; now normalize pt cloud


def pcd_to_np(paths, path_len, n_samples):

	for cur_path in paths:						# e.g., abs/lamp/

		# Create category directory
		try:
			os.mkdir('data/ModelNet40/' + cur_path[path_len:])
		except:
			pass
		
		for cur_dir in glob(cur_path + '/*/'):	# e.g., abs/lamp/train/

			# Create train/test directory
			try:
				os.mkdir('data/ModelNet40/' + cur_dir[path_len:])
			except:
				pass

			# Step through each file in pcd/
			for file in glob(cur_dir + 'pcd/*.pcd'):
				pts = fn_to_np(file, n_samples)

				# Save pts to numpy file
				f_name = file[file.rfind('/')+1:]	# e.g., bottle_0001.pcd

				np_fn = 'data/ModelNet40/' + cur_dir[path_len:] + \
						f_name.replace('pcd', 'npy')

				np.save(np_fn, pts)


pcd_to_np(categories, path_len, n_samples)
