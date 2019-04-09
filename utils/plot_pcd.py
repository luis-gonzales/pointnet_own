import argparse
import numpy as np
import matplotlib.pyplot as plt

from helpers import norm_pts
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser()
parser.add_argument('pcd', help='Absolute path to pcd file', type=str)
parser.add_argument('-n', '--n_vis', help='Number of pt cloud samples [default=1024]', default=1024, type=int)
FLAGS = parser.parse_args()


pcd_file = FLAGS.pcd
n_vis    = FLAGS.n_vis


def plot_pcd(file, n_vis):

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
	data = norm_pts(data[:n_vis, :])

	fig = plt.figure()
	ax  = fig.add_subplot(111, projection='3d')

	for pt in data:
		ax.scatter(pt[0], pt[1], pt[2], s=1, c='black')

	# Same range to improve visualization
	ax.set_xlim(-1.0, 1.0)
	ax.set_ylim(-1.0, 1.0)
	ax.set_zlim(-1.0, 1.0)

	plt.show()


plot_pcd(pcd_file, n_vis)
