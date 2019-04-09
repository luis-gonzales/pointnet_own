import argparse
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser()
parser.add_argument('npy', help='Absolute path to npy file', type=str)
FLAGS = parser.parse_args()

npy_file = FLAGS.npy


def plot_pcd(file):

	data = np.load(file)

	fig = plt.figure()
	ax  = fig.add_subplot(111, projection='3d')

	for pt in data:
		ax.scatter(pt[0], pt[1], pt[2], s=1, c='black')

	# Same range to improve visualization
	ax.set_xlim(-1.0, 1.0)
	ax.set_ylim(-1.0, 1.0)
	ax.set_zlim(-1.0, 1.0)

	plt.show()


plot_pcd(npy_file)
