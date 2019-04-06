import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

file = '/Users/luisgonzales/Downloads/ModelNet40/chair/train/chair_0001.off'

#header = np.fromfile(file, dtype=np.int32, count=5)
#print(header)
#with open(file, 'r') as fp:
fp = open(file, 'r')

line = fp.readline()
print(line)

line = fp.readline()
print(line)


for _ in range(2300):

	line = fp.readline()[:-1]	# don't want new line char
	#print(line)
	#print(type(line))

	#print( line.split(' ') )

	x, y, z = [float(val) for val in line.split(' ')]
	#print(x,y,z)
	ax.scatter(x,y,z)


plt.show()
