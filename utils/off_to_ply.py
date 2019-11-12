"""
Most of off_to_ply credited to github.com/seanhuang5104/OFFtoH5/blob/master/OFFtoH5.ipynb
"""

import os
import argparse
from glob import glob

import numpy as np

PARSER = argparse.ArgumentParser()
PARSER.add_argument('data_path', help='Absolute path to ModelNet40', type=str)
PARSER.add_argument('-c', '--category', help='Specific category for conversion (e.g., )', type=str)
FLAGS = PARSER.parse_args()

path = FLAGS.data_path
if not path.endswith('/'):                          # correct for '/'
    path += '/'

if not FLAGS.category:                              # if no input arg, do all
    categories = glob(path + '*/')
else:
    categories = [path + FLAGS.category + '/']

def off_to_ply(categories, group):

    for cat in categories:  # e.g. abs_path/bed/

        # Create ply dir
        ply_dir = cat + group + '/ply/' # e.g., abs_path/bed/train/ply/
        try:
            os.mkdir(ply_dir)
        except:
            pass

        # Process each .off file
        for file in glob(cat + group + '/*.off'):
            with open(file, 'r') as f:
                tmp = f.readline().rstrip()
                if tmp != 'OFF':
                    line = tmp[3:]
                else:
                    line = f.readline().rstrip()
                line = line.split(' ')

                # Get number of vertices and faces
                num_verts = int(line[0])
                num_faces = int(line[1])

                # Fill np.array with x,y,z points
                data = []
                for _ in range(num_verts):
                    line = [float(x) for x in f.readline().rstrip().split(' ')]
                    data.append(line)
                data = np.array(data)

                # Create ply file
                slash_idx = file.rfind('/')
                ply_file = ply_dir + file[slash_idx+1:]
                ply_file = ply_file.replace('off', 'ply')
                with open(ply_file, 'w') as plyFile:

                    # Write header
                    plyFile.write('ply\nformat ascii 1.0\nelement vertex ')
                    plyFile.write(str(num_verts))
                    plyFile.write('\nproperty float32 x\nproperty float32 y\nproperty float32 z\nelement face ')
                    plyFile.write(str(num_faces))
                    plyFile.write('\nproperty list uint8 int32 vertex_indices\nend_header\n')

                    # Accum face text and collect vertex indices that actually used (avoid floating points)
                    face_strs = ''
                    val_idxs = set()
                    for _ in range(num_faces):
                        cur = f.readline()
                        vals = [int(x) for x in cur.rstrip().split(' ')]
                        for i in vals[1:]:
                            val_idxs.add(i)
                        face_strs += cur

                    # Normalize strictly against connected points (val_idxs)
                    filt_data = data[list(val_idxs), :]
                    filt_mean = np.mean(filt_data, axis=0)
                    data_normed = data - filt_mean

                    filt_data = data_normed[list(val_idxs), :]
                    filt_dist = np.max(np.linalg.norm(filt_data, axis=1))
                    data_normed /= filt_dist

                    # Finally, write to ply file
                    for pt in data_normed:
                        plyFile.write(' '.join(map(str, pt)))
                        plyFile.write('\n')

                    plyFile.write(face_strs)


off_to_ply(categories, 'train')
off_to_ply(categories, 'test')
