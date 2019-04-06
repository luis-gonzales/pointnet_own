import os
import h5py
import numpy as np
#import pandas as pd

from sklearn import preprocessing


def off_to_ply(path, categories, DataGroup):
    for cat in categories:
        DataArray=[]
        #deal with train first
        files = os.listdir(path + cat + '/' + DataGroup + '/')
        print('files =', files)
        #files = [x for x in files if x[-4:] == '.off']
        files = [x for x in files if x.endswith('.off')]
        print('files =', files)
        for file in files:
            fileName = file.split('.')[0]
            print('open =', path + cat + '/' + DataGroup + '/' + file)
            with open(path + cat + '/' + DataGroup + '/' + file, 'r') as f:
                tmp=f.readline().replace('\n','')
                line=''
                if tmp !='OFF':
                    line = tmp[3:]
                else:
                    line = f.readline().replace('\n','')
                
                #get number of points in the model
                point_count = line.split(' ')[0]
                face_count = line.split(' ')[1]
            
                data = []
                #fill ndarray with datapoints
                for index in range(0,int(point_count)):
                    line = f.readline().rstrip().split()
                    line[0] = float(line[0])
                    line[1] = float(line[1])
                    line[2] = float(line[2])
                    data.append(line)
                data = np.array(data)
                #normalize data before conversion
                centroid = np.mean(data, axis=0)
                data = data - centroid
                m = np.max(np.sqrt(np.sum(data**2, axis=1)))
                data = data / m
                
                #create ply file,write in header first.
                with open(path + cat + '/' + DataGroup + '/' + fileName + ".ply",'w') as plyFile:
                    plyFile.write('ply\nformat ascii 1.0\nelement vertex ')
                    plyFile.write(point_count)
                    plyFile.write('\nproperty float32 x\nproperty float32 y\nproperty float32 z\nelement face ')
                    plyFile.write(face_count)
                    plyFile.write('\nproperty list uint8 int32 vertex_indices\nend_header\n')
                    for index in range(0,int(point_count)):
                        plyFile.write(' '.join(map(str, data[index])))
                        plyFile.write('\n')
                    for index in range(0,int(face_count)):
                        plyFile.write(f.readline())

categories = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']
#categories = []
#totalcat=[]
#with open('categories.txt','r') as catego:
#    content =catego.readlines()
    #categories = [w.replace('\n', '') for w in content]
#    totalcat = [w.replace('\n', '') for w in content]
#path = 'c:\\Users\\sean_\\Downloads\\ModelNet10\\'
path = '/Users/luisgonzales/Downloads/ModelNet40/'

off_to_ply(path, categories, 'train')
