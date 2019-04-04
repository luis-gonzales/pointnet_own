import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, Dense, Lambda, BatchNormalization, Dropout


def conv_wrapper(x, filters, kernel_sz, strides, activation, bn, bn_momentum):
	'''
	Handles convolution operation, BN, and activation.
	Input assumed to be of shape (B, N, 1, D).
	'''
	x = Conv2D(filters=filters, kernel_size=kernel_sz, strides=strides,
			   padding='valid', kernel_initializer=tf.initializers.he_normal())(x)

	if bn:
		x = BatchNormalization(axis=-1, momentum=bn_momentum, trainable=True)(x)
	
	if activation.lower() == 'relu':
		x = ReLU()(x)
	# else, assume linear activation

	return x


def fc_wrapper(x, units, activation, bn, bn_momentum):
	'''
	Handles FC layer, BN, and activation.
	Input assumed to be of shape (B, D)
	'''
	x = Dense(units=units, kernel_initializer=tf.initializers.he_normal())(x)

	if bn:
		x = BatchNormalization(axis=-1, momentum=bn_momentum, trainable=True)(x)

	if activation.lower() == 'relu':
		x = ReLU()(x)

	return x


def expand_dim(x):
	#import tensorflow as tf
	return tf.expand_dims(x, axis=2)	# (B, N, 1, K)

def max_pool(x):
	return tf.reduce_max(x, axis=1)	# (B, 1024)

def squeeze(x):
	return tf.squeeze(x, axis=2)

def mat_mult(x):
	a, b = x[0], x[1]
	return tf.matmul(a, b)	# (B, N, 3)

class MatMult(tf.keras.layers.Layer):
	# made into layer because of trainable variables
	def __init__(self, K):
		super(MatMult, self).__init__()
		self.w = tf.get_variable(name='weights_' + str(K), shape=(256,K*K),
								 dtype=tf.float32, trainable=True,
							  	 initializer=tf.zeros_initializer())
		self.b = tf.get_variable(name='biases_' + str(K), shape=(K*K),
								 dtype=tf.float32, trainable=True,
								 initializer=tf.zeros_initializer)
		I = tf.constant(np.eye(K).flatten(), dtype=tf.float32)
		self.b = tf.math.add(self.b, I)

	def call(self, x):
		return tf.matmul(x, self.w)

#biases = tf.get_variable(name='biases_' + str(K), shape=(K*K),
	#						 dtype=tf.float32,
	#						 initializer=tf.zeros_initializer(), trainable=True)
	#I = tf.constant(np.eye(K).flatten(), dtype=tf.float32)
	#biases = tf.math.add(biases, I)
	#biases = tf.keras.layers.Add()([biases, I])
'''
class Linear(layers.Layer):

  def __init__(self, units=32, input_dim=32):
    super(Linear, self).__init__()
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                              dtype='float32'),
                         trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                              dtype='float32'),
                         trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b
'''

def t_net(input_pts):
	'''
	T-Net from PointNet and inspired by Spatial Transformer.
	Map point cloud to higher-dim space, obtain global feature
	vector (max pool), lower dim with FC, and mat mult to get (B, K, K)
	'''
	K = input_pts.get_shape()[-1]

	# Expand dim in order to leverage Conv2D operation for MLP
	#input_pts = tf.expand_dims(input_pts, axis=2)	# (B, N, 1, K)
	input_pts = Lambda(expand_dim)(input_pts)

	# Embed to 1024-dim space with MLP (implemented using Conv2D)
	embed_64 = conv_wrapper(input_pts, 64, (1,1), (1,1), 'ReLU', bn=True, bn_momentum=0.9)
	embed_128 = conv_wrapper(embed_64, 128, (1,1), (1,1), 'ReLU', bn=True, bn_momentum=0.9)
	embed_1024 = conv_wrapper(embed_128, 1024, (1,1), (1,1), 'ReLU', bn=True, bn_momentum=0.9)
	#embed_1024 = tf.squeeze(embed_1024, axis=2)		# (B, N, K)
	embed_1024 = Lambda(squeeze)(embed_1024)

	# Obtain global feature vector
	#global_feat = tf.reduce_max(embed_1024, axis=1)	# (B, 1024)
	global_feat = Lambda(max_pool)(embed_1024)
	
	# Two FC layers to compress to 256-dim vector
	fc_512 = fc_wrapper(global_feat, 512, 'ReLU', bn=True, bn_momentum=0.9)
	fc_256 = fc_wrapper(fc_512, 256, 'ReLU', bn=True, bn_momentum=0.9)

	
	#weights = tf.get_variable(name='weights_' + str(K), shape=(256,K*K),
	#						  dtype=tf.float32,
	#						  initializer=tf.zeros_initializer(), trainable=True)
	#print( np.zeros((4,2)) )
	#weights = np.zeros((256,9), dtype=np.float32)
	#weights = tf.keras.backend.variable(np.zeros((256,K*K)), dtype=tf.float32)

	#biases = tf.get_variable(name='biases_' + str(K), shape=(K*K),
	#						 dtype=tf.float32,
	#						 initializer=tf.zeros_initializer(), trainable=True)
	#I = tf.constant(np.eye(K).flatten(), dtype=tf.float32)
	#biases = tf.math.add(biases, I)
	#biases = tf.keras.layers.Add()([biases, I])

	#transform = Lambda(mat_mult)([fc_256, weights])
	#transform = tf.matmul(fc_256, weights) #+ biases	# (B, 9)
	transform = MatMult(K)(fc_256)
	
	#return tf.reshape(transform, [-1, K, K])		# (B, K, K)
	test = tf.keras.layers.Reshape((K, K))(transform)
	print('test')
	print(test)
	
	return test
	#return tf.keras.layers.Reshape((K, K))		# (B, K, K)

#def pointnet(pt_cld):
def get_model():
	'''
	Map points to higher-dim space (with help from T-Net), obtain
	global feature vector, and pass through FC layers for class scores.
	Also output transformation matrix from 64-dim embedded for regularization.
	'''
	pt_cld = Input(shape=(None,3), dtype=tf.float32, name='pt_cloud')	# Nx3

	# Input transform (and expand dims to leverage Conv2D operation for MLP)
	pt_transformer = t_net(pt_cld)							# (B, 3, 3)
	
	#pt_cld_transformed = tf.matmul(pt_cld, pt_transformer)	# (B, N, 3)
	pt_cld_transformed = Lambda(mat_mult)([pt_cld, pt_transformer])
	#pt_cld_transformed = tf.expand_dims(pt_cld_transformed, axis=2)	# (B, N, 1, 3)
	pt_cld_transformed = Lambda(expand_dim)(pt_cld_transformed)
	
	# Embed to 64-dim space
	hidden_64 = conv_wrapper(pt_cld_transformed, 64, (1,1), (1,1), 'ReLU',
							 bn=True, bn_momentum=0.9)
	embed_64 = conv_wrapper(hidden_64, 64, (1,1), (1,1), 'ReLU',
							bn=True, bn_momentum=0.9)
	#embed_64 = tf.squeeze(embed_64, axis=2)					# (B, N, 64)
	embed_64 = Lambda(squeeze)(embed_64)
	
	# Feature transformation (and expand dims)
	embed_transformer = t_net(embed_64)						# (B, 64, 64)
	#embed_64_transformed = tf.matmul(embed_64, embed_transformer)
	embed_64_transformed = Lambda(mat_mult)([embed_64, embed_transformer])
	#embed_64_transformed = tf.expand_dims(embed_64_transformed, axis=2) # (B, N, 1, 64)
	embed_64_transformed = Lambda(expand_dim)(embed_64_transformed)
	
	# Embed to 1024-dim space
	hidden_64 = conv_wrapper(embed_64_transformed, 64, (1,1), (1,1), 'ReLU',
							 bn=True, bn_momentum=0.9)
	hidden_128 = conv_wrapper(hidden_64, 128, (1,1), (1,1), 'ReLU',
							  bn=True, bn_momentum=0.9)
	embed_1024 = conv_wrapper(hidden_128, 1024, (1,1), (1,1), 'ReLU',
							  bn=True, bn_momentum=0.9)
	#embed_1024 = tf.squeeze(embed_1024, axis=2)				# (B, N, 1024)
	embed_1024 = Lambda(squeeze)(embed_1024)
	
	# Obtain global feature vector
	#global_feat = tf.reduce_max(embed_1024, axis=1)			# (B, 1024)
	global_feat = Lambda(max_pool)(embed_1024)
	
	# FC layers to output k scores
	hidden_512 = fc_wrapper(global_feat, 512, 'ReLU', bn=True, bn_momentum=0.9)
	hidden_512 = Dropout(rate=0.3)(hidden_512)
	
	hidden_256 = fc_wrapper(hidden_512, 256, 'ReLU', bn=True, bn_momentum=0.9)
	hidden_256 = Dropout(rate=0.3)(hidden_256)

	scores_40 = fc_wrapper(hidden_256, 40, 'linear', bn=False, bn_momentum=None)
	
	#return scores_40, embed_transformer
	return Model(inputs=pt_cld, outputs=scores_40), embed_transformer
	#return Model(inputs=pt_cld, outputs=scores_40), embed_transformer

'''
def get_model():
	
	Wrap pointnet in Lambda layer to satisfy tf.keras.models.Model.
	Output Keras model and transformation matrix from 64-dim embedded for
	regularization.
	
	pt_cld = Input(shape=(None,3), dtype=tf.float32, name='pt_cloud')	# Nx3
	scores, embed_transformer = Lambda(pointnet)(pt_cld)

	return Model(inputs=pt_cld, outputs=scores), embed_transformer
'''
