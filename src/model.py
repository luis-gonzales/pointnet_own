import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Conv2D, BatchNormalization, Dense, Dropout


class TNet(Layer):
    def __init__(self, add_regularization=False, bn_momentum=0.99, **kwargs):
        super(TNet, self).__init__(**kwargs)
        self.add_regularization = add_regularization
        self.bn_momentum = bn_momentum
        self.conv0 = CustomConv(64, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.conv1 = CustomConv(128, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.conv2 = CustomConv(1024, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.fc0 = CustomDense(512, activation=tf.nn.relu, apply_bn=True, bn_momentum=bn_momentum)
        self.fc1 = CustomDense(256, activation=tf.nn.relu, apply_bn=True, bn_momentum=bn_momentum)

    def build(self, input_shape):
        self.batch_size, _, self.K = input_shape

        self.w = self.add_weight(shape=(256, self.K**2), initializer=tf.zeros_initializer,
                                 trainable=True, name='w')
        self.b = self.add_weight(shape=(self.K, self.K), initializer=tf.zeros_initializer,
                                 trainable=True, name='b')

        # Initialize bias with identity
        I = tf.constant(np.eye(self.K), dtype=tf.float32)
        self.b = tf.math.add(self.b, I)

    def call(self, x, training=None):
        input_x = x                                                     # BxNxK

        # Embed to higher dim
        x = tf.expand_dims(input_x, axis=2)                             # BxNx1xK
        x = self.conv0(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = tf.squeeze(x, axis=2)                                       # BxNx1024

        # Global features
        x = tf.reduce_max(x, axis=1)                                    # Bx1024

        # Fully-connected layers
        x = self.fc0(x, training=training)                              # Bx512
        x = self.fc1(x, training=training)                              # Bx256

        # Convert to KxK matrix to matmul with input
        x = tf.expand_dims(x, axis=1)                                   # Bx1x256
        x = tf.matmul(x, self.w)                                        # Bx1xK^2
        x = tf.squeeze(x, axis=1)
        x = tf.reshape(x, (-1, self.K, self.K))

        # Add bias term (initialized to identity matrix)
        x += self.b

        # Add regularization
        if self.add_regularization:
            eye = tf.constant(np.eye(self.K), dtype=tf.float32)
            x_xT = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
            reg_loss = tf.nn.l2_loss(eye - x_xT)
            self.add_loss(1e-3 * reg_loss)

        return tf.matmul(input_x, x)

    def get_config(self):
        config = super(TNet, self).get_config()
        config.update({
            'add_regularization': self.add_regularization,
            'bn_momentum': self.bn_momentum})

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomConv(Layer):
    def __init__(self, filters, kernel_size, strides, padding='valid', activation=None,
                 apply_bn=False, bn_momentum=0.99, **kwargs):
        super(CustomConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding=padding,
                           activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.conv(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomConv, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomDense(Layer):
    def __init__(self, units, activation=None, apply_bn=False, bn_momentum=0.99, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.dense = Dense(units, activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.dense(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_model(batch_size, bn_momentum, training):
    pt_cloud = Input(shape=(None, 3), dtype=tf.float32, name='pt_cloud')    # BxNx3

    # Input transformer (B x N x 3 -> B x N x 3)
    pt_cloud_transform = TNet(bn_momentum=bn_momentum)(pt_cloud, training)

    # Embed to 64-dim space (B x N x 3 -> B x N x 64)
    pt_cloud_transform = tf.expand_dims(pt_cloud_transform, axis=2)         # for weight-sharing of conv
    hidden_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                           bn_momentum=bn_momentum)(pt_cloud_transform, training=training)
    embed_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                          bn_momentum=bn_momentum)(hidden_64, training=training)
    embed_64 = tf.squeeze(embed_64, axis=2)

    # Feature transformer (B x N x 64 -> B x N x 64)
    embed_64_transform = TNet(bn_momentum=bn_momentum, add_regularization=True)(embed_64, training)

    # Embed to 1024-dim space (B x N x 64 -> B x N x 1024)
    embed_64_transform = tf.expand_dims(embed_64_transform, axis=2)
    hidden_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                           bn_momentum=bn_momentum)(embed_64_transform, training=training)
    hidden_128 = CustomConv(128, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(hidden_64, training=training)
    embed_1024 = CustomConv(1024, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(hidden_128, training=training)
    embed_1024 = tf.squeeze(embed_1024, axis=2)

    # Global feature vector (B x N x 1024 -> B x 1024)
    global_descriptor = tf.reduce_max(embed_1024, axis=1)

    # FC layers to output k scores (B x 1024 -> B x 40)
    hidden_512 = CustomDense(512, activation=tf.nn.relu, apply_bn=True,
                             bn_momentum=bn_momentum)(global_descriptor, training=training)
    hidden_512 = Dropout(rate=0.3)(hidden_512, training)

    hidden_256 = CustomDense(256, activation=tf.nn.relu, apply_bn=True,
                             bn_momentum=bn_momentum)(hidden_512, training=training)
    hidden_256 = Dropout(rate=0.3)(hidden_256, training)

    logits = CustomDense(40, apply_bn=False)(hidden_256, training=training)

    return Model(inputs=pt_cloud, outputs=logits)
