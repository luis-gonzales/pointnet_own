""" Project at: https://app.wandb.ai/lrg/pointnet_own """
import argparse
from glob import glob
from time import time
from datetime import timezone, datetime

import numpy as np
import tensorflow as tf

from model import get_model
from dataset_utils import tf_parse_filename, train_val_split

def get_timestamp():
    timestamp = str(datetime.now(timezone.utc))[:16]
    timestamp = timestamp.replace('-', '')
    timestamp = timestamp.replace(' ', '_')
    timestamp = timestamp.replace(':', '')
    return timestamp

tf.random.set_seed(0)


# CLI
PARSER = argparse.ArgumentParser(description='CLI for training pipeline')
PARSER.add_argument('--batch_size', type=int, default=32, help='Batch size per step')
PARSER.add_argument('--epochs', type=int, default=100, help='Number of epochs')
PARSER.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
PARSER.add_argument('--optimizer', type=str, default='adam', help='Either Adam or SGD (case-insensitive)')
PARSER.add_argument('--checkpt_freq', type=int, default=500, help='Freq of checkpt and validation')
PARSER.add_argument('--wandb', action='store_true', default=False, help='Whether to use wandb')
ARGS = PARSER.parse_args()

BATCH_SIZE = ARGS.batch_size
EPOCHS = ARGS.epochs
LEARNING_RATE = ARGS.learning_rate
LR_DECAY_STEPS = 5700
LR_DECAY_RATE = 0.7
OPTIMIZER = ARGS.optimizer
CHECKPT_FREQ = ARGS.checkpt_freq
WANDB = ARGS.wandb
INIT_TIMESTAMP = get_timestamp()
if WANDB:
    import wandb
    wandb.init(project='pointnet_own', name=INIT_TIMESTAMP)


# Create datasets (.map() after .batch() due to lightweight mapping fxn)
print('Creating train and val datasets...')
TRAIN_FILES, VAL_FILES = train_val_split()
TEST_FILES = glob('ModelNet40/*/test/*.npy')   # only used to get length for comparison
print('Number of training samples:', len(TRAIN_FILES))
print('Number of validation samples:', len(VAL_FILES))
print('Number of testing samples:', len(TEST_FILES))
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = tf.data.Dataset.list_files(TRAIN_FILES)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(tf_parse_filename, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = tf.data.Dataset.list_files(VAL_FILES)
val_ds = val_ds.repeat().batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(tf_parse_filename, num_parallel_calls=AUTOTUNE)
val_ds = iter(val_ds)
print('Done!')


# Create model
print('Creating model...')
bn_momentum = tf.Variable(0.5)
# test_var = tf.keras.backend.variable(value=0.5, dtype=tf.float32, name='test_var')
model = get_model(bn_momentum=0.99)
# model = get_model(bn_momentum=bn_momentum)
print('Done!')
model.summary()


# Instantiate optimizer and loss function
class ExponentialDecay():
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, staircase=False):
        self.initial_lr = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.step = -1
        self.current = None
    def get_next(self):
        self.step += 1
        if not self.staircase:
            coeff = self.decay_rate ** (self.step / self.decay_steps)
        else:
            coeff = self.decay_rate ** (self.step // self.decay_steps)
        self.current = self.initial_lr * coeff
        return self.current
    def peek(self):
        return self.current
exp_decay_obj = ExponentialDecay(LEARNING_RATE, LR_DECAY_STEPS, LR_DECAY_RATE, staircase=True)
if OPTIMIZER.lower() == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=exp_decay_obj.get_next)
elif OPTIMIZER.lower() == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=exp_decay_obj.get_next)
loss_fxn = tf.keras.losses.BinaryCrossentropy(from_logits=True) # sigmoid_cross_entropy


# Training
print('Training...')
print('Steps per epoch =', len(TRAIN_FILES) // BATCH_SIZE)
print('Total steps =', (len(TRAIN_FILES) // BATCH_SIZE) * EPOCHS)
step = 0
for epoch in range(EPOCHS):
    print('Epoch =', epoch)
    for x_train, y_train in train_ds:
        tic = time()

        # Forward pass with gradient tape and loss calc
        with tf.GradientTape() as tape:
            logits = model(x_train, training=True)
            loss = loss_fxn(y_train, logits) + sum(model.losses)

        # Obtain gradients of trainable vars w.r.t. loss and perform gradient descent
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        if WANDB:
            wandb.log({'learning_rate': exp_decay_obj.peek(),
                       'training_loss': loss.numpy(),
                       'time_per_step': time() - tic,
                       'mat_reg_loss': model.losses[0].numpy()}, step=step)

        if step % CHECKPT_FREQ == 0:
            # np.save('pt_cloud-' + str(step), x_train[0, :, :])
            # np.save('logits-' + str(step), logits[0, :])

            # test_pt_cloud = np.load('ModelNet40/airplane/test/airplane_0627.npy')
            # test_pt_cloud = np.expand_dims(test_pt_cloud, axis=0)
            # print('test_pt_cloud:', test_pt_cloud.shape)
            # test_inf = model(test_pt_cloud, training=False)
            # def sigmoid(x):
            #     return 1.0 / (1.0 + np.exp(-x))
            # np.save('test_inf_sigmoid-' + str(step), sigmoid(test_inf))
            # np.save('test_inf-' + str(step), test_inf)

            print('checkpoint at step', step)
            model.save('model/checkpoints/' + INIT_TIMESTAMP + '/iter-' + str(step), save_format='tf')

            x_val, y_val = next(val_ds)
            val_logits = model(x_val, training=False)
            val_loss = loss_fxn(y_val, val_logits)
            val_loss += sum(model.losses)

            if WANDB:
                wandb.log({'val_loss': val_loss.numpy(),
                           'mat_reg_val_loss': model.losses[0].numpy()}, step=step)
        step += 1
    print('\n')
print('Done training!')
