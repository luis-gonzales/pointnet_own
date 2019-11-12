""" Project at: https://app.wandb.ai/lrg/pointnet_own """
import argparse
from glob import glob
from time import time
from datetime import timezone, datetime

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
PARSER.add_argument('--epochs', type=int, default=200, help='Number of epochs')
PARSER.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
PARSER.add_argument('--optimizer', type=str, default='adam', help='Either Adam or SGD (case-insensitive)')
PARSER.add_argument('--wandb', action='store_true', default=False, help='Whether to use wandb')
ARGS = PARSER.parse_args()

BATCH_SIZE = ARGS.batch_size
EPOCHS = ARGS.epochs
LEARNING_RATE = ARGS.learning_rate
LR_DECAY_STEPS = 5700
LR_DECAY_RATE = 0.7
OPTIMIZER = ARGS.optimizer
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
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(tf_parse_filename, num_parallel_calls=AUTOTUNE)
print('Done!')


# Create model
def get_bn_momentum(step):
    return min(0.99, 1.5 + 0.005*step)
print('Creating model...')
bn_momentum = tf.Variable(get_bn_momentum(0))
model = get_model(bn_momentum=bn_momentum)
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
exp_decay = ExponentialDecay(LEARNING_RATE, LR_DECAY_STEPS, LR_DECAY_RATE, staircase=True)
if OPTIMIZER.lower() == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=exp_decay.get_next)
elif OPTIMIZER.lower() == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=exp_decay.get_next)
loss_fxn = tf.keras.losses.BinaryCrossentropy(from_logits=True) # uses sigmoid_cross_entropy


# Instantiate metric objects
train_acc = tf.keras.metrics.CategoricalAccuracy()
val_acc = tf.keras.metrics.CategoricalAccuracy()
train_prec = tf.keras.metrics.Precision()
train_recall = tf.keras.metrics.Recall()
val_prec = tf.keras.metrics.Precision()
val_recall = tf.keras.metrics.Recall()


# Training
print('Training...')
print('Steps per epoch =', len(TRAIN_FILES) // BATCH_SIZE)
print('Total steps =', (len(TRAIN_FILES) // BATCH_SIZE) * EPOCHS)

@tf.function
def train_step(inputs, labels):
    # Forward pass with gradient tape and loss calc
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = loss_fxn(labels, logits) + sum(model.losses)

    # Obtain gradients of trainable vars w.r.t. loss and perform gradient descent
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return logits, loss, model.losses[0]

@tf.function
def val_step(inputs):
    logits = model(inputs, training=False)
    return logits

step = 0
for epoch in range(EPOCHS):
    print('\nEpoch', epoch)

    # Reset metrics
    train_acc.reset_states()
    val_acc.reset_states()
    train_prec.reset_states()
    train_recall.reset_states()
    val_prec.reset_states()
    val_recall.reset_states()

    # Train on batches
    for x_train, y_train in train_ds:
        tic = time()

        train_logits, train_loss, mat_reg_loss = train_step(x_train, y_train)

        train_probs = tf.math.sigmoid(train_logits)
        train_acc.update_state(y_train, train_probs)

        max_idxs = tf.math.argmax(train_probs, axis=1)
        train_one_hot = tf.one_hot(max_idxs, depth=40, dtype=tf.float32)
        train_prec.update_state(y_train, train_one_hot)
        train_recall.update_state(y_train, train_one_hot)

        if WANDB:
            wandb.log({'time_per_step': time() - tic,
                       'learning_rate': exp_decay.peek(),
                       'training_loss': train_loss.numpy(),
                       'mat_reg_loss': mat_reg_loss.numpy(),
                       'bn_momentum': bn_momentum.numpy()}, step=step)
        step += 1
        bn_momentum.assign(get_bn_momentum(step))

    # Run validation at the end of epoch
    for x_val, y_val in val_ds:
        val_logits = val_step(x_val)

        val_probs = tf.math.sigmoid(val_logits)
        val_acc.update_state(y_val, val_probs)

        max_idxs = tf.math.argmax(val_probs, axis=1)
        val_one_hot = tf.one_hot(max_idxs, depth=40, dtype=tf.float32)
        val_prec.update_state(y_val, val_one_hot)
        val_recall.update_state(y_val, val_one_hot)

    # Save every epoch (.save_weights() since bn_momentum instance isn't serializable)
    print('model.save_weights() at step', step)
    model.save_weights('model/checkpoints/' + INIT_TIMESTAMP + '/iter-' + str(step), save_format='tf')

    if WANDB:
        wandb.log({'train_accuracy': train_acc.result().numpy(),
                   'train_precision': train_prec.result().numpy(),
                   'train_recall': train_recall.result().numpy(),
                   'val_accuracy': val_acc.result().numpy(),
                   'val_precision': val_prec.result().numpy(),
                   'val_recall': val_recall.result().numpy()}, step=step)

print('Done training!')
