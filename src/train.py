import tensorflow as tf

from model import get_model


print('Building model...')
model, embed_transformer = get_model()
print('Done!')
model.summary()

print(model)
print(embed_transformer)


# Load generator


# Perform training
adam = tf.keras.optimizers.Adam(lr=0.001)
