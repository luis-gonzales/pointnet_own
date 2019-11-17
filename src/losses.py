import tensorflow as tf


def anchor_loss(y_true, y_pred, gamma=0.5):
    pred_prob = tf.math.sigmoid(y_pred)

    # Obtain probabilities at indices of true class
    true_mask = tf.dtypes.cast(y_true, dtype=tf.bool)
    q_star = tf.boolean_mask(pred_prob, true_mask)
    q_star = tf.expand_dims(q_star, axis=1)

    # Calculate bce and add anchor loss coeff where labels equal 0
    loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    M = 1.0 - y_true
    loss_calc = (M * (1.0 + pred_prob - q_star + 0.05)**gamma + (1.0 - M)) * loss_bce

    return tf.math.reduce_mean(loss_calc)
