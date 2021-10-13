import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_addons as tfa


def tf_median_blur(filter_shape):
    def median_filter2d(image):
        return tfa.image.median_filter2d(
            image, filter_shape=filter_shape, padding='REFLECT'
        )

    return median_filter2d


def gms(x_true, y_pred, c=0.0026):
    x = tf.reduce_mean(x_true, axis=-1, keepdims=True)
    y = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
    g_true = tfio.experimental.filter.prewitt(tf_median_blur((3, 3))(x))
    g_pred = tfio.experimental.filter.prewitt(tf_median_blur((3, 3))(y))
    g_map = (2 * g_true * g_pred + c) / (g_true ** 2 + g_pred ** 2 + c)
    gms_loss = 1 - tf.reduce_mean(g_map)
    return gms_loss


def msgms(x_true, y_pred):
    total_loss = gms(x_true, y_pred)
    for _ in range(3):
        x_true = tf.nn.avg_pool2d(x_true, ksize=2, strides=2, padding='SAME')
        y_pred = tf.nn.avg_pool2d(y_pred, ksize=2, strides=2, padding='SAME')
        total_loss += gms(x_true, y_pred)

    return total_loss / 4