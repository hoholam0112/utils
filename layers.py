import tensorflow as tf

def _apply_activ_fn(h, activ_fn_name):
    if activ_fn_name:
        if activ_fn_name == 'relu':
            output = tf.nn.relu(h)
        elif activ_fn_name == 'lrelu':
            output = tf.maximum(0.1*h, h)
        elif activ_fn_name == 'sigmoid':
            output = tf.nn.sigmoid(h)
        elif activ_fn_name == 'tanh':
            output = tf.nn.tanh(h)
        else:
            raise ValueError('Invaild activation function name is passed')
    else:
        output = h
    return output

def dense(x, output_dim, is_train, activ_fn = None, batch_norm=False, dropout_rate=None, init_fn=None):
    """
    Args:
        x (Tensor): input for layer
        output_dim (int): output dimension
        is_train (bool): whether training is performed using the output
        batch_norm (bool): append batch norm layer before activation
        dropout_rate (float): dropout rate. If not provided, not performed
        init_fn (tf.initializer): initialization function
    """
    if dropout_rate:
        h = tf.layers.dropout(x, rate=dropout_rate, training=is_train)
    else:
        h = x

    if batch_norm:
        h = tf.layers.dense(h, output_dim, kernel_initializer=init_fn, use_bias=False)
        h = tf.layers.batch_normalization(h, training=is_train)
    else:
        h = tf.layers.dense(h, output_dim, kernel_initializer=init_fn)

    return _apply_activ_fn(h, activ_fn)

def conv2d(x, nb_filters, filter_shape, strides, is_train, padding='same', activ_fn = None,
        batch_norm=False, dropout_rate=None, init_fn=None):
    """
    Args:
        x (Tensor)
        nb_filters (int) = the number of convolution filters
        filter_shape ([height, width])
        stride ([height_stride, width_stride])
    """
    if dropout_rate:
        h = tf.layers.dropout(x, rate=dropout_rate, training=is_train)
    else:
        h = x

    if batch_norm:
        h = tf.layers.conv2d(h, nb_filters, filter_shape, strides, padding=padding,
                kernel_initializer=init_fn, use_bias=False)
        h = tf.layers.batch_normalization(h, training=is_train)
    else:
        h = tf.layers.conv2d(h, nb_filters, filter_shape, strides, padding=padding,
                kernel_initializer=init_fn)

    return _apply_activ_fn(h, activ_fn)

def conv2d_trans(x, nb_filters, filter_shape, strides, is_train, padding='same', activ_fn = None,
        batch_norm=False, dropout_rate=None, init_fn=None):
    """
    Args:
        x (Tensor)
        nb_filters (int) = the number of convolution filters
        filter_shape ([height, width])
        stride ([height_stride, width_stride])
    """
    if dropout_rate:
        h = tf.layers.dropout(x, rate=dropout_rate, training=is_train)
    else:
        h = x

    if batch_norm:
        h = tf.layers.conv2d_transpose(h, nb_filters, filter_shape, strides, padding=padding,
                kernel_initializer=init_fn, use_bias=False)
        h = tf.layers.batch_normalization(h, training=is_train)
    else:
        h = tf.layers.conv2d_transpose(h, nb_filters, filter_shape, strides, padding=padding,
                kernel_initializer=init_fn)

    return _apply_activ_fn(h, activ_fn)

