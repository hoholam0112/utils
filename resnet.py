import tensorflow as tf

# Building Blocks for ResNet
def building_block_v1(x, nb_filters, filter_shape, strides, is_train, init_fn, projection_shortcut = None):
    """
    Args:
        nb_filters = int, number of convolutional filters
        filter_shape = int or tuple of ints, (filter_height, filter_width)
        strides = int or tuple of ints, strides
        is_train = boolean tensor used for batch normalization or dropout
        init_fn = weights initialization function
        projection_shortcut = function for fitting input shape with output shape
    Returns:
        output tensor = (batch_size, out_height, out_width, out_channel)
    """
    if projection_shortcut is not None:
        shortcut = projection_shortcut(x)
    else:
        shortcut = x

    h = tf.layers.conv2d(x, nb_filters, filter_shape, strides=strides, padding='same',
            kernel_initializer=init_fn, use_bias=False)
    h = tf.layers.batch_normalization(h, training=is_train)
    h = tf.nn.relu(h)

    h = tf.layers.conv2d(h, nb_filters, filter_shape, strides=1, padding='same',
            kernel_initializer=init_fn, use_bias=False)
    h = tf.layers.batch_normalization(h, training=is_train)
    output = tf.nn.relu(h + shortcut)
    return output

def block_layer(x, nb_filters, filter_shape, first_strides, nb_blocks, block_fn, bottleneck, is_train, init_fn):
    """ Block layer consisting of some building blocks
    Args:
        x = input tensor with shape (batch_size, height, width, channel)
        nb_filters = int, number of filters of building blocks
        filter_shape = int or tuple of ints, filter shape
        first_strides = int or tuple of ints, strides for the first building block of this block layer
        nb_blocks = int, number of building blocks that constitute this block layer
        block_fn = func, building block function
        bottleneck = bool, whether to use bottleneck layer or not
        is_train = boolean tensor for batch norm and dropout
        init_fn = weight initialization function
    Returns:
        output tensor with shape (batch_size, height, width, channel)
    """
    if bottleneck: # output channel of this block layer
        output_channel = 4*nb_filters
    else:
        output_channel = nb_filters

    def projection_shortcut(_input):
        return tf.layers.conv2d(inputs=_input, filters=output_channel, kernel_size=1, strides=first_strides,
                padding='same', kernel_initializer=init_fn, use_bias=False)

    # First Building Block
    h = block_fn(x, nb_filters, filter_shape, first_strides, is_train, init_fn, projection_shortcut)
    # Other Building Blocks
    for _ in range(1, nb_blocks):
        h = block_fn(h, nb_filters, filter_shape, 1, is_train, init_fn, None)
    return h
