import tensorflow as tf

# Utility functions for Tensorflow 
def scalar_tensor(name, dtype, init_val):
    """ Create a Tensorflow scalar variable
    Args:
        name: str, a name of a Tensorflow variable
        dtype: Tensorflow datatype
        init_val: An initial value
    Returns:
        var: tf.Variable
        assign_fn: function, used for update a variable
    """
    pl = tf.placeholder(dtype, [], name=name+'_pl')
    var = tf.get_variable(name, [], dtype, tf.constant_initializer(init_val),
            trainable=False)
    assign_op = tf.assign(var, pl)

    def assign_fn(sess, v):
        """ assign some value to a variable
        Args:
            sess: tf.Session object
            v: scalar, value to be assigned
        Returns:
            assinged value
        """
        return sess.run(assign_op, {pl:v})

    return var, assign_fn
