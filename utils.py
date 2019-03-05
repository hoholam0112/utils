import os
import argparse
import tensorflow as tf
import pandas as pd

def mkdir_auto(path):
    """ Make directories automatically """
    dir_list = []
    temp_path = path
    while not os.path.isdir(temp_path):
        split = temp_path.split('/')
        dir_list.append(split[-1])
        temp_path = '/'.join(split[:-1])

    while len(dir_list) != 0:
        temp_path = os.path.join(temp_path, dir_list.pop())
        os.mkdir(temp_path)

def str2bool(input_str):
    if input_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean Value Expected')

def ask():
    while True:
        query = input('Train history already exists. Do you really want to restart? (y/n)\n>>')
        res = query[0].lower()
        if query == '' or not res in ['y', 'n']:
            pass
        else:
            break

    if res == 'y':
        return True
    else:
        return False

def update_dataframe(file_path, args, which):
    """ Update dataframe by using parsed arguments
    Args:
        file_path: str, csv file generated from pandas dataframe object
        args: namespace, parsed arguments
        which: str, which type to be written 'train' or 'test'
    Return:
        df: updated pandas dataframe object
    """
    assert which == 'train' or which == 'test'
    def args_to_dict(args, append):
        """ Returns Dictionary From Arguments
        Args:
            args: Namespace object, parsed arguments
            append: Bool, Whether to be append to pandas dataframe
        Returns:
            args_dict: Dictionary to be append to pandas dataframe
        """
        args_dict = {}
        for key, value in args.__dict__.items():
            if key != 'dname':
                if append:
                    args_dict[key] = value
                else:
                    args_dict[key] = [value]

        if which == 'train':
            args_dict['progress'] = 0
        return args_dict

    if not os.path.exists(file_path):
        df = pd.DataFrame(args_to_dict(args, False))
    else:
        df = pd.read_csv(file_path)
        if which == 'train'  and args.eid in list(df['eid']):
            raise ValueError('Experiment \'{}\' already exists'.format(args.eid))
        df = df.append(args_to_dict(args, True), ignore_index=True)
    return df

# Utility functions for Tensorflow 
scalar_pl = {}
for dtype in [tf.int64, tf.float32]:
    scalar_pl[dtype] = tf.placeholder(dtype, [])
def scalar_variable(name, dtype, init_val, update_type):
    """ Create a Tensorflow scalar variable
    Args:
        name: str, a name of a Tensorflow variable
        dtype: Tensorflow datatype
        init_val: An initial value
        update_type: str, which update type to be used
    Returns:
        var: tf.Variable
        update_fn: function, when be called, update
        the variable depending on a given update_type argument
    """
    assert update_type == 'assign' or update_type == 'increment'
    var = tf.get_variable(name, [], dtype, tf.constant_initializer(init_val),
            trainable=False)
    assign_op = tf.assign(var, scalar_pl[dtype])
    increment_op = tf.assign_add(var, 1)

    if update_type == 'assign':
        def assign_fn(sess, v):
            v_ = sess.run(assign_op, {scalar_pl[dtype]:v})
        return var, assign_fn
    else:
        def increment_fn(sess):
            return sess.run(increment_op)
        return var, increment_fn

if __name__ == '__main__':
    path_ = '../train_logs/deep_SVDD/1'
    mkdir_auto(path_)
