import os
import argparse

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

if __name__ == '__main__':
    path_ = '../train_logs/deep_SVDD/1'
    mkdir_auto(path_)
