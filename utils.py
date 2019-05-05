import os, argparse

def str2bool(input_str):
    """ Convert string into boolean """
    if input_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean Value Expected')

def ask(question):
    """ Ask question and then return True or False depending on the answer
    Args:
        question: str, question
    Returns:
        answer: bool
    """
    while True:
        query = input('{}\n Reply (y/n) >>'.format(question))
        res = query[0].lower()
        if query == '' or not res in ['y', 'n']:
            pass
        else:
            break

    if res == 'y':
        return True
    else:
        return False

