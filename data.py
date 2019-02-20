from abc import ABC, abstractmethod
from collections import defaultdict

# Interface For DataSet
class DataSet(ABC):
    @abstractmethod
    def get_batch(self): pass # Return a batch of data

    @abstractmethod
    def get_attr_list(self): pass

    @abstractmethod
    def get_shape(self): pass

    @abstractmethod
    def explain(self): pass # Explain a given attribute


def int2onehot(x, length):
    """ Convert Integer to onehot vector """
    assert x > 0 and x < length
    assert isinstance(x, int)
    assert isinstance(length, int)
    ohv = np.zeros([length], dtype=np.float32)
    ohv[x] = 1
    return ohv

def cat2int(x):
    """ Convert Categorical attributes to integers """
    s = set(x)
    v2int = {}
    for i, v in enumerate(unique):
        v2int[v] = i
    assert len(s) >= 2, 'Kind of values is smaller than 2'
    return np.array([v2int[e] for e in x], dtype=np.int32), v2int

def df2dict(df):
    """ Convert pandas.DataFrame to Dictionary """
    attr_list = list(df.columns)
    res_dict = {}
    for attr in attr_list:
        res_dict[attr.lower()] = list(df[attr])

    return res_dict


if __name__ == '__main__':
    class ExampleDataSet(DataSet):
        def __init__(self):
            return

        def get_batch(self):
            return

        def get_attr_list(self):
            return

        def get_shape(self):
            return

        def explain(self, num):
            return

    dset = ExampleDataSet()

