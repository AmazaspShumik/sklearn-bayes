__all__ = ['load_olf_faithful','load_digits']

import pandas as pd
import numpy as np
import os
import inspect


def get_data(name):
    curdir = inspect.getfile(inspect.currentframe())
    filedir = curdir.split('/')
    filedir[-1] = name
    return pd.read_csv('/'.join(filedir))


def load_old_faithful():
    '''
    Load Old Faithful Geyser data
    '''
    data = get_data('old_faithful.txt')
    return {'X': np.array(data[['eruptions','waiting']]),
            'columns': ['eruptions','waiting']}
               
               
def load_digits():
    data = get_data('digits.csv')
    return data
    
