__all__ = ['load_olf_faithful']

import pandas as pd
import numpy as np
import os


def load_old_faithful():
    '''
    Load Old Faithful Geyser data
    '''
    curdir = os.getcwd()
    filepath = '/'.join([curdir,'old_faithful.txt']) 
    data = pd.read_csv(filepath)
    return {'X': np.array(data[['eruptions','waiting']]),
            'columns': data.columns}


