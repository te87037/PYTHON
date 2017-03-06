# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 10:33:22 2017

@author: acer
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 08:50:52 2017

@author: acer
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

df = pd.read_csv("train.csv")

data = df[1:,:]
train_cols = data.columns[1:]
logit = sm.Logit(data['label'], data[train_cols])

result = logit.fit()
print (result.summary())