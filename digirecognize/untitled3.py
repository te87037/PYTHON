# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 08:12:19 2016

@author: acer
"""

import FukuML.PLA as pla
pla_bc = pla.BinaryClassifier()
pla_bc.load_train_data('train.csv')
print (pla_bc)