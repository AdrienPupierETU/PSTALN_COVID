#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien
"""

from sklearn.utils import class_weight
import numpy as np
def getWeight(Y):
    return class_weight.compute_class_weight('balanced',np.unique(Y),Y)