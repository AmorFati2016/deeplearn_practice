#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plot

y=np.random.standard_normal((600,2))
plt.figure(figsize=(8,5))
plt.plot(y[:,0],y[:,1],'ro')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')