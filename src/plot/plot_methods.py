#vcompare methods

import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import xgb_analysis as xgb
import methods.nn as nn
import svm
import random_forest as rf


svm_result = svm.svm()
rf_result = rf.rf()
nn_result = nn.nnet()
xgb_result = xgb.xgb()

#plot results
plt.plot(['SVM', 'Random Forest', 'Neural Network', 'XGBoost'], [svm_result, rf_result, nn_result, xgb_result])
plt.title('Root Mean Squared Error (RMSE) of Different Methods')
plt.ylabel('Root Mean Squared Error (RMSE)')

plt.show()
