import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas import DataFrame 
import pandas as pd 
import seaborn as sns 
import math 
from scipy.stats import norm

# Import and clean data
data_y2 = pd.read_csv("database_y2.csv")
# Remove data values with repeating/incorrect data (actual data values/names removed for privacy reasons)
data_y2= data_y2[data_y2.MP68_MaskID != 263953]
data_y2 = data_y2[data_y2.MP68_MaskID != 540567]
# Create dummy variables 
sex = pd.get_dummies(data_y2['sex'], drop_first=True) 
siblingsDiabetesType = pd.get_dummies(data_y2['siblingsDiabetesType'],drop_first=True) 
probio = pd.get_dummies(data_y2['probio'], drop_first=True) 
data_y2 = pd.concat([data_y2, sex,siblingsDiabetesType,probio], axis = 1)


######
# Exploratory data analysis
#####

