import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas import DataFrame 
import pandas as pd 
import seaborn as sns 
import math 
from scipy.stats import norm

# Import and clean data (actual data values/names removed for privacy reasons)
data_y2 = pd.read_csv("database_y2.csv")
# Remove data values with repeating/incorrect data 
data_y2= data_y2[data_y2.IDnumber != 1000]
data_y2 = data_y2[data_y2.IDumber != 2000]
# Create dummy variables 
sex = pd.get_dummies(data_y2['sex'], drop_first=True) 
siblingsDiabetesType = pd.get_dummies(data_y2['siblingsDiabetesType'],drop_first=True) 
probio = pd.get_dummies(data_y2['probio'], drop_first=True) 
data_y2 = pd.concat([data_y2, sex,siblingsDiabetesType,probio], axis = 1)

# Remove invalid candidates; related to : those visited clinic recently? 
data_y2.loc[(data_y2['last_clinic_visit_agedys'] < 3285) & (data_y2['t1d']!=1), 'valid_candidate'] = 'InValid'
data_y2.loc[(data_y2['last_clinic_visit_agedys'] >= 3285) & (data_y2['t1d'] ==1), 'valid_candidate'] = 'Valid'
data_y2.loc[(data_y2['last_clinic_visit_agedys'] >= 3285) | (data_y2['t1d'] ==1), 'valid_candidate'] = 'Valid'
data_y2 = data_y2[data_y2.valid_candidate != 'InValid']

# Create dataset with relevant variables according professors in the clinical research facility
data_y2.drop(['sex','siblingsDiabetesType','probio','status','exclude', 'cc', 'positive','maternal','indeterminate', 'last_clinic_visit','last_clinic_visit_agedys','t1d_diag_agedys','persist_conf_agedys','Sex','HLA_Category','HLA_Category_all','HLAscore_grs1','FID','PHENO', 'grs1sntile','grs1strat','grs1strat2','positive_all_time','persist_conf_ab_all_time','last_test','persist_conf_gad','persist_conf_ia2a','persist_conf_miaa','persist_conf_gadclass','persist_conf_ia2aclass','persist_conf_miaaclass','multiple_autoantibody','multiple_autoantibody_all_time','totdrug','imputed_min_bmiz','imputed_mean_bmiz','vitamin_c_mg_l','vitd_nmo_l','valid_candidate'], axis = 1, inplace=True)


######
# Exploratory data analysis
#####

# Plot all variables 
i = 0
for col in data_y2:
    i = i + 1
    plt.subplot(10,3,i)
    data_y2[col].plot.hist(bins = 100, figsize=(100,50))
plt.show()

# Set data for those with T1D vs without T1D 
data_witht1d = data_y2[data_y2.t1d != 0]
data_withoutt1d = data_y2[data_y2.t1d != 1]

# Show that Genetic risk show does not predict T1D accurately 
sns.kdeplot(data_withoutt1d['grs1'],label="No T1D",color='blue',shade=True)
sns.kdeplot(data_witht1d['grs1'], label="T1D",color='r',shade=True)
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4.5})
plt.xlabel('Genetic risk score')
plt.ylabel('Density function')
#plt.savefig('GRS density plot high quality.png',dpi = 1000,bbox_inches = "tight")

