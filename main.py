import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas import DataFrame 
import pandas as pd 
import seaborn as sns 
import math 
import statsmodels.formula.api as smf
from scipy.stats import norm

# Import and clean data (data removed for privacy reasons)
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


######
# Logisitic regression and multipliative linear interaction term 
######

# Create array of all variables 
variables_t1d = np.array(['FatherDiabetesType', 'Male', 'MotherDiabetesType', 'acute_sinusitis', 'before1m', 'c_section', 'common_cold_tot_day','country_cd', 'fathers_age', 'fdr', 'fevergrp_tot_day','gastro_tot_day','grs1', 'hla_category', 'influenza_tot_day', 'laryngitis_trac_tot_day', 'maternal_age', 'persist_conf_ab', 'race_ethnicity', 'resp_gest_inf','resp_tot_day', 'start_daycare_yr1', 'unknown', 'weight','yes'])

# Create matrix of p values for the multiplicative linear interaction term of logistic regression 
p_values_matrix_twointer = np.zeros([len(variables_t1d), len(variables_t1d)])

for i in range(len(variables_t1d)):
    for j in range(len(variables_t1d)):
        if i == j:
            model_ij= smf.logit(formula="t1d ~" + variables_t1d[i], data= data_y2).fit()
            p_values_matrix_twointer[i,j]= model_ij.pvalues[1]
        else:
            try:
                model_ij= smf.logit(formula="t1d ~" + variables_t1d[i] +"+"+ variables_t1d[j] + "+" + variables_t1d[i]+":"+variables_t1d[j], data= data_y2).fit()
                p_values_matrix_twointer[i,j]= model_ij.pvalues[3]
            except :
                #print("2")
                p_values_matrix_twointer[i,j] = 1.01 
print(p_values_matrix_twointer) 

# Set labels for matrix
label_variables_t1d = np.array(['Father Diabetes Status','Male','Mother Diabetes Status','Acute sinusitis episodes','Probiotics before 1 month old ','Caesarean section','Common cold episodes','Country','Paternal age','First Dgree Relative','Fever episodes','Gastrointestinal infections ','Genetic risk score','HLA gene category', 'Influenza episodes','Laryngitis tracheitis episodes', 'Maternal age', 'Number of persistent antibodies ','Race / ethnicity','Maternal gestational respiratory infections',  'Respiratory infection episodes','Day-care before age 1', 'No Probiotics', 'Weight','Sibling Diabetes Status']) 

# Plot matrix 
plt.figure(figsize = (40,30))
p = sns.heatmap(p_values_matrix_twointer, annot=True,  linewidths=.5, yticklabels= label_variables_t1d, annot_kws={"size": 14},vmin=0, vmax=0.05) 
plt.tick_params(axis='both', which='major',labelsize=20, labelbottom = False, bottom=False, top = False, labeltop=True)
p.set_xticklabels(labels = label_variables_t1d, rotation= 90)
p.set_title("A matrix showing the p-values of the multiplicative interaction terms in the logistic regression\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n", fontsize=24, fontweight="bold")
#plt.savefig('MATRIX_0.05.pdf')

# Just lower LHS triangle 
trilow = np.tril(p_values_matrix_twointer)
trilow[np.triu_indices(trilow.shape[0], k = 1)] = np.nan

# Bonferroni corretion: divide the critical value by the number of repeats = 0.05/300


######
# Gradient boosting model and H statistic 
######


