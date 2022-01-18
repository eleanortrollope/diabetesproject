from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_curve, auc
from scipy import stats
from pandas import DataFrame 
from scipy.stats import norm 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 
import math 
import statsmodels.formula.api as smf

# Import and clean data (data removed for privacy reasons)
data_y2 = pd.read_csv("database_y2.csv")

# Remove data values with repeating/incorrect data 
data_y2= data_y2[data_y2.IDnumber != 1000]
data_y2 = data_y2[data_y2.IDumber != 2000]

# Create dummy variables 
sex = pd.get_dummies(data_y2['sex'], drop_first=True) 
siblingsDiabetesType = pd.get_dummies(data_y2['siblingsDiabetesType'],
                                      drop_first=True) 
probio = pd.get_dummies(data_y2['probio'], drop_first=True) 
data_y2 = pd.concat([data_y2, sex,siblingsDiabetesType,probio], axis = 1)

# Remove invalid candidates; related to : those visited clinic recently? 
data_y2.loc[(data_y2['last_clinic_visit_agedys'] < 3285) & (data_y2['t1d']!=1), 
            'valid_candidate'] = 'InValid'
data_y2.loc[(data_y2['last_clinic_visit_agedys'] >= 3285) & (data_y2['t1d'] ==1), 
            'valid_candidate'] = 'Valid'
data_y2.loc[(data_y2['last_clinic_visit_agedys'] >= 3285) | (data_y2['t1d'] ==1), 
            'valid_candidate'] = 'Valid'
data_y2 = data_y2[data_y2.valid_candidate != 'InValid']

# Create dataset with relevant variables according professors in the clinical 
# research facility
data_y2.drop(['sex','siblingsDiabetesType','probio','status','exclude', 'cc',
              'positive','maternal','indeterminate', 'last_clinic_visit',
              'last_clinic_visit_agedys','t1d_diag_agedys','persist_conf_agedys',
              'Sex','HLA_Category','HLA_Category_all','HLAscore_grs1','FID','PHENO', 
              'grs1sntile','grs1strat','grs1strat2','positive_all_time',
              'persist_conf_ab_all_time','last_test','persist_conf_gad',
              'persist_conf_ia2a','persist_conf_miaa','persist_conf_gadclass',
              'persist_conf_ia2aclass','persist_conf_miaaclass',
              'multiple_autoantibody','multiple_autoantibody_all_time','totdrug',
              'imputed_min_bmiz','imputed_mean_bmiz','vitamin_c_mg_l','vitd_nmo_l',
              'valid_candidate'], axis = 1, inplace=True)


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
variables_t1d = np.array(['FatherDiabetesType', 'Male', 'MotherDiabetesType', 
                          'acute_sinusitis', 'before1m', 'c_section',
                          'common_cold_tot_day','country_cd', 'fathers_age',
                          'fdr', 'fevergrp_tot_day','gastro_tot_day','grs1',
                          'hla_category', 'influenza_tot_day',
                          'laryngitis_trac_tot_day', 'maternal_age', 'persist_conf_ab',
                          'race_ethnicity', 'resp_gest_inf','resp_tot_day',
                          'start_daycare_yr1', 'unknown', 'weight','yes'])

# Create matrix of p values for the multiplicative linear interaction term of logistic regression 
p_values_matrix_twointer = np.zeros([len(variables_t1d), len(variables_t1d)])

for i in range(len(variables_t1d)):
    for j in range(len(variables_t1d)):
        if i == j:
            model_ij= smf.logit(formula="t1d ~" + variables_t1d[i], data= data_y2).fit()
            p_values_matrix_twointer[i,j]= model_ij.pvalues[1]
        else:
            try:
                model_ij= smf.logit(formula="t1d ~" + variables_t1d[i] +"+"+ variables_t1d[j] + 
                                    "+" + variables_t1d[i]+":"+variables_t1d[j], data= data_y2).fit()
                p_values_matrix_twointer[i,j]= model_ij.pvalues[3]
            except :
                #print("2")
                p_values_matrix_twointer[i,j] = 1.01 
print(p_values_matrix_twointer) 

# Set labels for matrix
label_variables_t1d = np.array(['Father Diabetes Status','Male','Mother Diabetes Status',
                                'Acute sinusitis episodes','Probiotics before 1 month old ',
                                'Caesarean section','Common cold episodes','Country','Paternal age',
                                'First Dgree Relative','Fever episodes','Gastrointestinal infections ',
                                'Genetic risk score','HLA gene category', 'Influenza episodes',
                                'Laryngitis tracheitis episodes', 'Maternal age', 
                                'Number of persistent antibodies ','Race / ethnicity',
                                'Maternal gestational respiratory infections',
                                'Respiratory infection episodes','Day-care before age 1', 'No Probiotics',
                                'Weight','Sibling Diabetes Status']) 

# Plot matrix 
plt.figure(figsize = (40,30))
p = sns.heatmap(p_values_matrix_twointer, annot=True,  linewidths=.5, yticklabels= label_variables_t1d, 
                annot_kws={"size": 14},vmin=0, vmax=0.05) 
plt.tick_params(axis='both', which='major',labelsize=20, labelbottom = False, bottom=False, top = False,
                labeltop=True)
p.set_xticklabels(labels = label_variables_t1d, rotation= 90)
p.set_title("A matrix showing the p-values of the multiplicative interaction terms in the logistic 
            regression\n", fontsize=24, fontweight="bold")
#plt.savefig('MATRIX_0.05.pdf')

# Just lower LHS triangle 
trilow = np.tril(p_values_matrix_twointer)
trilow[np.triu_indices(trilow.shape[0], k = 1)] = np.nan

# Bonferroni corretion: divide the critical value by the number of repeats = 0.05/300


######
# Gradient boosting model and H statistic 
######

data_y2_vars = data_y2[['country_cd', 'maternal_age','fathers_age', 'c_section', 'resp_gest_inf',
                        'race_ethnicity', 'hla_category', 'fdr', 't1d', 'persist_conf_ab',
                        'start_daycare_yr1', 'grs1','FatherDiabetesType','MotherDiabetesType',
                        'weight','fevergrp_tot_day','common_cold_tot_day', 'laryngitis_trac_tot_day',
                        'influenza_tot_day','acute_sinusitis', 'resp_tot_day', 'gastro_tot_day' ,'Male',
                        'yes', 'before1m', 'unknown']]
  
# Create x and y for variables and the T1d varibale = predictive y/n to T1D 
X = data_y2_vars.drop('t1d', axis= 1)
y = data_y2_vars['t1d']

# Create train and test set, 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Gradient Boosting Regressor 
#clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,learning_rate=0.1, loss='huber',random_state=1)
clf = GradientBoostingRegressor(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
            
# Compute accuracy 
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# ROC curve 
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
#roc_auc

print(confusion_matrix(y_test, predictions))    
            
###
# Plot PDP 
###


print('Custom 3d plot via ``partial_dependence``')
fig = plt.figure()

target_feature = (1, 10)
pdp, axes = partial_dependence(clf, target_feature,
                                   X=X_train, grid_resolution=50)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                           cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(names[target_feature[0]])
ax.set_ylabel(names[target_feature[1]])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.suptitle('Partial dependence of t1d on maternal\n'
              'age and grs1')
plt.subplots_adjust(top=0.9)
plt.savefig('z Mat age vs  grs PDP.png',dpi = 300)

plt.show()
           
            
            
####
# Function
####
def main():
    """Convenience plot with ``partial_dependence_plots``"""
    names = X.columns
    features = [1, 10,(1, 10)]
    fig, axs = plot_partial_dependence(clf, X_train, features,
                                       feature_names=names,
                                       n_jobs=3, grid_resolution=50)
    fig.suptitle('Partial dependence of t1d on mat age and grs1\n'
                 'for the dataset')
    plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

    print('Custom 3d plot via ``partial_dependence``')
    fig = plt.figure()
    
    target_feature = (1, 10)
    pdp, axes = partial_dependence(clf, target_feature,
                                   X=X_train, grid_resolution=50)
    XX, YY = np.meshgrid(axes[0], axes[1])
    Z = pdp[0].reshape(list(map(np.size, axes))).T
    ax = Axes3D(fig)
    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                           cmap=plt.cm.BuPu, edgecolor='k')
    ax.set_xlabel(names[target_feature[0]])
    ax.set_ylabel(names[target_feature[1]])
    ax.set_zlabel('Partial dependence')
    #  pretty init view
    ax.view_init(elev=22, azim=122)
    plt.colorbar(surf)
    plt.suptitle('Partial dependence of t1d on maternal\n'
                 'age and grs1')
    plt.subplots_adjust(top=0.9)

    
    plt.show()
            
# Needed on Windows because plot_partial_dependence uses multiprocessing
if __name__ == '__main__':
    main()

            
####
# H statistic
#### 

from sklearn_gbmi import *

h_all_pairs(clf, X_train)    

data = {('FatherDiabetesType', 'Male'): _ ,...}
            
# Get the unique set of keys, sort them alphabetically
keys = sorted(set(np.ravel(list(data.keys()))))

# Create a matrix to store all the data
mat2 = np.zeros((len(keys), len(keys)))

for i, k1 in enumerate(keys):
    for j, k2 in enumerate(keys):
        # get the value from data.  if key is not in the data, use NaN
        mat2[i,j] = data.get((k1, k2), 0)
#print(mat2)


# Seaborn matrix 
plt.figure(figsize = (30,30))
p = sns.heatmap(mat2,annot = True, linewidths=.5, yticklabels=keys)
plt.tick_params(axis='both', which='major',labelsize=20)
plt.tick_params(axis='both', which='major',labelsize=15, labelbottom = False, bottom=False, top = False, labeltop=True)
p.set_xticklabels(labels = keys, rotation= 90)
p.set_title("A matrix showing the H-statisticn\n\n\n\n\n\n\n\n\n", fontsize=15, fontweight="bold")   
          
