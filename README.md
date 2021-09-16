# Type 1 Diabetes Research Project: Gradient boosting classifier
Final year BSc research project entitled 'Identifying interactions between risk factors for Type 1 Diabetes'

### Background  
Type 1 diabetes (T1D) is an autoimmune disease that causes destruction of insulin-producing cells. It is recognised to result from a complex interplay between environmental and genetic risk factors. Accurately predicting T1D is important to avoid severe complications at diagnosis and enable precision medicine. Current single-variable models are limited in their predictive power (Fig.1). Hence, exploring interactions between risk factors could extract more understanding of T1D risk and improve prediction models.

### Method 
Use two statistical tests on two classifiers to gain complimentary information about the presence and strength of interactions between risk factors. 

1: Linear classifier (logistic regression) with a multiplicative linear interaction term. Matrix of interaction term p-values.
2: Non-linear classifier (gradient boosting). Matrix of H-statistic values. (Friedman, J. H., & Popescu, B.(2008). Predictive Learning Via Rule Ensembles. The Annals of Applied Statistics, Vol. 2, No. 3, 916–954, DOI: 10.1214/07-AOAS148.) 
