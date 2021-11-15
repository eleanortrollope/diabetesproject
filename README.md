# Type 1 Diabetes Machine Learning Research Project
Final year BSc research project entitled 'Identifying interactions between risk factors for Type 1 Diabetes'

### Background  
Type 1 diabetes (T1D) is an autoimmune disease that causes destruction of insulin-producing cells. It is recognised to result from a complex interplay between environmental and genetic risk factors. Accurately predicting T1D is important to avoid severe complications at diagnosis and enable precision medicine. Current single-variable models are limited in their predictive power (Fig.1). Hence, exploring interactions between risk factors could extract more understanding of T1D risk and improve prediction models.

### Method 
Use two statistical tests on two classifiers to gain complimentary information about the presence and strength of interactions between risk factors. 

1. Linear classifier (logistic regression) with a multiplicative linear interaction term. Matrix of interaction term p-values.
2. Non-linear classifier (gradient boosting). Matrix of H-statistic values. (Friedman, J. H., & Popescu, B.(2008). Predictive Learning Via Rule Ensembles. The Annals of Applied Statistics, Vol. 2, No. 3, 916–954, DOI: 10.1214/07-AOAS148.) (Gradient Boosting Regressor, n_estimators=100, random_state=0)

### Results 
#### Key Findings
1. No significant linear interactions found 
2. Non-linear interactions identified

![image](https://user-images.githubusercontent.com/59938778/141858960-93d2613a-99ee-4936-97f7-62c69910f4a7.png)

1. Linear interactions: Matrix of p-values of the logistic regression interaction term, characterising a linear interaction.
-  No p-values were significant when applying a significance level of 0.05 with Bonferroni 
correction. Values enclosed by blue rectangles were significant before correction and so were explored by PDPs. The leading diagonal is the 
p-value of a single variable logistic regression, showing only four variables were significant on their own.

2. Non-linear interactions: A matrix of H-statistic values for the non-linear model
- Higher H-statistic values indicate stronger interaction. Many interactions were identified. Due to the nature of the H-statistic, investigating 
variance of the output of the partial dependence, this could indicate any type of interaction. 

A novel result is that there is interaction between variables that are not significant on their own, namely weight and 
maternal age as well as weight and siblings with diabetes. See figures below. 


![image](https://user-images.githubusercontent.com/59938778/141858695-a05b93e6-b593-43c5-b8c6-91d804f62777.png)

![image](https://user-images.githubusercontent.com/59938778/141858748-4f94cd37-58b4-4ed7-b620-b7e3f8ce4e72.png)


