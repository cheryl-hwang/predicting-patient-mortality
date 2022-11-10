# predicting-patient-mortality

## Introduction/Background 
The need for intensive care unit (ICU) patient care expands each year (3). One of the greatest issues that ICU clinicians face is treating and monitoring acutely ill patients (3).  
## Problem Definition
Patients in the ICU must be treated based on the level of risk of mortality for the patient, but there is no universal way to test this between patients with different ailments. We will create a model that predicts mortality in ICU patients by assessing their level III APACHE score for various diagnostic bodies. With this model, clinicians will have a single metric to use to assess risk levels. 
## Problem Definition
This dataset includes the electronic health record data of 250,000 patients from the Australian and New Zealand Intensive Care Society Adult Patient Database (ANZICS-APD) and 130,000 patients from the US eICU Collaborative Research Database (eICU-CRD) (1) summarized by APACHE (Acute Physiology and Chronic Health Evaluation) data grouped by diagnosis bodies (e.g., cardiovascular, gastrointestinal, etc.). The data collected reflects the patient’s first 24 hours stay in the ICU and is labelled by patient mortality. 

Prior to data cleaning, there are 91,713 entries, each corresponding to a unique patient encounter, and 186 features. 
## Methods 
### Data Cleaning


We imputed missing values for continuous features with the median. For categorical variables, we imputed missing values with the label “Missing” and one-hot encoded the features. One-hot encoding led to the number of features increasing from 186 to 238. We chose not to use an ordinal encoding, as the categorical variables have no ranking or hierarchical relationship between each other.  


To account for the 10:1 data imbalance, we used a combination of under- and over-sampling with the package imblearn. First, we defined an oversampling strategy with 0.40; this means that in the original set of 0: 83798 and 1: 7915, we oversample the number of minority instances to achieve 40% of the majority class. However, the danger with oversampling the minority class is that we may increase the likelihood of overfitting, the phenomenon where we observe low training error but high testing error. To mitigate this error, we then undersampled the data with a strategy of 0.70.  This means that 70% of the minority instances should make up the number of majority instances. In the end, our Counter dict turned out like this: 

```
Counter({0: 83798, 1: 7915}) # prior to any sampling 

Counter({0: 83798, 1: 33519}) # after oversampling 

Counter({0: 47884, 1: 33519}) # after undersampling 
```

However, we observed a decrease in model performance for EBM after under- and over-sampling, and no appreciable effect on Random Forest performance. 


### Feature Selection

We threw out **too-dense features** such as “patient id”, “encounter id”, and “hospital id”; these features are merely unique identifiers and have no predictive value. The function df.nunique() allowed us to see the number of unique values per feature. We removed the feature “hospital death” from the feature set and made that the label. Additionally, we threw out the features that are an analog to the label. These features include “apache_4a_hospital_death_prob” and “apache_4a_icu_death_prob”, which are probabilities of mortality for the patient. In total, this brought the number of features down from 238 to 233.  


We attempted **PCA** with 50 components, but we observed a decrease in model performance. 
After normalizing the data, we also attempted to feature select using selecting k means based on ANOVA (analysis of variance). This was done on the training data. As a result, the ten best features were selected. These features were:
```
['gcs_eyes_apache' 'gcs_motor_apache' 'gcs_verbal_apache' 'ventilated_apache' 'd1_spo2_min' 'd1_sysbp_min' 'd1_sysbp_noninvasive_min' 'd1_temp_min' 'd1_lactate_max' 'd1_lactate_min']
```


We also attempted to **feature select** using the near zero variance method after normalizing the data. Using a 0.005 threshold, meaning that all features with less than 0.005 variance were removed, we were left with 208 features.  


Feature selection from scikit-learn's **forward feature selection** (on a smaller subset) gives that the top 15 features are:
```
['elective_surgery' 'readmission_status' 'arf_apache' 'gcs_unable_apache' 'ph_apache' 'urineoutput_apache' 'd1_albumin_min' 'd1_hemaglobin_min' 'd1_lactate_max' 'd1_arterial_ph_max' 'aids' 'cirrhosis' 'lymphoma' 
 'ethnicity_Native American' 'icu_type_SICU']
 ```
We captured the results of these methods and their corresponding models in *Table 1*. 

### Supervised Learning
We attempted 3 models: Explainable Boosted Machine (EBM), XGBoost, and Random Forest. The following table summarizes the model performances and parameters. 

## Results and Discussion 

### Model Performance Metrics

For which metrics are most relevant, we decided upon area under the ROC curve after speaking with our mentor. We included both AUC ROC score and accuracy in *Table 1*.

Below is the baseline ROC, bolded in *Table 1*.
![*Figure 1*. The ROC for the baseline model.](baseline_AUROC.png)

#### *Figure 1*. The ROC for the baseline model.


### Model Performances and Parameters
| Model         | Variance Threshold (p=0.8)| PCA | Number of features | Train/test split | Train Time | AUC Score | Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| **EBM**           | **N**  | **N** | **236**  | **Test size = 0.33** | **2 min, 30 s**  | **0.8962**  | **--**  |
| EBM           | Y | N  | 158  | Test size = 0.33  | 1 min, 16 s  | 0.8920  | --- |
| EBM           | N  | Y | 50  | Test size = 0.33   |54 s | 0.8691  | ---  |
| RF            | N  | N  | 236 | Test size = 0.33  | 10 s  | 0.8500 | ---  |
| XGBoost       | N  | N  | 236  | Test size = 0.15  | 13 min  | 0.9000  | --- |
| EBM           | N  | N | 233 | Test size = 0.33 | 2 min  | 0.8939| ---|
| RF (after NZV dimension reduction) | 0.005  | N  | 208  | Test size = 0.33  | 42 s  | 0.6098 | 0.9300  |
| RF (after feature selection)| N  | N  | 10  | Test size = 0.33  | 15 s  | 0.6058 | 0.9200  |
| XGBoost (after NZV dimension reduction)| 0.005 | N  | 208  | Test size = 0.33  | 4 min, 59 s | 0.6598 | 0.9269  |


#### *Table 1. Displays model performance based on parameters. The best-performing model is also the baseline.*

### Discussion
We hypothesize that a variety of indicators play an important role in patient survival, including age, BMI, and the ICU admit source. Furthermore, traits that indicate that the patient was healthy prior to the admission would boost their survival chances. 

We will do a detailed analysis of how our model performs in predictions. For supervised learning, we want our model to be able to predict the mortality rate of patients based on the given data. For quantitative metrics, we are looking to maximize several metrics of the model, including AUROC, F1 score, and recall.  


## Proposed Timeline
[Machine Learning Gantt Chart_Phase1.xlsx](https://github.com/cheryl-hwang/predicting-patient-mortality/files/9730315/Machine.Learning.Gantt.Chart_Phase1.xlsx)


## Contribution Table
| Team Member   | Contribution  |
| ------------- | ------------- |
| Shravani Dammu | Attempted running Hierarchical clustering, second round of KMeans (along with Positive/Negative percentage per cluster). Found top Feature_Importances from the trained Random Forest Classifier to compare with other visualizations  |
| Jennifer Deng  | Cleaned dataset, trained and evaluated the EBM model, created visualizations using interpret package, attempted hierarchical clustering, first round of DBSCAN, KMeans, PCA |
| Cheryl Hwang  | 2nd attempt at cleaning dataset by running dimension reduction after using median imputing and one-hot encoding. Used feature selection (k-means with ANOVA scoring) and near zero variance feature reduction. Tested these methods by running Random Forest Classifier. Attempted first and second rounds of XGBoost   |
| Mina Zakhary  | ---  |
| Lixin Zheng  | Attempted to run clustering models with different parameters. Performed forward feature selection on the dataset.  |



### References 
1. Raffa, Jesse & Johnson, Alistair & Celi, Leo & Pollard, Tom & Pilcher, David & Badawi, Omar. (2019). 33: THE GLOBAL OPEN SOURCE SEVERITY OF ILLNESS SCORE (GOSSIS). Critical Care Medicine. 47. 17. 10.1097/01.ccm.0000550825.30295.dd. 
2. Adhikari, N. K., Fowler, R. A., Bhagwanjee, S., & Rubenfeld, G. D. (2010). Critical care and the global burden of critical illness in adults. Lancet (London, England), 376(9749), 1339–1346. https://doi.org/10.1016/S0140-6736(10)60446-1 
3. John C. Marshall, Laura Bosco, Neill K. Adhikari, Bronwen Connolly, Janet V. Diaz, Todd Dorman, Robert A. Fowler, Geert Meyfroidt, Satoshi Nakagawa, Paolo Pelosi, Jean-Louis Vincent, Kathleen Vollman, Janice Zimmerman, What is an intensive care unit? A report of the task force of the World Federation of Societies of Intensive and Critical Care Medicine. Journal of Critical Care. Volume 37, 2017. Pages 270-276. ISSN 0883-9441. https://doi.org/10.1016/j.jcrc.2016.07.015. 
