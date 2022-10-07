# predicting-patient-mortality

## Introduction/Background 
The need for intensive care unit (ICU) patient care expands each year (3). One of the greatest issues that ICU clinicians face is treating and monitoring acutely ill patients (3). This dataset includes the electronic health record data of 250,000 patients from the Australian and New Zealand Intensive Care Society Adult Patient Database (ANZICS-APD) and 130,000 patients from the US eICU Collaborative Research Database (eICU-CRD) (1) summarized by APACHE (Acute Physiology and Chronic Health Evaluation) data grouped by diagnosis bodies (e.g., cardiovascular, gastrointestinal, etc.). The data collected reflects the patient’s first 24 hours stay in the ICU and is labelled by patient mortality. 
## Problem Definition
Patients in the ICU must be treated based on the level of risk of mortality for the patient, but there is no universal way to test this between patients with different ailments. We will create a model that predicts mortality in ICU patients by assessing their level III APACHE score for various diagnostic bodies. With this model, clinicians will have a single metric to use to assess risk levels. 
## Methods 
- For **data cleaning**, we will have to look at the percentage of missing values and near-zero variance to eliminate features. In preliminary data processing, we observe that 74 features out of the original 186 have at least 50% of the values missing. For the remaining features, we will impute missing values to the best of our ability. We may consider binning features such as age, and for categorical features, we will have to find an encoding (perhaps one-hot). There is a 10:1 data imbalance, so we may do some undersampling to correct for that.   
- For **unsupervised**, we can use hierarchical clustering to find similarities between data points. We wish to cluster features together to see which ones might correlate together to patient mortality.  
- For **supervised learning**, we will do a standard 80/10/10 training/validation/test split to perform a classification task, predicting whether a patient expires or not. We plan on using the Python package InterpretML, which allows us to build accurate but also interpretable models, as well as numpy, pandas, and scikit-learn. 
- The models we intend to try are Random Forest (RF), Neural Network (NN), Explainable Boosting Machine (EBM), and XGBoost. Neural networks are notorious for being difficult to interpret but accurate, but EBMs are found to be comparably accurate and interpretable. 

## Potential Results and Discussion 
We hypothesize that a variety of indicators play an important role in patient survival, including age, BMI, and the ICU admit source. Furthermore, traits that indicate that the patient was healthy prior to the admission would boost their survival chances. 

We will do a detailed analysis of how our model performs in predictions. For supervised learning, we want our model to be able to predict the mortality rate of patients based on the given data. For quantitative metrics, we are looking to maximize several metrics of the model, including AUROC, F1 score, and recall.  

## Proposed Timeline
[Machine Learning Gantt Chart_Phase1.xlsx](https://github.com/cheryl-hwang/predicting-patient-mortality/files/9730315/Machine.Learning.Gantt.Chart_Phase1.xlsx)


## Contribution Table
<img width="653" alt="Screen Shot 2022-10-05 at 9 20 10 PM" src="https://user-images.githubusercontent.com/115046770/194192815-c4be91dc-0f74-4568-a799-bc9ff1cac4f1.png">

### References 
1. Raffa, Jesse & Johnson, Alistair & Celi, Leo & Pollard, Tom & Pilcher, David & Badawi, Omar. (2019). 33: THE GLOBAL OPEN SOURCE SEVERITY OF ILLNESS SCORE (GOSSIS). Critical Care Medicine. 47. 17. 10.1097/01.ccm.0000550825.30295.dd. 
2. Adhikari, N. K., Fowler, R. A., Bhagwanjee, S., & Rubenfeld, G. D. (2010). Critical care and the global burden of critical illness in adults. Lancet (London, England), 376(9749), 1339–1346. https://doi.org/10.1016/S0140-6736(10)60446-1 
3. John C. Marshall, Laura Bosco, Neill K. Adhikari, Bronwen Connolly, Janet V. Diaz, Todd Dorman, Robert A. Fowler, Geert Meyfroidt, Satoshi Nakagawa, Paolo Pelosi, Jean-Louis Vincent, Kathleen Vollman, Janice Zimmerman, What is an intensive care unit? A report of the task force of the World Federation of Societies of Intensive and Critical Care Medicine. Journal of Critical Care. Volume 37, 2017. Pages 270-276. ISSN 0883-9441. https://doi.org/10.1016/j.jcrc.2016.07.015. 
