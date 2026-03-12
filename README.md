# Healthcare Stroke Predictive Analysis

## Project Overview

Stroke is the second leading cause of death globally, accounting for approximately 11% of total deaths according to the World Health Organization. Early identification of high-risk patients is critical, timely intervention can significantly reduce mortality and long-term disability. This project builds a machine learning pipeline that predicts the likelihood of stroke in a patient based on clinical and demographic features, with a strong emphasis on maximising the detection of true stroke cases.

The project covers the full data science lifecycle. From exploratory data analysis and feature engineering through to model development, hyperparameter tuning, evaluation, and deployment as a live web application.


## Problem Statement

Given a patient’s clinical profile, can we accurately predict whether they are at risk of stroke?

This is a **binary classification problem** where:

- 1 = Patient had a stroke
- 0 = Patient did not have a stroke

The primary challenge is a severe **class imbalance**  only 4.87% of patients in the dataset experienced a stroke. This makes standard accuracy an unreliable metric and requires careful strategy around evaluation metrics, resampling techniques, and model selection.


## Dataset

|Property          |Details                             |
|------------------|------------------------------------|
|Source            |Healthcare stroke prediction dataset|
|Total Patients    |5,110                               |
|Features          |11                                  |
|Target            |stroke (binary — 0 or 1)            |
|Missing Values    |BMI (201 missing values)            |
|Class Distribution|95.13% No Stroke / 4.87% Stroke     |
|Imbalance Ratio   |19.5 : 1                            |

### Feature Descriptions

|Feature          |Type           |Description                                             |
|-----------------|---------------|--------------------------------------------------------|
|age              |Numerical      |Patient age in years                                    |
|avg_glucose_level|Numerical      |Average blood glucose level                             |
|bmi              |Numerical      |Body mass index                                         |
|hypertension     |Binary         |0 = No, 1 = Yes                                         |
|heart_disease    |Binary         |0 = No, 1 = Yes                                         |
|gender           |Categorical    |Male, Female, Other                                     |
|ever_married     |Categorical    |Yes or No                                               |
|work_type        |Categorical    |Children, Govt job, Never worked, Private, Self-employed|
|Residence_type   |Categorical    |Urban or Rural                                          |
|smoking_status   |Categorical    |Never smoked, Formerly smoked, Smokes, Unknown          |
|stroke           |Binary (Target)|0 = No Stroke, 1 = Stroke                               |


## Exploratory Data Analysis

A comprehensive EDA was conducted across all features, covering univariate, bivariate, and multivariate analysis. Below are the key findings.

### Class Imbalance

The dataset is severely imbalanced with a 19.5:1 ratio. A naive model predicting “no stroke” for every patient would achieve 95.13% accuracy without learning anything useful, confirming that **accuracy is not an appropriate evaluation metric** for this problem.

### Feature Analysis Summary

|Feature         |Stroke Rate|Key Finding                                            |
|----------------|-----------|-------------------------------------------------------|
|Age 70+         |17.84%     |Strongest individual predictor, nearly 4x the baseline|
|Heart Disease   |17.03%     |Independent cardiovascular risk pathway                |
|Hypertension    |13.25%     |Significant but partially age-driven                   |
|Glucose 200+    |12.90%     |Clear threshold effect above 150                       |
|Glucose 150-200 |11.49%     |Over 3x the baseline rate                              |
|Both Conditions |20.31%     |Compounding effect results to highest risk group                |
|Formerly Smoked |7.91%      |Moderate signal                                        |
|Gender (Male)   |5.11%      |Near baseline,  minimal predictive value               |
|Urban Residence |5.20%      |Near baseline, no independent signal                  |
|BMI (standalone)|Weak       |Heavy distribution overlap with no stroke group        |

### Key EDA Insights

**Age** is the dominant predictor by a significant margin. Stroke rate rises from 0.14% in patients under 18 to 17.84% in patients over 70, a non-linear exponential pattern that correlation analysis alone would underestimate.

**Glucose Level** shows a clear threshold effect. Below 150 the stroke rate hovers between 3.5% and 3.8%. Above 150 it jumps to 11.49% and above 200 it reaches 12.90%, a genuine independent signal beyond what age alone can explain.

**Confounding Variables** were identified throughout the analysis. Ever married (0.45 correlation with age), work type, and BMI were all found to be proxies for age rather than independent predictors. These were excluded from the final model.

**Multivariate Analysis** confirmed that stroke risk is driven by the convergence of multiple risk factors simultaneously. Patients with both hypertension and heart disease face a 20.31% stroke rate resulting to six times higher than patients with neither condition.


## Data Preprocessing

|Step                    |Decision                                                  |Rationale                                                      |
|------------------------|----------------------------------------------------------|---------------------------------------------------------------|
|Missing BMI             |Imputed using age group and gender median                 |More accurate than global median imputation                    |
|BMI Outliers            |Capped at IQR upper fence                                 |Extreme values above Q3 + 1.5×IQR replaced with median         |
|Glucose Outliers        |Preserved                                                 |High glucose values are clinically meaningful, meaning they are not noises     |
|Smoking Unknown (minors)|Relabelled to “never smoked”                              |Children below 18 with unknown status are logically non-smokers|
|Categorical Encoding    |OneHotEncoder (smoking) + OrdinalEncoder (binary features)|Appropriate per feature type                                   |
|Feature Scaling         |StandardScaler on numerical features                      |Required for Logistic Regression                               |
|Class Imbalance         |SMOTE inside pipeline + class_weight=“balanced”           |Prevents data leakage, handles minority class                  |

### Features Dropped

|Feature       |Reason                                                                                   |
|--------------|-----------------------------------------------------------------------------------------|
|BMI           |Has a weak relationship with stroke.|
|Gender        |0.40% stroke rate difference and near zero correlation (0.009)                             |
|Ever Married  |Age Driven, having a 0.45 correlation with age                                              |
|Work Type     |Age proxy, stroke rate differences entirely explained by age variation                  |
|Residence Type|0.67% stroke rate difference, making it the weakest predictor in entire dataset                       |

### Final Feature Set

```
age, avg_glucose_level, hypertension, heart_disease, smoking_status
```


## Model Development

### Evaluation Metrics

Given the severe class imbalance and the clinical context of the problem:

- **Primary Metric: Recall:** Missing a real stroke patient (False Negative) is far more dangerous than a false alarm (False Positive). In medical screening, maximising stroke detection is the priority.
- **Secondary Metric: AUC-ROC:** Measures overall discriminative ability independent of threshold.
- **Accuracy: Excluded:** Misleading on imbalanced datasets.

### Baseline Pipeline Results (Default Hyperparameters)

|Model              |Recall|AUC-ROC|
|-------------------|------|-------|
|Logistic Regression|0.80  |0.8406 |
|XGBoost            |0.58  |0.7852 |
|Gradient Boosting  |0.56  |0.7905 |
|Random Forest      |0.18  |0.7839 |
|KNN                |0.46  |0.6904 |

Top 3 models selected for hyperparameter tuning: **Logistic Regression, XGBoost, Gradient Boosting**

### Hyperparameter Tuning

GridSearchCV with 5-fold stratified cross validation was used to optimise Recall across all combinations.

|Model              |Combinations|Total Fits|
|-------------------|------------|----------|
|Logistic Regression|5×2×2 = 20  |100       |
|XGBoost            |3×3×3 = 27  |135       |
|Gradient Boosting  |3×3×3×3 = 81|405       |

### Tuned Model Results

|Model              |CV Recall |Test Recall|AUC-ROC   |Precision |F1        |
|-------------------|----------|-----------|----------|----------|----------|
|Logistic Regression|0.8194    |0.8000     |0.8423    |0.1307    |0.2247    |
|**XGBoost**        |**0.9750**|**0.9800** |**0.8221**|**0.0702**|**0.1310**|
|Gradient Boosting  |0.8292    |0.8000     |0.8403    |0.1166    |0.2036    |

### Best XGBoost Hyperparameters

```python
XGBClassifier(
    learning_rate    = 0.01,
    max_depth        = 3,
    n_estimators     = 100,
    scale_pos_weight = 19.5,
    random_state     = 42,
    eval_metric      = "logloss"
)
```

## Final Model: XGBoost

**XGBoost** was selected as the final model based on its superior Recall performance, the primary evaluation metric for this medical screening task.

### Confusion Matrix

|                    |Predicted No Stroke|Predicted Stroke|
|--------------------|-------------------|----------------|
|**Actual No Stroke**|323                |649             |
|**Actual Stroke**   |1                  |49              |

```
Strokes caught    →  49 out of 50  (98%)
Strokes missed    →  1  out of 50  (2%)
False alarms      →  649 healthy patients flagged
```

### Generalisation Check

|Metric            |Value |
|------------------|------|
|CV Recall (5-fold)|0.9750|
|Test Recall       |0.9800|
|Gap               |+0.005|

The near-zero gap between CV Recall and Test Recall confirms the model generalises well to unseen data with no signs of overfitting.

### Feature Importance (XGBoost)

|Feature                         |Importance Score|
|--------------------------------|----------------|
|Age                             |0.558           |
|Smoking Status — Never Smoked   |0.063           |
|Smoking Status — Formerly Smoked|0.055           |
|Heart Disease                   |0.052           |
|Hypertension                    |0.051           |
|Avg Glucose Level               |0.050           |
|Smoking Status — Unknown        |0.042           |
|Smoking Status — Smokes         |0.034           |

Age alone contributes 55.8% of all model decisions,  consistent with every finding in the EDA.

### Model Limitations

- **Low Precision (0.07):** The model generates a high number of false positives as a direct consequence of optimising for Recall. For every 100 patients flagged as stroke, only 7 actually have stroke. This is an intentional and clinically acceptable tradeoff for a screening tool — false alarms result in extra tests, while missed strokes can be fatal.
- **Threshold:** The default 0.5 decision threshold wrongly flags approximately 63.5% of all patients as stroke, meaning 649 out of 972 healthy patients are incorrectly predicted as high risk. In production, this threshold should be calibrated in consultation with clinical stakeholders based on acceptable false positive rates.


## Data Limitations

**Small Minority Class**
The dataset contains only 249 confirmed stroke cases out of 5,110 patients. While SMOTE was applied during training to address this imbalance, the model ultimately learned stroke patterns from a very small number of real stroke examples. A larger dataset with more confirmed stroke cases would produce a more robust and generalisable model.

**Single Source Bias**
The dataset originates from a single source and may not represent the full diversity of patient populations across different demographics, geographic regions, ethnicities, or healthcare systems. A model trained on this data may underperform when applied to patient populations that differ significantly from the training distribution.

**Missing Smoking Data**
18.8% of smoking status entries are labelled “Unknown”, the largest single category after “never smoked”. This missing information introduces uncertainty into the model’s smoking signal, particularly since smoking status was found to carry moderate predictive value during EDA.

**No Temporal Information**
The dataset contains no timestamps or longitudinal records. It is a single snapshot of patient information,  meaning the model cannot account for how a patient’s risk evolves over time as they age or as their clinical conditions progress.

**BMI Missing Values**
201 patients had missing BMI values which were imputed using age group and gender median. While this is a statistically sound approach, imputed values introduce a small degree of uncertainty into the dataset that real measured values would not.


## Data Recommendations

**Threshold Calibration**
Before any clinical deployment the decision threshold should be calibrated in direct consultation with medical professionals. The current default of 0.5 incorrectly flags 63.5% of healthy patients as high risk. A threshold analysis should be conducted to find the operating point that balances an acceptable false positive rate against the clinical requirement to minimise missed stroke cases.

**Collect More Stroke Cases**
The single biggest improvement to model performance would come from expanding the dataset with more confirmed stroke cases. The current 249 stroke cases limit the model’s ability to learn subtle and complex patterns. Even doubling the stroke cases to 500 would meaningfully improve model reliability and confidence.

**Include Additional Clinical Features**
Several clinically established stroke risk factors are absent from the current dataset, notably cholesterol levels, family history of stroke, physical activity levels, alcohol consumption, and atrial fibrillation. Including these features in future data collection would likely improve both predictive performance and clinical credibility.

**Regular Model Retraining**
Patient demographics and clinical patterns shift over time. The model should be retrained periodically as new patient data becomes available to ensure predictions remain accurate and reflective of the current patient population.

**Clinical Validation Study**
Before any real world deployment the model should be prospectively validated on a fresh patient cohort from a different data source to confirm that the 0.98 Recall generalises beyond the current dataset and holds across diverse patient populations.

**Explainability Layer**
For clinical adoption doctors need to understand why the model flagged a specific patient as high risk, not just that it did. Integrating an explainability library such as SHAP into the deployment app would show which features drove each individual prediction, building clinician trust and enabling more informed medical decision making.


## Model Deployment

The final model is deployed as an interactive web application using Streamlit.

**Running Locally**

```
# Clone the repository
git clone https://github.com/borby05/Stroke-Analysis-Prediction---Capstone-Project.git
cd C:\Users\USER\OneDrive\Desktop\StrokeDeployment

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py

```
**Note:** This app would open locally

### App Features

- Patient input form with age, glucose level, hypertension, heart disease and smoking status
- Instant stroke probability score
- HIGH RISK / LOW RISK classification with color coded result
- Patient summary table

## Technologies Used

|Category               |Tools                    |
|-----------------------|-------------------------|
|Language               |Python 3.9+              |
|Data Manipulation      |Pandas, NumPy            |
|Visualisation          |Matplotlib, Seaborn      |
|Machine Learning       |Scikit-learn, XGBoost    |
|Class Imbalance        |Imbalanced-learn (SMOTE) |
|Model Persistence      |Joblib                   |
|Web Application        |Streamlit                |
|Development Environment|Jupyter Notebook, VS Code|


## Results Summary

|Metric        |Baseline (XGBoost)|Tuned (XGBoost)|Improvement |
|--------------|------------------|---------------|------------|
|Recall        |0.58              |**0.98**       |+69%        |
|AUC-ROC       |0.7852            |**0.8221**     |+4.7%       |
|Strokes Caught|29/50             |**49/50**      |+20 patients|


## Conclusion

This project demonstrates the complete machine learning workflow applied to a real-world medical classification problem. The key technical contributions include rigorous EDA with confounding variable analysis, principled feature selection backed by statistical evidence, a properly structured pipeline that prevents data leakage, and a tuned XGBoost model that achieves 0.98 Recall catching 49 out of 50 stroke patients on the held-out test set.

The project also demonstrates an understanding of the clinical context behind the modelling decisions recognising that in medical screening the cost of a missed stroke vastly outweighs the cost of a false alarm, and making every technical choice in service of that reality.



## Authors

This project was developed as a group capstone by the following team members:

|NAME            |REG NUMBER                               |
|----------------|------------------------------------|
|Udobuzor Chinaecherem Augustus |DS/2025/TC5/083|
|⁠Goodness Chimelum Onyenakazi |   DS/2025/TC5/148   |
|Adetola Isaac Olamide | DS/2025/TC5/072 |
|⁠Julius Ayomide | DS/2025/TC5/094   |
|[Member 5 Name] |    |
|[Member 6 Name] |     |
|[Member 7 Name] |     |
|[Member 8 Name] |  |
|[Member 9 Name] |           |
|[Member 10 Name]|    |
|[Member 11 Name]|    |

**INSTITUTION: TECH CRUSH**



## Disclaimer

This project is developed for educational purposes as part of a data science capstone. It is not intended for clinical use and should not replace professional medical diagnosis or treatment. All predictions should be reviewed by a qualified healthcare professional.
