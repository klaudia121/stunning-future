
# Future prediction model 
This is the project for the advanced programming and machine learning classes. We have tried to create the classification model 
based on the questionaries implement from the LEMON/dataset (*https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/Behavioural_Data_MPILMBB_LEMON/*).

# Hypothesis
The topic of the project: *"Relationship between perception of remaining time in life and other traits."*

Research question: Can model predict wether someone has open-ended vs limited time perspective based on other information like gender, 
usage of the alcohol, anxiety trait, stress perception, socio-emotional support?

# Variables
Our considered variables include:

**DEPENDENT**: Future Time Perspective (FTP)

**INDEPENDENT**: 

- METAfile: age, geneder, alcohol usage (AUDIT)
- anxiety trait (STAI-G-X2)
- stress perception (PSQ)
- socio-emotional support (F-SozU K-22)

The datasets used in this project are located in the `data/` folder, and can be also downloaded directly from the forementioned LEMON dataset.
Other datasets from LEMON can also be used with this code after ensuring datatypes are suitable and some modifications in MAIN_CODE.py file

# How to use the code

**1) Install all necessary libraries (see requirements.txt)**

**2) Data exploration process (checking type, missing values)**

**3) Ensure the data type is correct before using it with the AddData class**

**4) Use AddData class for preprocessing and creating main dataframe**

**5) Exclude missing values from main dataframe**

**6) Use CreateLabels class to create labels based on the dependent variable**

**7) SVM MODEL for classification**

The **train_svm_classifier function** is designed to train a Support Vector Machine (SVM) classifier using hyperparameter tuning via Grid Search, trains an SVM classifier, subsequently evaluates the best model (best hyperparameters, display accuracy and classification reports), and visualizes the performance.

**GridSearchCV** is a hyperparameter tuning technique that systematically searches through a predefined set of 
hyperparameters for a given model. It evaluates all possible combinations of these parameters using cross-validation to find the best-performing model. Here, we used: C (controls regularization), kernel (linear or rbf for decision    boundaries), and gamma (determines influence of training examples in RBF kernel).

However, feature selection and preprocessing (e.g., normalization, handling missing values) should be considered before training for better results. You can also compare   the resulst with other ML models (e.g., Random Forest, Logistic Regression).

Visualisation includes:

- Bar Plot: Shows performance vs. C for different kernels
- Scatter Plots: Displays true vs. predicted labels based on social support and alcohol consumption
- Confusion Matrix: Displays model classification errors

Here is an example of how to implement this function in your code:

(1) *Loading data*

indep_df = pd.read_csv('path_to_your_data.csv')  # for independent variables; a DataFrame or NumPy array

y = indep_df['Future_Time_Perspective']  # for target variable as a part of the bigger dataframe ; a Series or NumPy array

(2) *Calling the SVM training function*

best_model, test_features, test_labels = train_svm_classifier(indep_df, y)
