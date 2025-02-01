# **The relationship between perceived remaining lifetime and psycho-social traits**


# Future prediction model 
This is the project for the advanced programming and machine learning classes. We have tried to create the classification model 
based on the questionaries implement from the LEMON/dataset (*https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/Behavioural_Data_MPILMBB_LEMON/*).

# Research question
Can a model predict wether someone has open-ended vs limited time perspective based on other information like gender, usage of the alcohol, anxiety trait, stress perception, socio-emotional support?

# Variables
The variables we considered:

**DEPENDENT**: Future Time Perspective (FTP)

**INDEPENDENT**: 

- METAfile: age, geneder, alcohol usage (AUDIT)
- anxiety trait (STAI-G-X2)
- stress perception (PSQ)
- socio-emotional support (F-SozU K-22)

The datasets used in this project are located in the `data/` folder, and can be also downloaded directly from the forementioned LEMON dataset.
Other datasets from LEMON can also be used with this code after ensuring datatypes are suitable and some modifications in MAIN_CODE.py file

# How to use the code

**1) Install all necessary libraries using 'pip install requirements.txt' (see requirements.txt)**

**2) Explore the data (check type, missing values etc.)**

**3) Ensure your data is in dataframes before using it with the AddData class**

**4) Use AddData class for preprocessing and creating the main dataframe**

**5) Exclude missing values from main dataframe**

**6) Use CreateLabels class to create classes for the numeric target variable**

**7) Use SVM MODEL for classification**

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


**CreateLabels class**
accepts 3 arguments:  the dataframe containing the target variable, the number of labales(classes) you want to obtain and the column name of the target variable. It creates an attribute *whole_df* which is a dataframe with 3 columns : 'ID' (like participant ID), 'score' - the target value, and 'label' - the target label(class). 

You can use the *summary* method on *whole_df* to get a distribution plot with the classes marked with colors and print the class counts.

Use the *merge* method to merge the *whole_df* with another dataframe. The other dataframe's name must be provided as an argument to the method. 

*usage example*: <br />
ftp_df = pd.read_csv('/home/ree/lemon/FTP.csv') <br />
demo_df = pd.read_csv('/home/ree/lemon/demographic.csv') <br />
labels=CreateLabels(ftp_df, 3, 'FTP_SUM') <br />
labels.summary() <br />
labels.merge(demo_df) <br />
print(labels.whole_df.head) <br />
