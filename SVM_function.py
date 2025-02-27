import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


#Function that based on the GridSearch chooses the best parameters for the implemented data:

def train_svm_classifier(indep_df, y, test_size=0.3, random_state=42):
    """
    Function trains an SVM classifier with hyperparameter tuning. 
    It is based on hyperpatameter tunning technique called Grid (Grid Search). It it searches through a specified subset of hyperparameters for a given model and evaluates all 
    possible combinations (incorporates cross-validation). You can find more on scikit-learn page (https://scikit-learn.org/stable/modules/svm.html).

    Parameters:
    - indep_df: pd.DataFrame or np.ndarray
        Features for classification.
    - y: pd.Series or np.ndarray
        Target variable.
    - test_size: float, default 0.2
        Proportion of the dataset to include in the test split (usually 0.2, but it perform better on 0.3 in our small dataset).
    - random_state: int, default 42
        You can check 42 or 123.


    Returns:
    - model: trained SVM model
    - indep_df_test: test features
    - y_test: test labels


    """
    # Spliting the dataset into training and testing sets
    indep_df_train, indep_df_test, y_train, y_test = train_test_split(indep_df, y, test_size=test_size, random_state=random_state)

    # Defining the SVM model
    svm_model = svm.SVC()

    #Grid Search - evaluates all possible combinations of the provided hyperparameters and selects the combination that results in the best performance based on a specified metric
    param_grid = {
        'C': [0.1, 1, 10, 100],        # C parameter for regularization (maximizing the margin and minimizing classification errors)
        'kernel': ['linear', 'rbf'],  # Kernel types (linear or non-linear)
        'gamma': ['scale', 'auto']    # Gamma - influences how much influence individual training examples have on the decision boundary
    }

    # Using GridSearchCV to find the best parameters with corss-tabulation
    grid_search = GridSearchCV(svm_model, param_grid, cv=5) 

    # Fitting the model on the training data
    grid_search.fit(indep_df_train, y_train)

    # Getting the best model from grid search
    best_model = grid_search.best_estimator_

    # Making predictions on the test set
    y_pred = best_model.predict(indep_df_test)
    x_pred = best_model.predict(indep_df_train)

  

    # Printing evaluation metrics
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    #print(' test predicted labels:', y_pred, len(y_pred))
    print("Accuracy training:", accuracy_score(y_train, x_pred))
    #print(x_pred)
    print(classification_report(y_test, y_pred, zero_division=True)) # zero_division=True)

    #Creating classification report

    results = classification_report(y_test, y_pred, zero_division=np.nan, output_dict=True)
    print(type(results))
    results = pd.DataFrame(results)
    
    grid_results_df = pd.DataFrame(grid_search.cv_results_)
    print(grid_results_df.columns)

    grid_results_df['param_C'] = grid_results_df['param_C'].astype(float)

    #BARPLOT - plotting all of the parameters, evaluating which one was the best for the implemented data

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=grid_results_df,
        x='param_C',
        y='mean_test_score',
        hue='param_kernel',
        palette='viridis'
    )
    plt.xlabel('param_C')
    plt.ylabel('Mean Test Score')
    plt.title('Performance vs C for Different Kernels')
    plt.legend(title='Kernel')
    plt.show()
    

    # An example scatter plots with two features and predictions to check the data
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(indep_df_test['sozu_PC1'],indep_df_test['AUDIT'], c = y_test)
    ax1.set_title('true labels')
    ax1.set_xlabel('perceived social support component')
    ax1.set_ylabel('alcohol consumption')
    #plt.xlabel('perceived social support component')
    #plt.ylabel('alcohol consumption')
    
    ax2.scatter(indep_df_test['sozu_PC1'],indep_df_test['AUDIT'], c = y_pred)
    ax2.set_title('predicted labels')
    ax2.set_xlabel('perceived social support component')
    ax2.set_ylabel('alcohol consumption')

    # plt.scatter(indep_df_test['sozu_PC1'],indep_df_test['AUDIT'], c = y_test)
    # plt.xlabel('perceived social support component')
    # plt.ylabel('alcohol consumption')
    #plt.scatter(indep_df_test['AUDIT'],indep_df_test['STAI_Trait_Anxiety'], c = y_pred)
    plt.show()
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


    return best_model, indep_df_test, y_test
 
