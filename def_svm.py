# filtering from merged df (whole_df) future prediction as a target variable and the the rest of the dataframe as the indpenedent variables
#indep = whole_df.drop('label', 'score', axis=1)
#y = whole_df['label']


def train_svm_classifier(indep, y, test_size=0.2, random_state=42):
    """
    Function trains an SVM classifier with hyperparameter tuning. 
    It is based on hyperpatameter tunning technique called Grid (Grid Search). It it searches through a specified subset of hyperparameters for a given model and evaluates all 
    possible combinations (incorporates cross-validation).

    Parameters:
    - indep: pd.DataFrame or np.ndarray
        Features for classification.
    - y: pd.Series or np.ndarray
        Target variable.
    - test_size: float, default 0.2
        Proportion of the dataset to include in the test split (20% of the dataset for testing, 80% for testing)
    - random_state: int, default 42
        42 is used as a defu
        Random seed for reproducibility.

    Returns:
    - model: trained SVM model.
    - indep_test: test features
    - y_test: test labels
    """
    # Spliting the dataset into training and testing sets
    data_train, data_test, y_train, y_test = train_test_split(indep, y, test_size=test_size, random_state=random_state)

    # Defining the SVM model
    svm_model = svm.SVC()

    #Grid Search - evaluates all possible combinations of the provided hyperparameters and selects the combination that results in the best performance based on a specified metric
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter - where is the margin of the errors during classification 
        'kernel': ['linear', 'rbf'],  # Kernel types
        'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf' - impact on the decision boundry? 
    }

    # Ussing GridSearchCV to find the best parameters with corss-tabulation
    grid_search = GridSearchCV(svm_model, param_grid, cv=5) 

    # Fiting the model on the training data
    grid_search.fit(indep_train, y_train)

    # Getting the best model from grid search
    best_model = grid_search.best_estimator_

    # Making predictions on the test set
    y_pred = best_model.predict(indep_test)

    # Printing evaluation metrics
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    #print("Best Cross-Validation Accuracy:", grid_search.best_score_)
    #print("Best Cross-Validation Precision:", grid_search.cv_results_['mean_test_precision'][grid_search.best_index_])
    print(classification_report(y_test, y_pred))

    return best_model, indep_test, y_test


  model, indep_test, y_test = train_svm_classifier(indep, y)

  #for plots
  pip install plotly==5.24.1   #interactive visualisation 
    
