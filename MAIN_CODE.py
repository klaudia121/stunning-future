# import libraries
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
# import CreateLabels
# import AddData



class AddData:

  main_df = pd.DataFrame()

  def __init__(self, df, columns=None, dataset_name=None):

    if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")

    self.df = df
    self.columns = columns if columns is not None else []
    self.dataset_name = dataset_name

        # Select numerical features
    numerical_features = self.df.select_dtypes(include=np.number).columns

    print(f"Numerical features: {list(numerical_features)}")
    if len(numerical_features) == 0:
      raise ValueError("No numerical features found for scaling.")

        # normalise
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(self.df[numerical_features])
    scaled_df = pd.DataFrame(scaled_data, index=self.df.index, columns=numerical_features)

        # Add back non-numerical columns
    excluded_columns = self.df.drop(columns=numerical_features).columns
    self.df = pd.concat([self.df[excluded_columns], scaled_df], axis=1)

  def pick_columns(self):
    self.df = self.df[self.columns]

    if AddData.main_df.empty:
      AddData.main_df = self.df
    else:
      AddData.main_df = pd.merge(AddData.main_df, self.df, on='ID', how='outer')

  def feature_extraction(self):

    data_for_pca = self.df.drop(columns=['ID'], errors='ignore')

    # PCA
    pca = PCA(n_components=0.85)
    pca_result = pca.fit_transform(data_for_pca)

    # Create a DataFrame for PCA results with appropriate column names
    pca_df = pd.DataFrame(
        pca_result, index=self.df.index, columns=[f'{self.dataset_name}_PC{i+1}' for i in range(pca_result.shape[1])])

    # Add the 'ID' column back and merge PCA results
    if 'ID' in self.df.columns:
      pca_df = pd.concat([self.df[['ID']], pca_df], axis=1)

    self.df = pca_df

    # merge with main_df
    if AddData.main_df.empty:
      AddData.main_df = self.df
    else:
      AddData.main_df = pd.merge(AddData.main_df, self.df, on='ID', how='outer')



# filtering from merged df (whole_df) future prediction as a target variable and the the rest of the dataframe as the indpenedent variables
#indep = whole_df.drop('label', 'score', axis=1)
#y_df = whole_df['label']


def train_svm_classifier(indep_df, y_df, test_size=0.2, random_state=42):
    """
    Function trains an SVM classifier with hyperparameter tuning. 
    It is based on hyperpatameter tunning technique called Grid (Grid Search). It it searches through a specified subset of hyperparameters for a given model and evaluates all 
    possible combinations (incorporates cross-validation).

    Parameters:
    - indep: pd.DataFrame or np.ndarray
        Features for classification.
    - y_df: pd.Series or np.ndarray
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
    indep_df_train, indep_df_test, y_df_train, y_df_test = train_test_split(indep_df, y_df, test_size=test_size, random_state=random_state)

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
    grid_search.fit(indep_df_train, y_df_train)

    # Getting the best model from grid search
    best_model = grid_search.best_estimator_

    # Making predictions on the test set
    y_pred = best_model.predict(indep_df_test)

    # Printing evaluation metrics
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_df_test, y_pred))
    #print("Best Cross-Validation Accuracy:", grid_search.best_score_)
    #print("Best Cross-Validation Precision:", grid_search.cv_results_['mean_test_precision'][grid_search.best_index_])
    print(classification_report(y_df_test, y_pred))

    return best_model, indep_df_test, y_df_test


  #model, indep_test, y_test = train_svm_classifier(indep, y_df)

  #for plots
  #pip install plotly==5.24.1   #interactive visualisation 
    


class CreateLabels:
    def __init__(self, data_df, n_labels, target_var):
        # data_df must be pandas DataFrame
        self.data_df = data_df[['ID', target_var]]
        #print(self.data_df.values[:,1])
        self.n_labels = n_labels
        self.target_var = target_var
        
        # Remove missing observations
        self.clean_df = pd.DataFrame.dropna(self.data_df)
        self.values = self.clean_df[target_var].values.reshape(-1,1)
        print(self.values.shape)

        # standardize the raw data
        scaler = StandardScaler()
        self.standard_data = scaler.fit_transform(self.values)

        #kmeans for clustering
        kmeans= KMeans(n_clusters= n_labels, random_state=123)
        self.clusters = kmeans.fit_predict(self.standard_data)

        # Create DataFrame with the subject IDs and labels
        self.ids = self.clean_df.values[:,0].flatten()
        self.scores = self.clean_df[target_var].values.flatten()
        
        self.labeled_df = pd.DataFrame({
            'ID': self.ids,
            'score': self.scores,
            'label': self.clusters
        })
    def summary(self):
        print(self.labeled_df.head,"class counts:",self.labeled_df['label'].value_counts())
        sns.displot(x=self.scores, binwidth=1, hue= self.labeled_df['label'])
        plt.show()
    # merge with another dataset for training
    def merge(self,big_data_df):
        self.big_data_df = big_data_df
        #create a whole df
        # keep only participants present in target df
        self.big_matched_df = self.big_data_df[big_data_df['ID'].isin(self.ids)]
        self.big_matched_df = self.big_matched_df.reset_index(drop=True).drop(columns='ID')
        self.whole_df = pd.concat([self.labeled_df, self.big_matched_df], axis=1)

# end product: a dataframe with subjects IDs and their assigned labels
# ftp_df = pd.read_csv('/home/ree/lemon/FTP.csv')
# demo_df = pd.read_csv('/home/ree/lemon/demographic.csv')
# labels=CreateLabels(ftp_df, 3, 'FTP_SUM')
# labels.summary()
# labels.merge(demo_df)
# print(labels.whole_df.head)

#demo_set = pd.read_csv('/home/ree/lemon/demographic.csv')
# print(demo_set['AUDIT'].values[0:5].flatten().reshape(-1,1).shape)
#print(demo_set[0].head)
# scaler = StandardScaler()
# standard_data = scaler.fit_transform(demo_set['AUDIT'].values.reshape(-1,1))


# demo=CreateLabels(demo_set, 3,'Standard_Alcoholunits_Last_28days')
# demo.summary()


## Checking the data from the dataset

## Include: type, head, shape, sum of the missing data, basic statistic

  #FTP
ftp_df = pd.read_csv("/home/ree/lemon/FTP.csv")
print(type(ftp_df))
print(ftp_df.head())
print(ftp_df.shape)
print(ftp_df.describe())
print(ftp_df.isnull().sum())

  #Stress Perception 
psq_df = pd.read_csv("/home/ree/lemon/PSQ.csv")
print(type(psq_df))
print(psq_df.head())
print(psq_df.shape)
print(psq_df.describe())
print(psq_df.isnull().sum())

  #Social support
sozu_df = pd.read_csv("/home/ree/lemon/F-SozU_K-22.csv")
type(sozu_df )
sozu_df .shape
print(sozu_df .describe()) 
print(sozu_df .isnull().sum())

#Anxiety 
stai_df= pd.read_csv("/home/ree/lemon/STAI_G_X2.csv")
type(stai_df)
stai_df.shape
print(stai_df.describe())
print(stai_df.isnull().sum())

#Metafile
meta_df = pd.read_csv("/home/ree/lemon/META_File.csv")
type(meta_df)
meta_df.shape
print(meta_df.describe())
print(meta_df.isnull().sum())


    # convert psq_df values from str to numeric
psq_df['PSQ_Tension'] = pd.to_numeric(psq_df['PSQ_Tension'], errors='coerce')
psq_df['PSQ_Joy'] = pd.to_numeric(psq_df['PSQ_Joy'], errors='coerce')
psq_df['PSQ_Demands'] = pd.to_numeric(psq_df['PSQ_Demands'], errors='coerce')
psq_df['PSQ_Worries'] = pd.to_numeric(psq_df['PSQ_Worries'], errors='coerce')
psq_df['PSQ_OverallScore'] = pd.to_numeric(psq_df['PSQ_OverallScore'], errors='coerce')


    # convert values of gender in meta_df from numeric to str
meta_df['Gender_ 1=female_2=male'] = meta_df['Gender_ 1=female_2=male'].astype(str)

    # psq fill missing values with interpolation
psq_df = psq_df.interpolate()

    # create class AddData instances using FTP, meta and stai files and merge with main_df 
ftp_df = AddData(ftp_df, columns=['ID','FTP_SUM'])
ftp_df.pick_columns()

meta_df = AddData(meta_df, columns=['ID','Gender_ 1=female_2=male','AUDIT'])
meta_df.pick_columns()

stai_df = AddData(stai_df, columns=['ID','STAI_Trait_Anxiety'])
stai_df.pick_columns()

    # create class AddData instances using sozu and psq files, perform feature extraction and merge with main_df 
sozu_df = AddData(sozu_df, columns=[], dataset_name='sozu')
sozu_df.feature_extraction()

psq_df = AddData(psq_df, columns=[], dataset_name='psq')
psq_df.feature_extraction()

    # print head of main_df
print(AddData.main_df.head())

#remove psq_PC1
total_df = AddData.main_df.drop(columns='psq_PC1')

indep_df = total_df.drop(columns="FTP_SUM")
y_df = pd.read_csv('/home/ree/lemon/FTP.csv')

#create labels using class CreateLabels
labels=CreateLabels(y_df, 3, 'FTP_SUM')
labels.summary()

#remember that indep_df values are standardized, they're not the original scores
labels.merge(indep_df)
#drop one missing observation from audit
new_total_df = labels.whole_df.dropna()

print(new_total_df.head())

#print(labels.labeled_df.shape, indep_df.shape, new_total_df.shape)

#breakpoint for debuggin and data veiewing
#print(new_total_df[0])

#print(y_df.head())
#use svm function

model, indep_df_test, y_df_test = train_svm_classifier(new_total_df.drop(columns=['ID','score','label','Gender_ 1=female_2=male']), new_total_df[['label']].values.ravel())

print(indep_df_test)

# clf = svm.SVC()
# X=new_total_df.drop(columns=['ID','score','label','Gender_ 1=female_2=male'])
# y=new_total_df[['label']].values.ravel()

# clf.fit(X,y)
# predictions = clf.predict(new_total_df.drop(columns=['ID','score','label','Gender_ 1=female_2=male']))
# print(predictions)


# print("Accuracy of SVM:", clf.score(X, y))

#print(new_total_df.head(),indep_df.head(),y_df.head())


    # check if there are any missing values
#print(AddData.main_df.isnull().sum())


   # drop missing values
#AddData.main_df.dropna()

    # create heatmap with correlation between different variables excluding FTP and gender
#dataplot = sns.heatmap(AddData.main_df.drop(columns='FTP_SUM').corr(numeric_only=True), cmap="YlGnBu", annot=True)
#plt.show()

