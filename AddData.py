import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class AddData:
  ''' 
     class Add_data can merge multiple dataframes on ID column (main_df).
     All numerical variables are initially normalized and merged back with non-numerical columns
     Input arguments:
       - df - pandas DataFrame; dataset to be merged with main_df
       - columns (optional) - list of strings,  list of columns in df, that should be merged with main_df
       - dataset_name (optional) - string, used for naming PCA components
  '''

  main_df = pd.DataFrame()

  def __init__(self, df, columns=None, dataset_name=None):

    self.df = df
    self.columns = columns
    self.dataset_name = dataset_name

          # check if df is pd.DataFrame
    if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")

        # Select numerical features
    numerical_features = self.df.select_dtypes(include=np.number).columns

    print(f"Numerical features: {list(numerical_features)}")
    if len(numerical_features) == 0:
      raise ValueError("No numerical features found for scaling.")

        # normalise numerical features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(self.df[numerical_features])
    scaled_df = pd.DataFrame(scaled_data, index=self.df.index, columns=numerical_features)

        # Add back non-numerical columns
    excluded_columns = self.df.drop(columns=numerical_features).columns
    self.df = pd.concat([self.df[excluded_columns], scaled_df], axis=1)

  def pick_columns(self):
      '''
      checks if main_df is empty. If it is, it assigns self.df[self.columns] to main_df
      if main_df is not empty, it marges self.df[self.columns] to main_df on 'ID' column
      '''
       # check if columns is either None or a list
    if columns is not None and not isinstance(columns, list):
        raise ValueError("columns must be a list or None.")

    self.df = self.df[self.columns]

    if AddData.main_df.empty:
      AddData.main_df = self.df
    else:
      AddData.main_df = pd.merge(AddData.main_df, self.df, on='ID', how='outer')


  def feature_extraction(self):  
      '''
       performs feature extraction using principal component analysis (PCA) on df
       number of components: return number of components that explains 85% of variance
       method concats 'ID' column with dataframe with components and merges it into main_df
      '''
      # check if dataset_name is a string
    if not isinstance(dataset_name, str):
      raise ValueError("dataset_name must be a string.")

        # drop 'ID' column for PCA
    data_for_pca = self.df.drop(columns=['ID'])

        # PCA resulting with components that explain 85% of variance
    pca = PCA(n_components=0.85)
    pca_result = pca.fit_transform(data_for_pca)

        # Create a DataFrame for PCA results with appropriate column names
    pca_df = pd.DataFrame(
      pca_result, index=self.df.index, columns=[f'{self.dataset_name}_PC{i+1}' for i in range(pca_result.shape[1])])

        # Add the 'ID' column back and merge PCA results
    pca_df = pd.concat([self.df[['ID']], pca_df], axis=1)

    self.df = pca_df

        # merge with main_df
    if AddData.main_df.empty:
      AddData.main_df = self.df
    else:
      AddData.main_df = pd.merge(AddData.main_df, self.df, on='ID', how='outer')
