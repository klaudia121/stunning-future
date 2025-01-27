class Add_data:
  ### 
     class Add_data can merge multiple dataframes on ID column (main_df).
     All numerical variables are initially normalized and merged back with non-numerical columns
     Input arguments:
       - df - dataset to be merged with main_df, must be a pandas DataFrame
       - columns (optional) - list of columns in df, that should be merged with main_df
       - dataset_name (optional) - name of the dataset, used for PCA output, must be a string
  ###

  main_df = pd.DataFrame()

  def __init__(self, df, columns=None, dataset_name=None):


        # check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")

    self.df = df
    self.columns = columns
    self.dataset_name = dataset_name

        # select numerical features
    numerical_features = self.df.select_dtypes(include=np.number).columns

        # print numerical features of df
    print(f"Numerical features: {list(numerical_features)}")
    if len(numerical_features) == 0:
      raise ValueError("No numerical features found for scaling.")

        # normalise
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(self.df[numerical_features])
    scaled_df = pd.DataFrame(scaled_data, index=self.df.index, columns=numerical_features)

        # add back non-numerical columns
    excluded_columns = self.df.drop(columns=numerical_features).columns
    self.df = pd.concat([self.df[excluded_columns], scaled_df], axis=1)

  def pick_columns(self):
    ### checks if main_df is empty. If it is, it assigns self.df[self.columns] to main_df
        if main_df is not empty, it marges self.df[self.columns] to main_df on 'ID' column
    ###
    self.df = self.df[self.columns]

    if Add_data.main_df.empty:
      Add_data.main_df = self.df
    else:
      Add_data.main_df = pd.merge(Add_data.main_df, self.df, on='ID', how='outer')

  def feature_extraction(self):
        
    ###
       performs feature extraction using principal component analysis (PCA) on df
       number of components: return number of components that explains 85% of variance
       method concats 'ID' column with dataframe with components and merges it into main_df
    ###

    if not isinstance(dataset_name, str):
      raise ValueError("dataset_name must be a string.")

      # drop 'ID' column for PCA
    data_for_pca = self.df.drop(columns=['ID'])

      # PCA
    pca = PCA(n_components=0.85)
    pca_result = pca.fit_transform(data_for_pca)

      # create a DataFrame for PCA results with appropriate column names
    pca_df = pd.DataFrame(
        pca_result, index=self.df.index, columns=[f'{self.dataset_name}_PC{i+1}' for i in range(pca_result.shape[1])])

      # add the 'ID' column back 
    pca_df = pd.concat([self.df[['ID']], pca_df], axis=1)

    self.df = pca_df

      # merge PCA results with the main DataFrame
    if Add_data.main_df.empty:
      Add_data.main_df = self.df
    else:
      Add_data.main_df = pd.merge(Add_data.main_df, self.df, on='ID', how='outer')
