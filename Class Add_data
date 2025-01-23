class Add_data:

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

    if Add_data.main_df.empty:
      Add_data.main_df = self.df
    else:
      Add_data.main_df = pd.merge(Add_data.main_df, self.df, on='ID', how='outer')

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
    if Add_data.main_df.empty:
      Add_data.main_df = self.df
    else:
      Add_data.main_df = pd.merge(Add_data.main_df, self.df, on='ID', how='outer')
