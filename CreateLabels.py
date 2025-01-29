import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

# usage example:
# ftp_df = pd.read_csv('/home/ree/lemon/FTP.csv')
# demo_df = pd.read_csv('/home/ree/lemon/demographic.csv')
# labels=CreateLabels(ftp_df, 3, 'FTP_SUM')
# labels.summary()
# labels.merge(demo_df)
# print(labels.whole_df.head)
