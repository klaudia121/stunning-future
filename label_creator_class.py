import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns


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
    #def merge with the rest of the dataframe

# end product: a dataframe with subjects IDs and their assigned labels
ftp_df= pd.read_csv('/home/ree/lemon/FTP.csv')
# labels=CreateLabels(ftp_set, 3, 'FTP_SUM')
# labels.summary()

demo_set = pd.read_csv('/home/ree/lemon/demographic.csv')
# print(demo_set['AUDIT'].values[0:5].flatten().reshape(-1,1).shape)
#print(demo_set[0].head)
# scaler = StandardScaler()
# standard_data = scaler.fit_transform(demo_set['AUDIT'].values.reshape(-1,1))


# demo=CreateLabels(demo_set, 3,'Standard_Alcoholunits_Last_28days')
# demo.summary()

lotr_set= pd.read_csv('/home/ree/lemon/LOT-R.csv')
lotr = CreateLabels(lotr_set,3, "LOT_Optimism")
lotr.summary()
