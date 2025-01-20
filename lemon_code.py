import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the datasets, replace the path according to yours
ftp_set = pd.read_csv('/home/ree/lemon/FTP.csv')
df_demo = pd.read_csv('/home/ree/lemon/demographic.csv')
# Remove missing observations
ftp_rm_na = pd.DataFrame.dropna(ftp_set)
print(ftp_rm_na.info)
# Get the scores themselves 
ftp_scores = ftp_rm_na.values[:,1]
ftp_scores = ftp_scores.reshape(-1,1)

#Initialize k-means on raw data
kmeans = KMeans(n_clusters=3)
kmeans.fit(ftp_scores)
clusters = kmeans.predict(ftp_scores)

# standardize the raw data
scaler = StandardScaler()
ftp_scores_s = ftp_scores
scaler.fit(ftp_scores_s)
ftp_scores_s = scaler.transform(ftp_scores_s)

# do clustering on the standardized data
kmeans_s = KMeans(n_clusters=3)
kmeans_s.fit(ftp_scores)
clusters_s = kmeans_s.predict(ftp_scores)

# inverse scaling after clustering to see the true values of ftp
ftp_inverse = scaler.inverse_transform(ftp_scores_s)

# prepare scatter plots to see how observations are labeled
clusters_col=clusters.reshape(-1,1)
figure, axis = plt.subplots(2, 2)
axis[0,0].scatter(ftp_scores_s, ftp_scores_s,c = clusters)
axis[1,0].scatter(ftp_inverse, ftp_inverse,c = clusters)

ftp_inverse = ftp_inverse.flatten()
clusters_col= clusters_col.flatten()
sns.displot(x=ftp_inverse, binwidth=1, hue= clusters_col)
plt.show()

# Create DataFrame with the subject number, ftp scores and labels
ids = ftp_rm_na.values[:,0]
ids = ids.flatten()
df_ftp = pd.DataFrame({
    'ID': ids,
    'score': ftp_inverse,
    'cluster': clusters_col
})
print(df_ftp.head)

# see the counts of each class, seems a litlle bit imbalanced
value_counts = df_ftp['cluster'].value_counts()
print(value_counts)


# In the demogrpahic dataframe keep only the observations of the participants included in the ftp dataframe

df_demo_fil = df_demo[df_demo['ID'].isin(ids)]
df_demo_fil = df_demo_fil.reset_index(drop=True)
print(df_demo_fil.head)
# keep only the info about gender age and audit, name the datafram df_demo
df_demo = df_demo_fil[['ID','Gender_ 1=female_2=male','Age', 'AUDIT']]

print(df_demo.head)

# import the behavioral data:
# - **(K) State-Trait Anxiety Inventory (STAI-G-X2)** - one scale
# - (K) self-blame **(CERQ)  +** refocusing on planning **(CERQ)** → dimensionality reduction (feature extraction)
# - (M) Social Support Questionnaire **(F-SozU K-22)  -**(feature extraction)  ****`Satisfaction with Social Support` + emotional support
# - **(M) Perceived Stress Questionnaire (PSQ)-**   `OverallScore`
# - **(K) Optimism Pessimism Questionnaire-Revised (LOT-R) -**`Overall Score`