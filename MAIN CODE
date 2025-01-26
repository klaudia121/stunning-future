# import libraries
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


## Checking the data from the dataset

## Include: type, head, shape, sum of the missing data, basic statistic

  #FTP
ftp_df = pd.read_csv("FTP.csv")
type(ftp_df)
ftp_df.head
ftp_df.shape
print(ftp_df.isnull().sum())

  #Stress Perception 
PSQ_df = pd.read_csv("PSQ.csv")
type(PSQ_df)
PSQ_df.shape
print(PSQ_df.describe())
print(PSQ_df.isnull().sum())

  #Social support
sozu_df = pd.read_csv("F-SozU_K-22.csv")
type(sozu_df )
sozu_df .shape
print(sozu_df .describe()) 
print(sozu_df .isnull().sum())

#Anxiety 
stai_df= pd.read_csv("STAI_G_X2.csv")
type(stai_df)
stai_df.shape
print(stai_df.describe())
print(stai_df.isnull().sum())

#Metafile
meta_df = pd.read_csv("META_File.csv")
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

    # create class Add_data instances using FTP, meta and stai files and merge with main_df 
ftp_df = Add_data(ftp_df, columns=['ID','FTP_SUM'])
ftp_df.pick_columns()

meta_df = Add_data(meta_df, columns=['ID','Gender_ 1=female_2=male','AUDIT'])
meta_df.pick_columns()

stai_df = Add_data(stai_df, columns=['ID','STAI_Trait_Anxiety'])
stai_df.pick_columns()

    # create class Add_data instances using sozu and psq files, perform feature extraction and merge with main_df 
sozu_df = Add_data(sozu_df, columns=[], dataset_name='sozu')
sozu_df.feature_extraction()

psq_df = Add_data(psq_df, columns=[], dataset_name='psq')
psq_df.feature_extraction()

    # print head of main_df
print(Add_data.main_df.head())

    # check if there are any missing values
Add_data.main_df.isnull().sum()

    # drop missing values
Add_data.main_df.dropna()

    # create heatmap with correlation between different variables excluding FTP and gender
dataplot = sns.heatmap(Add_data.main_df.drop(columns='FTP_SUM').corr(numeric_only=True), cmap="YlGnBu", annot=True)


