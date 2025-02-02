# import libraries
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from CreateLabels import CreateLabels
from AddData import AddData
from SVM_function import train_svm_classifier



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

AddData.main_df.isnull().sum()

AddData.main_df.dropna()

# create heatmap with correlation between different variables excluding FTP and gender
dataplot = sns.heatmap(AddData.main_df.drop(columns='FTP_SUM').corr(numeric_only=True), cmap="YlGnBu", annot=True)
plt.show()

#remove psq_PC1 and FTP_SUM
indep_df = AddData.main_df.drop(columns=['psq_PC1','FTP_SUM'])

# df for target variable
y_df = pd.read_csv('/home/ree/lemon/FTP.csv')

#create labels using class CreateLabels
labels=CreateLabels(y_df, 3, 'FTP_SUM')
labels.summary()

#remember that indep_df values are standardized, they're not the original scores
# this method concatenates indep_df with the albaled target df, craeting a new df called whole df within the class
labels.merge(indep_df)
#drop one missing observation from audit
total_df = labels.whole_df.dropna()

print(total_df.head())

model, indep_df_test, y_df_test = train_svm_classifier(total_df.drop(columns=['ID','score','label']), total_df[['label']].values.ravel())

#'Gender_ 1=female_2=male'
print(indep_df_test.head(), indep_df_test.shape)
print(y_df_test)
