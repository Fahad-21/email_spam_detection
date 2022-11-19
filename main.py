import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data PreProcessing
# loading the data from csv file to pandas Dataframe.
raw_mail_data = pd.read_csv('dataset.csv')

#print(raw_mail_data)

# replacing the null values with a nulll string.

mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# printing the first 5 rows of the dataframe

#print(mail_data.head())

# checking number of rows and columns in dataframe
#print(mail_data.shape)

#labeling spam mail as 0; and ham mail as 1;

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

#seperating the data as texts and lebel

X = mail_data['Message']
Y = mail_data['Category']

print(X)
print(Y)