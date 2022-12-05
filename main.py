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

#print(X)
#print(Y)

# Training the Model.

# Splitting the data into training data and testing data.

X_train, X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

#print(X.shape)
#print(X_train.shape)
#print(X_test.shape)

#Feature Extraction

# transform the test data to feature vectors that can be used as input to the Logistic Regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase='True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#print(X_train)
#print(X_train_features)

model = LogisticRegression()

# training the Logistic Regression model with the training data

model.fit(X_train_features, Y_train)

# prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print("Accuracy on traininf data: ", accuracy_on_training_data)

# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


