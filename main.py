# Support Vector Machine (SVM)
# Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix




# Importing the dataset
dataset = pd.read_csv(".ar_reviews_100k.tsv", sep = "\t" )
X = dataset.iloc[:, -1].values
y = dataset.iloc[:, 1].values
# print(X)


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# random_state ??



# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Creating the Bag of Words model
cv = CountVectorizer(max_features = 1219)
# max_features ??
X = cv.fit_transform(X_train).toarray()
# y = dataset.iloc[:, 1].values


#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting SVM to the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X, y_train)




# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
confusion_matrix