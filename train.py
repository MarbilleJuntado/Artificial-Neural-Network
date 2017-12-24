import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('Churn_Modelling.csv')

# Build matrix of features and matrix of target variable
# Exclude first two colums (0, 1)
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Dynamically encode different labels
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Create dummy variable
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Standardize scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize neural network
classifier = Sequential()

# Add input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation =
                     'relu', input_dim =  11))

# Add second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Add output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile neural network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fit the model
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predict test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)













