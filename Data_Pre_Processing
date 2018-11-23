import pandas as pd
# Importing the dataset
data = pd.read_csv("Cricket_Salary_Data.csv")
print(data)


features = data.iloc[:,:-1].values
labels = data.iloc[:,-1].values


#Handeling Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis= 0)
imputer = imputer.fit(features[:, 1:2])
features[:, 1:2] = imputer.transform(features[:, 1:2])


#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder= LabelEncoder()
features[:,0] = labelencoder.fit_transform(features[:,0])


#OneHotEncoding the laelled data
onehotencoder = OneHotEncoder(categorical_features=[0])
features= onehotencoder.fit_transform(features).toarray()


#Encoding the depedent Variables
labels= labelencoder.fit_transform(labels)


#Splitting the Dataset into training set and training set
from sklearn.model_selection import train_test_split
features_train , features_test, labels_train, labels_test = train_test_split(features, labels, test_size= 0.2, random_state= 0)


print(features_train)

print(features_test)

print(labels_train)

print(labels_test)
