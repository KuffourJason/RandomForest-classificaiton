from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

import graphviz

#loads the dataset
data = load_breast_cancer()

#Splits dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(data['data'], data['target'], random_state=0)

forest5 = RandomForestClassifier(n_estimators=5, random_state=0)
forest5.fit(x_train, y_train)

print( "The training set accuracy, with 5 trees, is " + str(forest5.score(x_train,y_train)) )
print( "The test set accuracy, with 5 trees, is " + str(forest5.score(x_test,y_test)) ) 

forest10 = RandomForestClassifier(n_estimators=10, random_state=0)
forest10.fit(x_train, y_train)

print( "The training set accuracy, with 10 trees, is " + str(forest10.score(x_train,y_train)) )
print( "The test set accuracy, with 10 trees, is " + str(forest10.score(x_test,y_test)) )