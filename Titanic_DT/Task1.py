import sys
import pandas
import numpy
import seaborn
import random
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, cross_val_score
from sklearn import tree

# reading dataset
train_df = pandas.read_csv('data/train.csv')
test_df = pandas.read_csv('data/test.csv')

# joining train and test data
combine = [train_df, test_df]

pandas.options.display.max_columns = None
numpy.set_printoptions(threshold=numpy.inf)

# data prerpocessing
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Filling Age (KNN)
imputer = KNNImputer(n_neighbors=10)
filled = imputer.fit_transform(train_df[['Pclass', 'Sex', 'Age']])
filled_df = pandas.DataFrame(filled, columns = ['Pclass', 'Sex', 'Age'])
train_df['Age'] = filled_df['Age']

filled = imputer.fit_transform(test_df[['Pclass', 'Sex', 'Age']])
filled_df = pandas.DataFrame(filled, columns = ['Pclass', 'Sex', 'Age'])
test_df['Age'] = filled_df['Age']

combine = [train_df, test_df]

for dataset in combine:
    dataset['Age'] = dataset['Age'].astype(int)

# Completing Embarked (Highest freq)
freq_emb = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_emb)

# Convert Embarked to integers
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
combine = [train_df, test_df]

# Correct
#droping columns that not required
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]


# Create

# creating Age band

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']


# Creating Family size from SibSp and Parch column
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Using FmailySize to find Is alone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]

# Create Fare Band

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass



X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)
feature_names = X_train.columns.values
class_names = ['N', 'Y']
# print(feature_names)
# print(class_names)
# sys.exit()



# LR

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print('LR', acc_log)



# SVM

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print('SVM', acc_svc)

# L-SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print('L-SVC', acc_linear_svc)

# KNN

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print('KNN', acc_knn)

# NB

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print('NB', acc_gaussian)

# P - Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print('P', acc_perceptron)

# SGD

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print('SGD', acc_sgd)

# DT

decision_tree = DecisionTreeClassifier(criterion='gini')#,max_depth = 3)
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print('DT', acc_decision_tree)

# RF

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('RF', acc_random_forest)

# Comparison

models = pandas.DataFrame({
    'Model': ['LR', 'SVM', 'L-SVC', 'KNN', 'NB', 'P', 'SGD', 'DT', 'RF'],
    'Score': [acc_log, acc_svc, acc_linear_svc, acc_knn, acc_gaussian, acc_perceptron, acc_sgd, acc_decision_tree, acc_random_forest]})
print(models.sort_values(by='Score', ascending=False))

fig = plt.figure()
tree.plot_tree(decision_tree, feature_names = feature_names, class_names=class_names, filled = True)
#plt.show()

kf = KFold(n_splits=5)
decision_tree=DecisionTreeClassifier(criterion='gini')#, max_depth=3)
scores = cross_val_score(decision_tree, X_train, Y_train, cv=kf)
print(scores)
avg_score = numpy.mean(scores)
print('DT Average', avg_score)

kf = KFold(n_splits=5)
random_forest = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(random_forest, X_train, Y_train, cv=kf)
print(scores)
avg_score = numpy.mean(scores)
print('RF Average', avg_score)
