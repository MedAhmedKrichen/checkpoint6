import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz 
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics         


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
    


def preprocess_data(data):
    new_dataFrame=pd.DataFrame()
    new_dataFrame['Age']=data.Age.fillna(data.Age.mean())
    new_dataFrame['Sex']=pd.Series([1 if s=='male' else 0 for s in data.Sex],name='Sex')
    return new_dataFrame
train_dataset=preprocess_data(train)
test_dataset=preprocess_data(test)

train_labels=train.Survived 
classifier=tree.DecisionTreeClassifier()
classifier.fit(train_dataset,train_labels)
predicted=classifier.predict(test_dataset)

print('score:{}'.format(classifier.score(train_dataset,train_labels)))

"""
dot_data=tree.export_graphviz(classifier,out_file=None)
graph=graphviz.Source(dot_data)
graph.render()
graph
"""
def preprocess_data(data):
    new_dataFrame=pd.DataFrame()
    new_dataFrame['PassengerId']=data.Age.fillna(data.Age.mean())
    new_dataFrame['Age']=data.Age.fillna(data.Age.mean())
    return new_dataFrame
train_dataset=preprocess_data(train)
test_dataset=preprocess_data(test)
train_labels=train.Survived 
classifier=tree.DecisionTreeClassifier()
classifier.fit(train_dataset,train_labels)
predicted=classifier.predict(test_dataset)

print('score:{}'.format(classifier.score(train_dataset,train_labels)))



def preprocess_data(data):
    new_dataFrame=pd.DataFrame()
    new_dataFrame['Age']=data.Age.fillna(data.Age.mean())
    new_dataFrame['Sex']=pd.Series([1 if s=='male' else 0 for s in data.Sex],name='Sex')
    return new_dataFrame
train_dataset=preprocess_data(train)
test_dataset=preprocess_data(test)


X=train_dataset[['Age','Sex']]
y=train.Survived


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
predection=clf.predict(X_test)

print('score:{}'.format(metrics.accuracy_score(y_test,predection)))
