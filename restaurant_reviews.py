#classify movie reviews positive or negative 0"negative" 1"positive"


#reading dataset
import pandas as pd
df=pd.read_csv("Restaurant_Reviews.tsv",sep='\t')


#split in dependent andindependent arrays and vectorize
x=df['Review']
y=df['Liked']
from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer()
x=vect.fit_transform(x)


#train test splitting
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=40)


#classify and predict using suport vector classifier
from sklearn.svm import LinearSVC
clf=LinearSVC()
clf.fit(xtrain, ytrain)

predictions=clf.predict(xtest)


#check accuracy
from sklearn import metrics
metrics.accuracy_score(ytest,predictions)


#predict for any review
result=clf.predict(vect.transform(["i liked this place and their facilities"]))
