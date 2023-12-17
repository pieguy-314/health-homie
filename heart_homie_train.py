#use ml to train model to predict whether or not patient has cardiocascular disease
#written by Jaeden K - team bobo remastered 

import pandas # data processing, CSV file I/O (e.g. pandas.read_csv)
from sklearn.ensemble import RandomForestClassifier

#read date for training ml model from file location below 
#downloaded from https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data (open source)
train_data = pandas.read_csv("C:\\Users\\Jaeden K\\Desktop\\python stuff\\heart_homie\\heart.csv")


#set the variable for the column in the dataset that the program is trying to predict
y = train_data["HeartDisease"]

#these are features from the dataset that the program will use to predict "HeartDisease"
features = ["Age", "RestingBP","Cholesterol", "FastingBS", "MaxHR", "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope" ]

#category_features is used for non numerical data such as "Male" or "Female"
category_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

#get_dummies is used to create dummy variables for non numerical data so that it can be used as a "feature"
X = pandas.get_dummies(train_data[features], columns=category_features) 

#These two lines are for creating the model and training it
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# loading library
import pickle
# create an iterator object with write permission - model.pkl
files=open('C:\\Users\\Jaeden K\\Desktop\\python stuff\\heart_homie\\model.pkl', 'wb')  
pickle.dump(model, files)

