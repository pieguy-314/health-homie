#use ml to predict whether patient has cardiovascular disease based on given factors
#written by Jaeden K - team bobo remastered 
import pickle
import pandas # data processing, CSV file I/O (e.g. pandas.read_csv)
from sklearn.ensemble import RandomForestClassifier

# load saved model from health_homie_train.py
files=open('C:\\Users\\Jaeden K\\Desktop\\python stuff\\health homie\\model.pkl', 'rb')
model = pickle.load(files)
#these are features from the dataset that the program will use to predict "HeartDisease"
features = ["Age", "RestingBP","Cholesterol", "FastingBS", "MaxHR", "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope" ]

#category_features is used for non numerical data such as "Male" or "Female"
category_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


#this is the file location that the test data is read from (I took the last 21 lines out of the train_data)
test_data = pandas.read_csv("C:\\Users\\Jaeden K\\Desktop\\python stuff\\health homie\\test1.csv")

#making dummy variables for the testing data (we did this for the train data earlier)
X_test = pandas.get_dummies(test_data[features], columns=category_features)

#we are running the model in order to output the predictions
predictions = model.predict(X_test)

#an output of 1 means that patient is likely to have heart disease
output = pandas.DataFrame({'HeartDisease': predictions})
output.to_csv('Health_Homie_Predictions.csv', index=False)
print(str(output.head()))