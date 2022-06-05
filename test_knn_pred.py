import pickle
import Knn_model as knn


features = [28.21,0,0,0,0.0,0.0,0,1,4,2,0,0,1,6.0,0,0,1]
#features =[28.87,"Yes","No","No",6.0,0.0,"Yes","Female","75-79","Black","No","No","Fair",12.0,"No","No","No"]
filename = 'KNN_model.sav'

knn.BuidlKNN('Dataset\heart_2020_cleaned_all_numetal.csv', filename)
knn.getAccuracy()
knn.getComfusionMatrix()


# model = pickle.load(open(filename, "rb"))
# result = model.predict(features)

result = knn.makePrediction(filename, features)

print(result)
