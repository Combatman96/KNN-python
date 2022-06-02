import pickle
import Knn_model as knn


features = [[24.21,0,0,0,0.0,0.0,0,1,4,2,0,0,1,6.0,0,0,1]]
filename = 'KNN_model.sav'

knn.BuidlKNN('Dataset\heart_2020_cleaned.csv', filename)
knn.getAccuracy()
knn.getComfusionMatrix()


model = pickle.load(open(filename, "rb"))
result = model.predict(features)

print(result)
