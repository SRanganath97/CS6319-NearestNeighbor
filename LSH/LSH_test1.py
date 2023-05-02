import lshashpy3 as lshash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statistics
import time
class KNN:
  def __init(self,k):
    self.k = k
    self.y_predict = []
    
  def find_distances(self,vector_train, vector_test):
    distance = {}
    l = []
    #print("vector_train", vector_train)
    #print("vector_test", vector_test)
    for i in range(len(vector_test)):
      for j in range(len(vector_train)):
        distance[j] = (np.linalg.norm(np.asarray(vector_test[i]) - np.asarray(vector_train[j])))
      points = self.sort(distance)
      l.append(points)
    return (l)

  def sort(self,distance):
    listofTuples = sorted(distance.items() , reverse=False, key=lambda x: x[1])
    return listofTuples[:k]



def compare(kdpoints, knn_preds):
  error = 0
  error_dist = []
  error_cent = []
  error_count = 0
  #print(len(kdpoints),len(knn_preds))
  for i in range(len(kdpoints)):
    for j in range(k):
    #   for d in range(4)
      try:
        diff = np.linalg.norm(np.asarray(kdpoints[i][j]) - np.asarray(knn_preds[i][j]))
        if(diff != 0):
          #print(i,j,diff)
          error_dist.append(diff)
          error_cent.append(j)
        error += diff
        error_count += 1
      except IndexError:
        #print(i,j)
        continue
  return error,error_cent,error_dist,error_count



df =  pd.read_csv("Iris.csv")

del df['Id']
y = df['Species']
del df['Species']
X = df

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# knn = KNN()
# predictions = (knn.find_distances(list(X_train.to_numpy()), list(X_test.to_numpy())))
# #print(predictions)
# knn_preds = []
# for i in range(len(predictions)):
#   n = []
#   for j in range(k):
#     n.append(X_train[predictions[i][j][0]])
#   #print(n)
#   knn_preds.append(n)
# print(knn_preds)

k = 1

lsh = lshash.LSHash(3,4)
for row in X_train.itertuples():
    # print(list(row)[1:])
    # break
    lsh.index(list(row)[1:])

#print(X_test)
#y1 = X_test.iloc[0,:].values.flatten().tolist()
# query a data point
lshpreds = []
LSH_time1 = time.time()
for row in X_test.itertuples():
    nn = lsh.query(list(row)[1:], num_results=k, distance_func="euclidean")
    lsh_p = []
    for ((vec,extra_data),distance) in nn:
        lsh_p.append(vec)
    #print(len(lsh_p))
    lshpreds.append(lsh_p)
#print(lshpreds)
LSH_time2 = time.time()

LSH_IRIS = LSH_time2 - LSH_time1
print(LSH_IRIS)
# unpack vector, extra data and vectorial distance
# top_n = 3
# nn = lsh.query([6.2,2.1,4.0,1.3], num_results=top_n, distance_func="euclidean")
# for ((vec,extra_data),distance) in nn:
#     print(vec, extra_data, distance)

import random

def generate_vectors(num_vectors, dimensions):
    vectors = []
    
    for i in range(num_vectors):
        vector = [round(random.uniform(0, 100), 1) for _ in range(dimensions)]
        vectors.append(vector)
    
    return vectors

tot_vec = [5000, 10000, 15000]
dimensions = [5, 10,15]


nearest = KNN()
points_train = list(zip(X_train.SepalLengthCm, X_train.SepalWidthCm, X_train.PetalLengthCm, X_train.PetalWidthCm))
points_test = list(zip(X_test.SepalLengthCm, X_test.SepalWidthCm, X_test.PetalLengthCm, X_test.PetalWidthCm))
#print(points_test)
#print(points_train)
predictions = (nearest.find_distances(points_train, points_test))
knn_preds = []
for i in range(len(predictions)):
  n = []
  for j in range(k):
    n.append(points_train[predictions[i][j][0]])
  knn_preds.append(n)
#print(knn_preds)

e,e_cent,e_dist,e_count = compare(lshpreds,knn_preds)
print(e,statistics.fmean(e_cent),statistics.fmean(e_dist))
print(e_count)

import time
k_vec = [1,3,5]
LSH_train_time = []
LSH_runtime = []
KNN_runtime = []
errors = []

for k_val in k_vec:
  k = k_val
  print("K = ",k)
  for i in range(len(tot_vec)):
    print("Data = ",tot_vec[i])
    lsh_t = []
    knn_t = []
    for j in range(len(dimensions)):
      print("Dimensions = ",dimensions[j])
      new_data = generate_vectors(tot_vec[i], dimensions[j])
      train_data = new_data[:tot_vec[i] - (tot_vec[i]//3)]
      test_data = new_data[tot_vec[i] - (tot_vec[i]//3):]
      #print(len(train_data), len(test_data))
      LSH_train_start_time = time.time()
      lsh = lshash.LSHash(8, dimensions[j])
      for row in train_data:
        #print(list(row))
        # break
        lsh.index(list(row))
      LSH_train_end_time = time.time()
      lsh_tt = LSH_train_end_time - LSH_train_start_time
      LSH_train_time.append(lsh_tt)
      LSH_run_start_time = time.time()
      lshpreds = []
      for row in test_data:
        nn = lsh.query(list(row), num_results=k, distance_func="euclidean")
        lsh_p = []
        for ((vec,extra_data),distance) in nn:
            lsh_p.append(vec)
        lshpreds.append(lsh_p)
      LSH_run_end_time = time.time()
      lsh_rt = LSH_run_end_time - LSH_run_start_time
      lsh_t.append(lsh_rt)
      print(lsh_rt,'LSH_time')
      KNN_run_start_time = time.time()
      nearest = KNN()
      predictions = (nearest.find_distances(train_data, test_data))
      # print((predictions))
      KNN_run_end_time = time.time()
      knn_rt = KNN_run_end_time - KNN_run_start_time
      knn_t.append(knn_rt)
      print(knn_rt,'KNN_time')
      new_knn_preds = []
      for c in range(len(predictions)):
        n = []
        for d in range(k):
          n.append(train_data[predictions[c][d][0]])
        new_knn_preds.append(n)
      #print(new_knn_preds)
      e,e_cent,e_dist,e_count = compare(np.array(lshpreds), np.array(new_knn_preds))
      print(e,statistics.fmean(e_cent),statistics.fmean(e_dist),e_count)
    LSH_runtime.append(lsh_t)
    KNN_runtime.append(knn_t)
