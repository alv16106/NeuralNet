import numpy as np

def sigmoid(z):
  return 1/(1+np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z)*(1-sigmoid(z))

def vectorized_result(y, labels): 
  result = np.zeros((len(y), labels))
  for index, j in enumerate(y):  
    result[index][j] = 1.0 
  return result
