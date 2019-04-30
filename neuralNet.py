import numpy as np
import utils

class Network(object):

  def __init__(self, shape):
    # Cuantas layers queremos
    self.num_layers = len(shape)
    self.shape = shape
    # Inicialización random de pesos
    self.weights = [np.random.randn(y, x+1)
      for x, y in zip(shape[:-1], shape[1:])]
  
  def GD(x, y, tetha, alpha, cost, derivative_cost, max_iter = 10000):
    iterations = 0
    derivative = 10000
    current_tetha = tetha
    while((iterations < max_iter) and (derivative != 0)):
      current_cost = cost(x, y, current_tetha)
      current_tetha = current_tetha - (alpha * derivative_cost(x, y, current_tetha))
      derivative = abs(cost(x, y, current_tetha) - current_cost)
      iterations += 1
  
  def Cost(self, X, y, lmbda):
    m = len(y)
    a1, z2, a2, z3, h = self.feedForward(X)
    # Funcion de costo y*log(h) - (1-y)*log(1-h)
    J = (np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))).sum() / m
    # Tomar en cuenta el learning rate + lmda/2m * suma de thetas^2
    J += (float(lmbda) / (2 * m)) * (np.sum(np.power(self.weights[0][:, 1:], 2)) + np.sum(np.power(self.weights[1][:, 1:], 2)))
    return J

  def feedForward(self, X):
    m = len(X)
    # Bias
    ones = np.ones((m,1))
    # A;adir el bias
    a1 = np.hstack((ones, X))
    z2 = a1 @ self.weights[0].T
    a2 = np.hstack((ones, utils.sigmoid(z2)))
    z3 = a2 @ self.weights[1].T
    # Sacar la hipotesis
    h = utils.sigmoid(z3)
    return a1, z2, a2, z3, h
  
  def backProp(self, X, y):
    ones = np.ones(1)
    a1, z2, a2, z3, h = self.feedForward(X)
    m = len(X)
    delta1 = np.zeros(self.weights[0].shape)  # (25, 401)
    delta2 = np.zeros(self.weights[1].shape) # (10, 26)
    output = utils.sigmoid(z3)
    for i in range(m):
      a1i = a1[i, :]
      z2i = z2[i, :]
      a2i = a2[i, :]
      hi = h[i, :]
      outputi = output[i, :]
      yi = y[i, :]

      diff = outputi - y[i][np.newaxis,:]
      z2i = np.hstack((ones, z2i))
      print((self.weights[1].T @ diff.T).shape)
      d2 = np.multiply(self.weights[1].T @ diff.T, utils.sigmoid_prime(z2i)[:,np.newaxis])
      delta1 = delta1 + d2[1:,:] @ a1i[np.newaxis,:]
      delta2 = delta2 + diff.T @ a2i[np.newaxis,:]
        
    delta1 /= m
    delta2 /= m
  
net = Network([5, 3, 3])
y = np.array([[0],[1],[2],[0],[1]])
y_d = utils.vectorized_result(y, 3)
x = np.array([[1,2,3,4,5],[2,3,5,4,8],[8,8,9,8,4],[1,5,8,6,9],[1,5,6,8,9]])
print(net.Cost(x, y_d, 1))
net.backProp(x,y_d)