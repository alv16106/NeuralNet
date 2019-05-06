import numpy as np
import utils
import data

class Network(object):

  def __init__(self, shape):
    # Cuantas layers queremos
    self.num_layers = len(shape)
    self.shape = shape
    # Inicialización random de pesos
    self.weights = [np.random.randn(y, x+1)
      for x, y in zip(shape[:-1], shape[1:])]

  def GD(self, x, y, alpha, max_iter = 10000, treshold = 0.001):
    iterations = 0
    current_cost = 100000
    # Llegamos a las iteraciones maximas o nuestro costo es mas peque;o que el threshold
    while((iterations < max_iter) and (current_cost > treshold)):
      current_cost, deltas = self.backProp(x, y, alpha)
      # Actualizamos pesos
      self.weights[0] = self.weights[0] - (alpha * deltas[0])
      self.weights[1] = self.weights[1] - (alpha * deltas[1])
      s = np.concatenate((np.ravel(deltas[0]), np.ravel(deltas[1])))
      current_cost = np.linalg.norm(s)
      iterations += 1
      print('Iteration' + str(iterations))
    return self.weights
  
  def Cost(self, h, y, lmbda):
    m = len(y)
    # Funcion de costo y*log(h) - (1-y)*log(1-h)
    J = (np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))).sum() / m
    # Tomar en cuenta el learning rate + lmda/2m * suma de thetas^2
    J += (float(lmbda) / (2 * m)) * (np.sum(np.power(self.weights[0][:, 1:], 2)) + np.sum(np.power(self.weights[1][:, 1:], 2)))
    return J

  def feedForward(self, X):
    m = X.shape[0]
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

  def predict(self, X):
    h = self.feedForward(X)[4]
    return h
  
  def backProp(self, X, y, lmbda):
    ones = np.ones(1)
    a1, z2, a2, z3, h = self.feedForward(X)
    J = self.Cost(h,y,lmbda)
    m = X.shape[0]
    delta1 = np.zeros(self.weights[0].shape)  # (3, 6)
    delta2 = np.zeros(self.weights[1].shape) # (3, 4)
    ones = np.ones((m,1))
    diff = h - y
    z2 = np.hstack((ones, z2)) # (5,4)
    d2 = np.multiply(np.dot(self.weights[1].T, diff.T).T, utils.sigmoid_prime(z2))  # (5000, 26)
    delta1 += np.dot((d2[:, 1:]).T, a1)
    delta2 += np.dot(diff.T, a2)

    delta1 = delta1 / m
    delta2 = delta2 / m
    
    # Añadir la regularización, pero no al bias
    delta1[:, 1:] = delta1[:, 1:] + (self.weights[0][:, 1:] * lmbda) / m
    delta2[:, 1:] = delta2[:, 1:] + (self.weights[1][:, 1:] * lmbda) / m

    return J, [delta1, delta2]

net = Network([784, 25, 10])
""" y = np.array([[0],[1],[2],[0],[1]])
print(y.shape)
y_d = utils.vectorized_result(y, 3)
x = np.array([[1,2,3,4,5],[2,3,5,4,8],[8,8,9,8,4],[1,5,8,6,9],[1,5,6,8,9]])
# print(net.Cost(x, y_d, 1)) """
x, y, test = data.load_data(1000, 10)
y_d = utils.vectorized_result(y, 10)
net.GD(x, y_d, 1, 300, 0.01)
print(net.predict(test)[0])