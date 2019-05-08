import numpy as np
from sklearn.model_selection import train_test_split

def load_data(training_size, test_size):
  test_index = test_size + training_size
  
  print('Loading Circles...')
  circles = np.load('data/circle.npy')
  cy = np.empty(test_index)
  cy.fill(0)

  print('Loading Houses...')
  houses = np.load('data/house.npy')
  hy = np.empty(test_index)
  hy.fill(1)

  print('Loading Smileys...')
  smileys = np.load('data/smiley.npy')
  smy = np.empty(test_index)
  smy.fill(2)

  print('Loading Squares...')
  squares = np.load('data/square.npy')
  sqy = np.empty(test_index)
  sqy.fill(3)

  print('Loading Trees...')
  trees = np.load('data/tree.npy')
  ty = np.empty(test_index)
  ty.fill(4)

  print('Loading Triangles...')
  triangles = np.load('data/triangle.npy')
  tiy = np.empty(test_index)
  tiy.fill(5)

  print('Loading sad_faces...')
  sad_faces = np.load('data/sad_face.npy', allow_pickle=True)
  sad_faces = np.array([abs(item - 255) for item in sad_faces if item.shape[0] == 784])
  sad_faces = np.concatenate(sad_faces).reshape((sad_faces.shape[0], 784))
  sfy = np.empty(test_index)
  sfy.fill(6)

  print('Loading question_marks...')
  question_marks = np.load('data/question_mark.npy', allow_pickle=True)
  question_marks = np.array([abs(item - 255) for item in question_marks if item.shape[0] == 784])
  question_marks = np.concatenate(question_marks).reshape((question_marks.shape[0], 784))
  qmy = np.empty(test_index)
  qmy.fill(7)

  print('Loading eggs...')
  eggs = np.load('data/eggs.npy', allow_pickle=True)
  eggs = np.array([abs(item - 255) for item in eggs if item.shape[0] == 784])
  eggs = np.concatenate(eggs).reshape((eggs.shape[0], 784))
  ey = np.empty(test_index)
  ey.fill(8)

  print('Loading mickeys...')
  mickeys = np.load('data/mickey.npy', allow_pickle=True)
  mickeys = np.array([abs(item - 255) for item in mickeys if item.shape[0] == 784])
  mickeys = np.concatenate(mickeys).reshape((mickeys.shape[0], 784))
  mky = np.empty(test_index)
  mky.fill(9)

  y = np.concatenate((cy, hy, smy, ty, sqy, tiy, sfy, qmy, ey, mky))[:, np.newaxis]
  data = np.concatenate((
    circles[:test_index],
    houses[:test_index],
    smileys[:test_index],
    trees[:test_index],
    squares[:test_index],
    triangles[:test_index],
    sad_faces[:test_index],
    question_marks[:test_index],
    eggs[:test_index],
    mickeys[:test_index]
  ))

  train, test, y, y_t = train_test_split(data, y, test_size=0.1, random_state = 0)

  print(train.shape, test.shape)
  # Zero centering
  data -= np.mean(data, axis = 0)

  return train, y, test, y_t
