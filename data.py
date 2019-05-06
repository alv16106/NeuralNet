import numpy as np

def load_data(training_size, test_size):
  print('Loading Circles...')
  circles = np.load('data/circle.npy')
  cy = np.empty(training_size)
  cy.fill(0)
  print('Loading Houses...')
  houses = np.load('data/house.npy')
  hy = np.empty(training_size)
  hy.fill(1)
  print('Loading Smileys...')
  smileys = np.load('data/smiley.npy')
  smy = np.empty(training_size)
  smy.fill(2)
  print('Loading Squares...')
  squares = np.load('data/square.npy')
  sqy = np.empty(training_size)
  sqy.fill(3)
  print('Loading Trees...')
  trees = np.load('data/tree.npy')
  ty = np.empty(training_size)
  ty.fill(4)
  print('Loading Triangles...')
  triangles = np.load('data/triangle.npy')
  tiy = np.empty(training_size)
  tiy.fill(5)
  print('Loading sad_faces...')
  sad_faces = np.load('data/sad_face.npy', allow_pickle=True)
  sad_faces = np.array([abs(item - 255) for item in sad_faces if item.shape[0] == 784])
  sad_faces = np.concatenate(sad_faces).reshape((sad_faces.shape[0], 784))
  sfy = np.empty(training_size)
  sfy.fill(6)
  print('Loading question_marks...')
  question_marks = np.load('data/question_mark.npy', allow_pickle=True)
  question_marks = np.array([abs(item - 255) for item in question_marks if item.shape[0] == 784])
  question_marks = np.concatenate(question_marks).reshape((question_marks.shape[0], 784))
  qmy = np.empty(training_size)
  qmy.fill(7)
  print('Loading eggs...')
  eggs = np.load('data/eggs.npy', allow_pickle=True)
  eggs = np.array([abs(item - 255) for item in eggs if item.shape[0] == 784])
  eggs = np.concatenate(eggs).reshape((eggs.shape[0], 784))
  ey = np.empty(training_size)
  ey.fill(8)
  print('Loading mickeys...')
  mickeys = np.load('data/mickey.npy', allow_pickle=True)
  mickeys = np.array([abs(item - 255) for item in mickeys if item.shape[0] == 784])
  mickeys = np.concatenate(mickeys).reshape((mickeys.shape[0], 784))
  mky = np.empty(training_size)
  mky.fill(9)

  test_index = test_size + training_size
  y = np.concatenate((cy, hy, smy, ty, sqy, tiy, sfy, qmy, ey, mky))[:, np.newaxis]
  test_data = np.concatenate((
    circles[training_size:test_index],
    houses[training_size:test_index],
    smileys[training_size:test_index],
    trees[training_size:test_index],
    squares[training_size:test_index],
    triangles[training_size:test_index],
    sad_faces[training_size:test_index],
    question_marks[training_size:test_index],
    eggs[training_size:test_index],
    mickeys[training_size:test_index]
  ))
  training_data = np.concatenate((circles[:training_size], houses[:training_size], smileys[:training_size], trees[:training_size], squares[:training_size], triangles[:training_size], sad_faces[:training_size], question_marks[:training_size], eggs[:training_size], mickeys[:training_size]))
  # Zero centering
  training_data -= np.mean(training_data, axis = 0)
  test_data -= np.mean(test_data, axis = 0)
  return training_data, y, test_data
