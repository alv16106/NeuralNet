import glob
import numpy as np
from PIL import Image
import pickle

filelist = glob.glob('data/Mickey_new/*.bmp')
x = np.array([np.ravel(Image.open(fname).convert('L')) for fname in filelist])
count = 0
x = np.array([item for item in x if item.shape[0] == 784])
x = np.concatenate(x).reshape((x.shape[0], 784))
pickle.dump(x, open('data/mickey.npy', "wb"))