import scipy.io as sio
import numpy as np
from keras.utils import np_utils


# функция для считывания датасета
def load_data():
    mat_train = sio.loadmat('train_32x32.mat')
    train_X = mat_train['X']
    train_y = mat_train['y']
    mat_test = sio.loadmat('test_32x32.mat')
    test_X = mat_test['X']
    test_y= mat_test['y']
    return train_X,train_y,test_X,test_y


# подготовка данных
def prepare_data(train_X,train_y,test_X,test_y,num_classes):
    # транспонируем из формы [32,32,3,73257] в [73257, 32, 32, 3]
    train_X = train_X.astype('float32').transpose((3, 0, 1, 2))
    test_X = test_X.astype('float32').transpose((3, 0, 1, 2))

    train_X /= np.max(train_X)  # нормируем
    test_X /= np.max(test_X)

    train_y[train_y == 10] = 0 # т.к. у 0 метка 10, заменяем ее на 0
    test_y[test_y == 10] = 0

    train_y = np_utils.to_categorical(train_y, num_classes)
    test_y = np_utils.to_categorical(test_y, num_classes )
    return train_X,train_y,test_X,test_y

