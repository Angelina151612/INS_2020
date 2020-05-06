import numpy as np
import var1
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow.keras.callbacks
from PIL import Image


class Save_Map(tensorflow.keras.callbacks.Callback):

    def __init__(self, epochs):
        super(Save_Map, self).__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch not in self.epochs:
            return
        l = 0
        weights = self.model.get_weights()
        for weight in weights:
            if len(weight.shape) == 4:
                filter = weight.shape[3]
                for i in range(0, weight.shape[2]):
                    for j in range(0, weight.shape[3]):
                        m = weight[:, :, i, j]
                        m *= 255
                        m = np.uint8(m)
                        image = Image.fromarray(m)
                        image_name = f'{l+ 1}_{i * filter + j}_{epoch}.png'
                        image.save(image_name)
                l+=1


def get_data():
    X,Y = var1.gen_data(10000, 25)
    X, Y = shuffle(X, Y)
    X,Y= np.asarray(X), np.asarray(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train.reshape(x_train.shape[0],25,25, 1)
    x_test = x_test.reshape(x_test.shape[0], 25,25, 1)
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    encoder.fit(y_test)
    y_test = encoder.transform(y_test)

    return x_train, y_train, x_test, y_test


def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(25, 25, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plots(history):
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    model = build_model()
    train_x, train_y, test_x, test_y = get_data()
    history = model.fit(train_x, train_y, batch_size=35, epochs=50, verbose=1, validation_data=(test_x, test_y), callbacks=[Save_Map([0, 24, 48])])
    plots(history)
    model.evaluate(test_x, test_y)
