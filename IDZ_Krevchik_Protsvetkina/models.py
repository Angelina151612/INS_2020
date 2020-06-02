from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
import keras
import numpy as np

kernel_size = 5
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.2
drop_prob_2 = 0.4
hidden_size = 512


# Построение модели
def build_model_1(depth, height, width, num_classes):
    inp = Input(shape=(depth, height, width))

    conv_1 = Convolution2D(conv_depth_1, kernel_size, kernel_size,
                           border_mode='same', activation='relu')(inp)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
    drop_1 = Dropout(drop_prob_1)(pool_1)

    conv_2 = Convolution2D(conv_depth_2, kernel_size, kernel_size,
                           border_mode='same', activation='relu')(drop_1)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_2 = Dropout(drop_prob_1)(pool_2)

    conv_3 = Convolution2D(conv_depth_2, kernel_size, kernel_size,
                           border_mode='same', activation='relu')(drop_2)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)

    flat = Flatten()(pool_2)
    dense_1 = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(dense_1)
    out = Dense(num_classes, activation='softmax')(drop_3)

    model = Model(input=inp, output=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Схематичное изображение модели
    plot_model(model, to_file='model_11_plot.png', show_shapes=True, show_layer_names=True)
    return model


def build_model_2():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), padding='same',  activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='model_2_plot.png', show_shapes=True, show_layer_names=True)
    return model


def ensemble_prediction(models, weights, X):
    pred1 = models[0].predict(X)
    pred2 = models[1].predict(X)
    weighted_res = pred1*weights[0] + pred2*weights[1]
    return np.argmax(weighted_res, axis=1)


# Сохранение модели
# i - индекс для названия файлов
def save_model(model, i):
    model_json = model.to_json()
    with open("model_"+str(i)+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_"+str(i)+".h5")
    print("Модель " + str(i) + " сохранена на диск")


# Загрузка модели
def load_model(model_name, weights_name):
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_name)
    print("Модель и веса успешно загружены")
    return loaded_model
