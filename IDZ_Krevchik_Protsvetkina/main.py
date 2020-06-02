import numpy as np
from sklearn.metrics import accuracy_score

from plots import print_image, plots, plot_distribution_y, plot_distribution_X, plot_heatmap
from models import build_model_1, build_model_2, save_model, load_model, ensemble_prediction
from data import load_data, prepare_data
from my_callback import show_time_callback
from keras import optimizers
from keras.utils.vis_utils import plot_model

batch_size = 50
num_epochs = 4

# Загрузка данные
train_X, train_y, test_X, test_y = load_data()

# Построение графиков распредеоления для Х и у
plot_distribution_y(train_y,test_y)
plot_distribution_X(train_X[:, :, :, 0], [0, 255])

# Вывод загруженного изображения
print_image(train_X, train_y, 0)

num_classes = np.unique(train_y).shape[0]

# Подготовка даных(нормирование и т.д.)
train_X, train_y, test_X, test_y = prepare_data(train_X, train_y, test_X, test_y, num_classes)
num_train, depth, height, width = train_X.shape

# Построение и обучение модели 1
model_1 = build_model_1(depth, height, width, num_classes)
history = model_1.fit(train_X, train_y, batch_size=batch_size, nb_epoch=num_epochs, verbose=1, validation_split=0.1,
                      callbacks=[show_time_callback()])
plots(history)

# Построение и обучение модели 2
batch_size = 64
num_epochs = 5

model_2 = build_model_2()
history = model_1.fit(train_X, train_y, batch_size=batch_size, nb_epoch=num_epochs, verbose=1, validation_split=0.1,
                      callbacks=[show_time_callback()])
plots(history)

# Сохранение модели.
# i - индекс для названия файлов.
save_model(model_1, i=11)

# Загрузка модели 1 из файлов
loaded_model_1 = load_model('model_1.json', 'model_1.h5')
loaded_model_1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Загрузка модели 2 из файлов
loaded_model_2 = load_model('model_2.json', 'model_2.h5')
optimizer = optimizers.Adam(lr=1e-3, amsgrad=True)
loaded_model_2.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print(model_1.evaluate(test_X, test_y, verbose=1))
print(loaded_model_1.evaluate(test_X, test_y, verbose=1))

print(model_2.evaluate(test_X, test_y, verbose=1))
print(loaded_model_2.evaluate(test_X, test_y, verbose=1))

# Построение confusion matrix
y_pred = model_1.predict(test_X)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(test_y, axis=1)
plot_heatmap(y_test, y_pred)

y_pred = model_2.predict(test_X)
y_pred = np.argmax(y_pred, axis=1)
plot_heatmap(y_test, y_pred)

# Ансамблирование
y_hat_eq = ensemble_prediction([model_1, model_2], [0.5, 0.5], test_X)
y_test = np.argmax(test_y, axis=1)
plot_heatmap(y_test, y_hat_eq)
acc = accuracy_score(y_test, y_hat_eq)
print("Точность ансамбля:%.2f%%" % (acc * 100))

