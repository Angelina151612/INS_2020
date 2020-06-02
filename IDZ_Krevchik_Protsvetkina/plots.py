import matplotlib.pyplot as plt
#import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np

# Вывод изображения после считывания
def print_image(train_X, train_y,i):
    plt.imshow(train_X[:,:,:,i])
    plt.show()
    print(train_y[i])


# Графики потерь и точности
def plots(history):
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
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


# Гистограмма распределения меток
def plot_distribution_y(train_y,test_y):
    plt.hist(x=train_y, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('y')
    plt.ylabel('Frequency')
    plt.title('Distribution of train_y')
    plt.show()

    plt.hist(x=test_y, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('y')
    plt.ylabel('Frequency')
    plt.title('Distribution of test_y')
    plt.show()


# Гистограмма цветов изображения
def plot_distribution_X(train_X,range):
    # Закрашенная гистограмма
    img = train_X
    plt.hist(img.ravel(), 256, [0, 256]);
    plt.show()

    # Гистограмма с RGB линиями
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim(range)
    plt.show()
    
    
def plot_heatmap(y_train, y_pred):
    matrix = confusion_matrix(y_train, y_pred)
    df_cm = pd.DataFrame(matrix, columns=np.unique(y_train), index=range(10))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d', ax=ax)
    plt.title('Confusion Matrix for testing dataset')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
