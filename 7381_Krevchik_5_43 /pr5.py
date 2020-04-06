from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(0)

def plots(history):
    plt.subplot(311)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.subplot(312)
    plt.plot(history.history['out_main_loss'])
    plt.plot(history.history['val_out_main_loss'])
    plt.title('model out_main_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.subplot(313)
    plt.plot(history.history['out_aux_loss'])
    plt.plot(history.history['val_out_aux_loss'])
    plt.title('model out_aux_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.show()


def gen_of_data(n):
    data = np.zeros((n,7))
    for i in range(n):
        X = np.random.normal(0, 10)
        e = np.random.normal(0, 0.3)
        data[i,:] = (np.cos(X)+e, -X+e, np.sqrt(np.fabs(X)) + e,X ** 2 + e,-np.fabs(X) + 4,X - (X ** 2) / 5 + e,np.sin(X) * X + e)
    x = data[:,0:6]
    y = data[:, 6]
    return x,y

def norm_data(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x -= mean
    x /= std
    return(x)

x,y = gen_of_data(600)
x = norm_data(x)
x_test, y_test = gen_of_data(100)
x_test = norm_data(x_test)

input = Input(shape=(6,))
encoded = Dense(64, activation='relu')(input)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(6, activation='linear')(encoded)

decoded = Dense(32, activation='relu', kernel_initializer='normal', name='d_1')(encoded)
decoded = Dense(64, activation='relu', name='d_2')(decoded)
decoded = Dense(16, activation='relu', name='d_3')(decoded)
decoded = Dense(6, name="out_aux")(decoded)

predicted = Dense(64, activation='relu', kernel_initializer='normal')(encoded)
predicted = Dense(32, activation='relu')(predicted)
#predicted = Dense(16, activation='relu')(predicted)
predicted = Dense(1, name="out_main")(predicted)

model = Model(input=input, outputs=[predicted, decoded])
model.compile(optimizer='adam', loss='mse')
history = model.fit(x, [y,x],epochs=150,batch_size=5, verbose=1,validation_data=(x_test, [y_test, x_test]))

plots(history)

encoder = Model(input, encoded)
predictor = Model(input, predicted)
dec_input = Input(shape=(6,))
decoder = model.get_layer('d_1')(dec_input)
decoder = model.get_layer('d_2')(decoder)
decoder = model.get_layer('d_3')(decoder)
decoder = model.get_layer('out_aux')(decoder)
decoder = Model(dec_input, decoder)

encoder.save('encoder.h5')
decoder.save('decoder.h5')
predictor.save('regression.h5')

pd.DataFrame(x).to_csv("x.csv")
pd.DataFrame(y).to_csv("y.csv")
pd.DataFrame(x_test).to_csv("x_test.csv")
pd.DataFrame(y_test).to_csv("y_test.csv")

encoder = Model(input, encoded)
encoded_data = encoder.predict(x)
pd.DataFrame(encoded_data).to_csv("encoded_data.csv")

decoder = Model(input, decoded)
decoded_data = decoder.predict(x_test)
pd.DataFrame(decoded_data).to_csv("decoded_data.csv")

predictor = Model(input, predicted)
predicted_data = predictor.predict(x_test)

pd.DataFrame(predicted_data).to_csv("predicted_data.csv")
