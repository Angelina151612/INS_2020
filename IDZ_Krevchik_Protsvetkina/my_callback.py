import tensorflow as tf
import datetime


class show_time_callback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end = datetime.datetime.now()

        # Если эпоха не последняя,
        # то выводится время, прошедшее от запуска,
        # и время, которое осталось
        if (epoch != (self.params['epochs'] - 1)):
            print("Прошло времени с запуска", epoch_end - self.start_train)
            print("Приблизительное время до конца",
                  (epoch_end - self.epoch_start) / (epoch + 1) * (self.params['epochs'] - (epoch + 1)))
        print("Длительность эпохи", epoch_end- self.epoch_start)


    def on_train_begin(self,epoch, logs={}):
        self.start_train=datetime.datetime.now()

    def on_train_end(self,epoch, logs={}):
        full_time = datetime.datetime.now() - self.start_train
        print("Общее время ",full_time)