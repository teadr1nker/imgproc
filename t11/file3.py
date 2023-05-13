#!/usr/bin/python3
import sys
try:
    trainSamples = int(sys.argv[1])
except:
    trainSamples = 800
# original script CatsVsDogs
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense

# Каталог с данными для обучения
train_dir = 'images/numbers/train'
# Каталог с данными для проверки
val_dir = 'images/numbers/val'
# Размеры изображения
img_width, img_height = 100, 100
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Количество эпох
epochs = 16
# Размер мини-выборки
batch_size = 32
# Количество изображений для обучения
nb_train_samples = int(trainSamples * 4)
# Количество изображений для проверки
nb_validation_samples = 640
# Количество изображений для тестирования
nb_test_samples = 80

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1. / 255)

# Генератор данных для обучения на основе изображений из каталога
train_generator = datagen.flow_from_directory(train_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='binary')
# Генератор данных для проверки на основе изображений из каталога
val_generator = datagen.flow_from_directory(val_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode='binary')

# Обучаем модель с использованием генераторов
# train_generator - генератор данных для обучения
# validation_data - генератор данных для проверки
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=nb_validation_samples // batch_size)
# Оцениваем качество работы сети с помощью генератора
# scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
# print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

print(model.summary())

# Каталог с данными для тестирования
test_dir = 'images/numbers/test'

for i in range(5):
    # Генератор данных для тестирования на основе изображений из каталога
    test_generator = datagen.flow_from_directory(test_dir,
                                                 target_size=(img_width, img_height),
                                                 batch_size=batch_size,
                                                 class_mode='binary')
    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
    print(f'{i} accuracy {scores[1]}')
    test_dir = f'images/numbers/test{i+2}'
    nb_test_samples = 160
