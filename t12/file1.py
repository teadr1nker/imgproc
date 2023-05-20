#!/usr/bin/python3
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
import keras.applications as apps
from sklearn.metrics import classification_report, confusion_matrix
from collections import OrderedDict
import pandas as pd
from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / possible_positives
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / predicted_positives
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


names = ['Xception', 'ResNet152V2', 'InceptionResNetV2', 'DenseNet201', 'NASNetLarge']
result = OrderedDict()
metrics = ['loss', 'accuracy', 'f1_score', 'precision', 'recall']

train_dir = 'images/numbers/train'
val_dir = 'images/numbers/val'
test_dir = 'images/numbers/test'
img_width, img_height = 100, 100
input_shape = (img_width, img_height, 3)
batch_size = 20
epochs = 4
classes = [0, 1, 3, 8]
# results = open("results.txt", "w")

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(train_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='categorical')
val_generator = datagen.flow_from_directory(val_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            shuffle=False,)

NNs = [apps.Xception(weights='imagenet',
                     include_top=False,
                     input_shape=input_shape,
                     classes=4),
       # apps.ResNet152V2(weights='imagenet',
       #                  include_top=False,
       #                  input_shape=input_shape,
       #                  classes=4),
       # apps.InceptionResNetV2(weights='imagenet',
       #                        include_top=False,
       #                        input_shape=input_shape,
       #                        classes=4),
       # apps.DenseNet201(weights='imagenet',
       #                  include_top=False,
       #                  input_shape=input_shape,
       #                  classes=4),
       # apps.NASNetLarge(weights='imagenet',
       #                  include_top=False,
       #                  input_shape=input_shape,
       #                  classes=4),
       ]
sizes = [('a', 100, 15, 20),
         # ('b', 250, 30, 20),
         # ('c', 400, 60, 20),
         # ('d', 700, 100, 20),
         ]

for size in sizes:
    letter, tr, vl, ts = size
    # results.write(f'{letter}')
    nb_train_samples = int(tr * 4)
    nb_validation_samples = int(vl * 4)
    nb_test_samples = int(ts * 4)

    sheet = []
    additionalSheet = [[''] + metrics]

    for i, nn in enumerate(NNs):
        # print(nn.summary())
        sheet.append([names[i]])
        nn.trainable = False
        model = Sequential()
        model.add(nn)
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', f1_m, precision_m, recall_m])
        model.fit(train_generator,
                  steps_per_epoch=nb_train_samples // batch_size,
                  epochs=epochs,
                  validation_data=val_generator,
                  validation_steps=nb_validation_samples // batch_size)

        test_generator = datagen.flow_from_directory(test_dir,
                                                     target_size=(img_width, img_height),
                                                     batch_size=batch_size,
                                                     class_mode='categorical',
                                                     shuffle=False,)
        report = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
        print('metrics', report)
        test_generator = datagen.flow_from_directory(test_dir,
                                                     target_size=(img_width, img_height),
                                                     batch_size=batch_size,
                                                     class_mode='categorical',
                                                     shuffle=False,)
        predictions = model.predict_generator(test_generator, nb_test_samples // batch_size).argmax(axis=1)
        # predictions = model.predict(test_generator, batch_size).argmax(axis=1)
        sheet.append(['metrics'])
        sheet.append(metrics)
        sheet.append(report)
        print(predictions)
        print(test_generator.classes)

        additionalSheet.append([names[i]] + report)

        # report = classification_report(predictions, test_generator.classes, labels = classes, output_dict=True)
        report2 = confusion_matrix(predictions, test_generator.classes)
        print(report2)

        # workaround 1
        # print(report)
        # df = pd.DataFrame(report).transpose()
        # sheet.append(['metrics'])
        # sheet.append(['classes'] + list(df.columns))
        # # print(df)
        # # print(df.values)
        # vals = df.values
        # for i, indx in enumerate(df.index):
        #     row = list(vals[i])
        #     row = [indx] + row
        #     for j in range(len(row)):
        #         row[j] = str(row[j])
        #     sheet.append(row)

        # workaround 2
        # print(report2)
        sheet.append(['confusion matrix'])
        for row in report2:
            row = list(row)
            for i in range(len(row)):
                row[i] = str(row[i])
            sheet.append(row)
        sheet.append(['#' * 8])

    result.update({letter: sheet})
    result.update({f'{letter} comp': additionalSheet})

from pyexcel_ods import save_data
save_data("results.ods", result)
