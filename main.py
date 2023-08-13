import pandas as pd
import numpy as np
import glob
import librosa
import ntpath
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import to_categorical
import keras

Test_root = glob.glob('./cats_dogs/test')[0]
Train_root = glob.glob('./cats_dogs/train')[0]
x_path = glob.glob(Test_root + "/dogs/*")
x_path = x_path + glob.glob(Test_root + "/cats/*")
x_path = x_path + glob.glob(Train_root + "/dog/*")
x_path = x_path + glob.glob(Train_root + "/cat/*")

y = np.empty((0, 1, ))
for f in x_path:
    if 'cat' in ntpath.basename(f):
        resp = np.array([0])
        resp = resp.reshape(1, 1, )
        y = np.vstack((y, resp))
    elif 'dog' in ntpath.basename(f):
        resp = np.array([1])
        resp = resp.reshape(1, 1, )
        y = np.vstack((y, resp))

# print(x_path) -> all the audios
# print(y) -> result (0 = cat, 1 = dog) of all audios

x_train, x_test, y_train, y_test = train_test_split(x_path, y, test_size=0.25, random_state=42)


def librosa_read_wav_files(wav_files):
    # Check if wav_files is a list or not.
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [librosa.load(fi)[0] for fi in wav_files]


wav_rate = librosa.load(x_train[0])[1]
x_train = librosa_read_wav_files(x_train)
x_test = librosa_read_wav_files(x_test)

# fig, axs = plt.subplots(2, 2, figsize=(16,7))
# axs[0][0].plot(x_train[0])
# axs[0][1].plot(x_train[1])
# axs[1][0].plot(x_train[2])
# axs[1][1].plot(x_train[3])
# plt.show()


def extract_features(audio_samples, sample_rate):
    extracted_features = np.empty((0, 41,))
    if not isinstance(audio_samples, list):
        audio_samples = [audio_samples]

    for sample in audio_samples:
        zero_cross_feat = librosa.feature.zero_crossing_rate(sample).mean()
        mfccs = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)
        mfccsscaled = np.append(mfccsscaled, zero_cross_feat)
        mfccsscaled = mfccsscaled.reshape(1, 41, )
        extracted_features = np.vstack((extracted_features, mfccsscaled))
    return extracted_features


x_train_features = extract_features(x_train, wav_rate)
x_test_features = extract_features(x_test, wav_rate)

train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)

# model = models.Sequential()
# model.add(layers.Dense(100, activation='relu', input_shape=(41, )))
# model.add(layers.Dense(50, activation="relu"))
# model.add(layers.Dense(2, activation="softmax"))

# model.compile(optimizer='adam',
#               loss=losses.categorical_crossentropy,
#               metrics=['accuracy'])

# best_model_weights = './base.model'
# checkpoint = ModelCheckpoint(
#     best_model_weights,
#     monitor='val_acc',
#     verbose=1,
#     save_best_only=True,
#     mode='min',
#     save_weights_only=False,
#     period=1
# )
# callbacks = [checkpoint]


# history = model.fit(x_train_features, train_labels, validation_data=(x_test_features, test_labels), epochs=200, verbose=1, callbacks=callbacks)
# model.summary()
#
# print(history.history.keys())
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# epochs = range(1, len(acc)+1)
# plt.plot(epochs, acc, 'b', label="training accuracy")
# plt.plot(epochs, val_acc, 'r', label="validation accuracy")
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.show()
#
# model.save_weights('model_wieghts.h5')
# model.save('model_keras.h5')

model = keras.models.load_model("model_keras.h5")

nr_to_predict = 68
pred = model.predict(x_test_features[nr_to_predict].reshape(1, 41, ))
print("Cat: {} Dog: {}".format(pred[0][0], pred[0][1]))
if (y_test[nr_to_predict] == 0):
    print("This is a cat meowing")
else:
    print("This is a dog barking")

plt.plot(x_test_features[nr_to_predict])
# ipd.Audio(x_test[nr_to_predict], rate=wav_rate)