# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
import seaborn as sns

import tensorflow as tf

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
import keras.backend as K
from keras.utils import to_categorical

# Train-Test
from sklearn.model_selection import train_test_split, StratifiedKFold

# Classification Report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def create_model(input_features, outputs, hidden_layers=[]):
    model = Sequential()
    model.add(Dense(512, input_shape = (input_features,), activation = "relu"))
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(outputs, activation = "softmax"))
    
    return model


data = pd.read_csv('./final/postop_binary_norm.csv')
X, y = data.iloc[:, :-1], data.iloc[:, -1]
print(X.head(), y.head())

y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X.values, y_cat, test_size=0.2, random_state=42)

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

scores = []
for train_idx, val_idx in skf.split(X_train, y_train.argmax(axis=1)):
    X_t, X_v = X_train[train_idx], X_train[val_idx]
    y_t, y_v = y_train[train_idx], y_train[val_idx]

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                                                
    model = create_model(input_features=9, outputs=2)
    model.summary()
    model.compile(Adam(learning_rate = 0.001), loss="binary_crossentropy", metrics='accuracy')
    # model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics='accuracy')
    model.fit(X_t, y_t, verbose=0, epochs=500, callbacks=[callback])

    y_prob = model.predict(X_v)
    y_pred_class = y_prob.argmax(axis=-1)
    y_v_pred_class = y_v.argmax(axis=-1)

    scores.append(np.mean([y_pred_class[i] == y_v_pred_class[i] for i in range(len(y_pred_class))]))

    print(classification_report(y_v_pred_class, y_pred_class))

print(scores)
print(np.mean(scores))