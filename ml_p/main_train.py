import numpy as np
from mlp_keras import MLP
import tensorflow as tf
import pandas as pd

folder = './report/NO_OS/postop'
file = '/data_norm.csv'

mlp = MLP(folder + file)
log = pd.DataFrame()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True)
rlr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=0.0000001)

result = mlp.train(
    9,
    3,
    'SGD',
    0.1,
    [early_stopping_callback, rlr_callback],
    dropout=0.24,
    save_folder=None,
    oversampling=False,
    test=False
)