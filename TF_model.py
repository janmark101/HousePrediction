
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns

def tf_model(X_train_float,y_train_float,epochs_):
    model_tf = Sequential([
        Dense(64,'relu',input_shape=(X_train_float.shape[1],)),
        Dense(64,'relu'),
        Dense(64,'relu'),
        Dense(1,'relu')
    ])
    
    model_tf.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_absolute_error'])
    
    
    epochs = epochs_
    history = model_tf.fit(X_train_float,y_train_float,epochs=epochs,validation_split=0.2,batch_size=32)

    return history, model_tf


def plots(history,epochs):
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    mse = history.history['mean_absolute_error']
    val_mse = history.history['val_mean_absolute_error']
    epochs_range = range(epochs)
    
    fig,axs = plt.subplots(1,2,figsize=(12,8))
    axs[0].plot(epochs_range,loss,label='Loss')
    axs[0].plot(epochs_range,val_loss,label='Val loss')
    axs[0].legend(loc='upper left')
    axs[0].set_title("Loss vs Val loss")
    
    axs[1].plot(epochs_range,mse,label='Mean absolute error')
    axs[1].plot(epochs_range,val_mse,label='Val Mean absolute error')
    axs[1].legend(loc='upper left')
    axs[1].set_title("Mse vs Val mse")
