"""
    Created by: Matthew Eadie
    Date: 10/01/22

    Work based off RAMS multiframe super resolution 
"""

"""MF-UNet model"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def model_MFAE(LR_size, channels): #Multi frame auto encoder model

    encoder_input = keras.Input(shape=(LR_size, LR_size, channels)) #[(None, 256, 256, 11)]

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_input) #(None, 256, 256, 32)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x) #(None, 256, 256, 64)

    x = layers.MaxPooling2D(2)(x) #(None, 128, 128, 64)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x) #(None, 128, 128, 128)
    #Encoder stop
        
    #Decoder start
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x) #(None, 128, 128, 128)

    x = layers.UpSampling2D(2)(x) #(None, 256, 256, 128)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x) #(None, 128, 128, 64)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x) #(None, 256, 256, 32)
    decoder_output = layers.Conv2D(3, 3, activation='relu', padding='same')(x) #(None, 256, 256, 3)

    MFAE = keras.Model(encoder_input, decoder_output, name="MFAE_Matt")

    return MFAE


def model_SIAE(LR_size, channels): #UNet designed for three channel gray images

    encoder_input = keras.Input(shape=(LR_size, LR_size, channels))                 #[(None, 256, 256, 1)]
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_input)  #(None, 256, 256, 32)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)              #(None, 256, 256, 64)
    x = layers.MaxPooling2D(2)(x)                                               #(None, 128, 128, 64)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)             #(None, 128, 128, 128)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)             #(None, 128, 128, 128)
    #Encoder stop
        
    #Decoder start
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)    #(None, 128, 128, 128
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)     #(None, 128, 128, 64)
    x = layers.UpSampling2D(2)(x)                                               #(None, 256, 256, 64)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)     #(None, 256, 256, 32)
    decoder_output = layers.Conv2D(3, 3, activation='relu', padding='same')(x) #(None, 256, 256, 1)
    SIAE = keras.Model(encoder_input, decoder_output, name="SIAE_Matt")

    return SIAE



def model_optimiser():

    optimiser = tf.keras.optimizers.Adam(1e-4)

    return optimiser

def model_loss(y_true, y_pred):#, y_true, y_pred):
        MAE_Loss_Percentage = 1e-3
        MSE_Loss_Percentage = 1.0
        VGG_Loss_Percentage = 1e-3
        SSIM_Loss_Percentage = 1e-3
        
        MSE_loss = tf.keras.losses.MSE(y_true=y_true,y_pred=y_pred) * MSE_Loss_Percentage
        MAE_loss = tf.keras.losses.MAE(y_true=y_true,y_pred=y_pred) * MAE_Loss_Percentage
        SSIM_loss = tf.reduce_mean(tf.image.ssim(img1=y_true,img2=y_pred,max_val=1)) * SSIM_Loss_Percentage
        SSIM_loss = 1-SSIM_loss    
        Pixel_loss = MSE_loss + MAE_loss + SSIM_loss
        
        #VGG_loss = tf.reduce_mean(tf.square(VGG_19(y_true) - VGG_19(y_pred)))    
        #VGG_loss = VGG_loss*self.VGG_Loss_Percentage
        
        return (Pixel_loss)