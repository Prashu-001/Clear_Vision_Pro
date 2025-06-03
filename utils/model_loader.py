import tensorflow as tf
from SRgan import Generator
def load_wgan():
    # Your code here to load WGAN TF model
    model = tf.keras.models.load_model('models/wgan_model.h5')
    return model

def load_srgan():
    mode1 = Generator((64,64,3),10)
    mode1.load_weights('models/srgan_model.h5')
    return mode1

def load_srwgan():
    model = tf.keras.models.load_model('models/srwgan_model.h5')
    return model
