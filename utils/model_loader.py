import tensorflow as tf
from SRgan import Generator

class DepthToSpace(Layer):
    def __init__(self, block_size, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size

    def call(self, x):
        return tf.nn.depth_to_space(x, self.block_size)

    def get_config(self):
        config = super().get_config()
        config['block_size'] = self.block_size
        return config

def load_wgan():
    # Your code here to load WGAN TF model
    model = tf.keras.models.load_model('models/wgan_model.h5')
    return model

def load_srgan():
    mode1 = Generator((64,64,3),10)
    mode1.load_weights('models/srgan_model.h5')
    return mode1

def load_srwgan():
    model = tf.keras.models.load_model('models/srwgan_model.h5',custom_objects={'DepthToSpace':DepthToSpace})
    return model
