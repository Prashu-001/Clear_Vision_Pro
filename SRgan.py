import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D,Dense,Conv2DTranspose,Flatten,BatchNormalization,LeakyReLU,PReLU

def ConvBlock(x,out_channels,kernel_size,use_bn=True,use_act=True,discriminator=False,**kwargs):
    x=Conv2D(out_channels,kernel_size,**kwargs)(x) #**kwargs for extra arguements
    if use_bn:
        x=BatchNormalization()(x)
    if use_act:
        if discriminator:
            x=LeakyReLU(alpha=0.2)(x)
        else:
            x=PReLU(shared_axes=[1,2])(x)
    return x
    
def ResidualBlock(x,channels):
    res=ConvBlock(x,channels,kernel_size=3,padding='same',strides=(1,1))
    res=ConvBlock(res,channels,kernel_size=3,padding='same',strides=(1,1),use_act=False)
    out = keras.layers.Add()([x, res]) 
    return out
    
def UpsampleBlock(x,filters,scale=2):
    y=Conv2D(filters*scale**2,(3,3),strides=(1,1),padding='same')(x)
    y=keras.layers.Lambda(lambda t: tf.nn.depth_to_space(t, scale))(y)
    y=PReLU(shared_axes=[1,2])(y)
    return y
    
def Generator(input_shape=(64, 64, 3), res_blocks=16):
    inputs = keras.Input(shape=input_shape)
    x = ConvBlock(inputs, out_channels=64, kernel_size=9, strides=(1, 1), use_bn=False, padding='same')
    res = x
    
    for _ in range(res_blocks):
        res = ResidualBlock(res, channels=64)
        
    res = ConvBlock(res, out_channels=64, kernel_size=3, strides=(1, 1), use_act=False, padding='same')
    x = keras.layers.Add()([x, res])
    
    x = UpsampleBlock(x, filters=64, scale=2)
    x = ConvBlock(x, out_channels=3, kernel_size=9, strides=(1, 1), use_bn=False, use_act=False, padding='same')
    x = keras.layers.Activation('tanh')(x)
    return keras.Model(inputs, x, name='Generator')