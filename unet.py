from tensorflow.keras.layers import Conv2D,Concatenate,BatchNormalization, Input, Activation, MaxPool2D, Conv2DTranspose
from tensorflow.keras.models import Model

def conv_block(inputs, num_filter):
    x = Conv2D(num_filter, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filter, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder_block(inputs, num_filter):
    x = conv_block(inputs, num_filter)
    p = MaxPool2D((2,2))(x)
    return x, p

def decoder_block(inputs, skip, num_filter):
    x = Conv2DTranspose(num_filter, (2,2), strides=2, padding="same")(inputs)
    x= Concatenate()([x,skip])
    x= conv_block(x, num_filter)
    return x

def build_unet(inputs_shape, num_classes):
    inputs = Input(inputs_shape)
    x1, p1 = encoder_block(inputs, 64)
    x2, p2 = encoder_block(p1, 126)
    x3, p3 = encoder_block(p2, 256)
    x4, p4 = encoder_block(p3, 512)
    b1 = conv_block(p4, 1024)
    print(b1.shape)
    print(x1.shape, x2.shape, x3.shape, x4.shape)
    d1= decoder_block(b1, x4, 512)
    d2 = decoder_block(d1, x3, 256)
    d3 = decoder_block(d2, x2, 128)
    d4 = decoder_block(d3, x1, 64)
    outputs= Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)
    model= Model(inputs, outputs)
    return model

if __name__ == "__main__":
    inputs_shape = (512, 512, 3)
    model= build_unet(inputs_shape, 11)
    model.summary()