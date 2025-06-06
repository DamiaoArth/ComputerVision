from keras.layers import *
from keras.models import Model

# Convolutional block
def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

# Build the model
def build_model(shape):
    num_filters = [64, 128, 256, 512]
    inputs = Input(shape)
    skip_x = []
    x = inputs
    
    # Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)
    
    x = conv_block(x, 1024)
    num_filters.reverse()  # 512, 256, 128, 64
    skip_x.reverse()
    
    # Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)
    
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)  # Binary
    return Model(inputs, x)

