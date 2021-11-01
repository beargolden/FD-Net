# coding=utf-8

# DILATION_RATE: 空洞卷积扩张率
DILATION_RATE = [2, 3, 5, 7]

from keras.layers import *
from keras.models import *


def conv_block(input_tensor, num_filters):
    out = Conv2D(num_filters, kernel_size=3, padding="same", kernel_initializer="he_normal")(input_tensor)
    # out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(num_filters, kernel_size=3, padding="same", kernel_initializer="he_normal")(out)
    # out = BatchNormalization()(out)
    out = Activation("relu")(out)
    return out


def dilated_conv(input_tensor, num_filters, dilation_rate):
    out = Conv2D(num_filters, kernel_size=3, dilation_rate=dilation_rate, padding="same", kernel_initializer="he_normal")(input_tensor)
    # out = BatchNormalization()(out)
    out = Activation("relu")(out)
    return out


def up_conv(input_tensor, num_filters, up_size):
    out = UpSampling2D(up_size, interpolation="bilinear")(input_tensor)
    out = Conv2D(num_filters, kernel_size=3, padding="same", kernel_initializer="he_normal")(out)
    # out = BatchNormalization()(out)
    out = Activation("relu")(out)
    return out


def down_conv(input_tensor, num_filters, down_size):
    out = MaxPooling2D(down_size)(input_tensor)
    out = Conv2D(num_filters, kernel_size=3, padding="same", kernel_initializer="he_normal")(out)
    # out = BatchNormalization()(out)
    out = Activation("relu")(out)
    return out


def DilatedNet(num_classes, input_height, input_width, num_filters):
    """
    Fully Dilated Convolutional Network - Basic Implementation
    """
    x = inputs = Input(shape=(input_height, input_width, 1))

    depth = 4
    skips = []

    for i in range(depth):
        x = conv_block(x, num_filters)
        skips.append(x)
        x = dilated_conv(x, num_filters, DILATION_RATE[i])  # input_tensor = MaxPooling2D(pool_size=2)(input_tensor)
        num_filters *= 2

    x = conv_block(x, num_filters)

    for i in reversed(range(depth)):
        num_filters //= 2
        x = dilated_conv(x, num_filters, DILATION_RATE[i])  # input_tensor = UpSampling2D(upscale_factor=2)(input_tensor)
        x = Concatenate()([skips[i], x])
        x = conv_block(x, num_filters)

    outputs = Conv2D(num_classes, kernel_size=3, padding="same", kernel_initializer="he_normal")(x)

    # outputHeight = Model(inputs, outputs).output_shape[1]
    # outputWidth = Model(inputs, outputs).output_shape[2]
    #
    # outputs = (Reshape((outputHeight * outputWidth, num_classes)))(outputs)
    # outputs = Activation("softmax")(outputs)
    outputs = Activation("sigmoid")(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    # model.outputHeight = outputHeight
    # model.outputWidth = outputWidth

    return model
