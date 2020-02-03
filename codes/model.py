import os

from keras import backend as K
from keras import objectives
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, AveragePooling2D, Dropout, AtrousConvolution2D, Conv2DTranspose, Lambda, DepthwiseConv2D, SeparableConv2D
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten
from keras.layers.merge import Concatenate, Multiply, Add
from keras.layers.normalization import BatchNormalization
from keras_contrib.layers import InstanceNormalization
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard

from groupnormalization import GroupNormalization

os.environ['KERAS_BACKEND'] = 'tensorflow'
K.set_image_dim_ordering('tf')

def res_block(x, n_filters):

    res = Conv2D(filters=n_filters, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    res = InstanceNormalization(axis=3)(res)
    res = LeakyReLU(0.2)(res)

    # res = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(res)
    res = AtrousConvolution2D(n_filters, 3, 3, atrous_rate=(2, 2), border_mode='same')(res)
    res = InstanceNormalization(axis=3)(res)
    res = Activation('relu')(res)
    # res = LeakyReLU(0.2)(res)
    #
    # res = Conv2D(filters=n_filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(res)
    # res = InstanceNormalization(axis=3)(res)
    # res = LeakyReLU(0.2)(res)

    x = Conv2D(filters=n_filters, kernel_size=(1,1), strides=(1,1), padding='same')(x)

    out = Add()([x, res])

    return out

def generator(img_size, n_filters, name='g'):
    """
    generate network based on unet
    """

    img_height, img_width = img_size[0], img_size[1]
    img_ch = 3
    out_ch = 1
    kernel_size = (3, 3)
    strides = (1, 1)
    padding = 'same'
    inputs = Input((img_height, img_width, img_ch))

    conv1 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    # atrous_conv1 = AtrousConvolution2D(n_filters, 3,3, atrous_rate=(2,2), border_mode='same')(conv1)
    conv1 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    down1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=n_filters * 2, kernel_size=kernel_size, strides=(1, 1), padding=padding)(down1)
    # conv2 = InstanceNormalization()(conv2)
    conv2 = GroupNormalization(groups=16, axis=3, scale=False)(conv2)
    # conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = LeakyReLU(0.2)(conv2)
    # atrous_conv2 = AtrousConvolution2D(n_filters*2, 3, 3, atrous_rate=(2, 2), border_mode='same')(conv2)
    conv2 = Conv2D(filters=n_filters * 2, kernel_size=kernel_size, strides=(1, 1), padding=padding)(conv2)
    # conv2 = InstanceNormalization()(conv2)
    conv2 = GroupNormalization(groups=16, axis=3, scale=False)(conv2)
    # conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = LeakyReLU(0.2)(conv2)
    # conv2 = Activation('relu')(conv2)
    down2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=n_filters * 4, kernel_size=kernel_size, strides=(1, 1), padding=padding)(down2)
    # conv3 = InstanceNormalization()(conv3)
    conv3 = GroupNormalization(groups=16, axis=3, scale=False)(conv3)
    # conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = LeakyReLU(0.2)(conv3)
    atrous_conv3 = AtrousConvolution2D(n_filters*4, 3, 3, atrous_rate=(2, 2), border_mode='same')(conv3)
    # conv3 = Conv2D(filters=n_filters * 4, kernel_size=kernel_size, strides=(1, 1), padding=padding)(conv3)
    # atrous_conv3 = InstanceNormalization()(atrous_conv3)
    atrous_conv3 = GroupNormalization(groups=16, axis=3, scale=False)(atrous_conv3)
    # conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    atrous_conv3 = LeakyReLU(0.2)(atrous_conv3)
    # conv3 = Activation('relu')(conv3)
    # down3 = AveragePooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters=n_filters * 4, kernel_size=kernel_size, strides=(1, 1), padding=padding)(atrous_conv3)
    # conv4 = InstanceNormalization()(conv4)
    conv4 = GroupNormalization(groups=16, axis=3, scale=False)(conv4)
    # conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = LeakyReLU(0.2)(conv4)
    # conv4 = Conv2D(filters=n_filters * 8, kernel_size=kernel_size, strides=(1, 1), padding=padding)(conv4)
    atrous_conv4 = AtrousConvolution2D(n_filters*4, 3, 3, atrous_rate=(2, 2), border_mode='same')(conv4)
    # atrous_conv4 = InstanceNormalization()(atrous_conv4)
    atrous_conv4 = GroupNormalization(groups=16, axis=3, scale=False)(atrous_conv4)
    # conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    atrous_conv4 = LeakyReLU(0.2)(atrous_conv4)
    # conv4 = Activation('relu')(conv4)

    # conv2 = res_block(down1, n_filters*2)
    # down2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    # conv3 = res_block(down2, n_filters*4)
    # down3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    # conv4 = res_block(down3, n_filters*8)

    att_g1 = attention_gate(x=atrous_conv3, signal=atrous_conv4, n_filters=n_filters * 4, padding='same')
    concat1 = Concatenate(axis=3)([conv3, att_g1])
    up1 = Conv2D(filters=n_filters * 4, kernel_size=kernel_size, strides=(1, 1), padding=padding)(concat1)
    # up1 = InstanceNormalization()(up1)
    up1 = GroupNormalization(groups=16, axis=3, scale=False)(up1)
    # up1 = BatchNormalization(scale=False, axis=3)(up1)
    up1 = LeakyReLU(0.2)(up1)
    # up1 = Conv2D(filters=n_filters * 4, kernel_size=kernel_size, strides=(1, 1), padding=padding)(up1)
    up1 = AtrousConvolution2D(n_filters*4, 3,3,atrous_rate=(2,2), border_mode='same')(up1)
    # up1 = InstanceNormalization()(up1)
    up1 = GroupNormalization(groups=16, axis=3, scale=False)(up1)
    # up1 = BatchNormalization(scale=False, axis=3)(up1)
    up1 = LeakyReLU(0.2)(up1)
    # up1 = Activation('relu')(up1)
    # up1 = res_block(concat1, n_filters*4)

    up2 = Conv2DTranspose(filters=n_filters * 2, kernel_size=(3,3), strides=(2,2), padding=padding)(up1)
    att_g2 = attention_gate(x=conv2, signal=up2, n_filters=n_filters * 2, padding='same')
    concat2 = Concatenate(axis=3)([up2, att_g2])
    up2 = Conv2D(filters=n_filters * 2, kernel_size=kernel_size, strides=(1, 1), padding=padding)(concat2)
    # up2 = InstanceNormalization()(up2)
    up2 = GroupNormalization(groups=16, axis=3, scale=False)(up2)
    # up2 = BatchNormalization(scale=False, axis=3)(up2)
    up2 = LeakyReLU(0.2)(up2)
    # up2 = Conv2D(filters=n_filters * 2, kernel_size=kernel_size, strides=(1, 1), padding=padding)(up2)
    up2 = AtrousConvolution2D(n_filters * 2, 3, 3, atrous_rate=(2, 2), border_mode='same')(up2)
    # up2 = InstanceNormalization()(up2)
    up2 = GroupNormalization(groups=16, axis=3, scale=False)(up2)
    # up2 = BatchNormalization(scale=False, axis=3)(up2)
    up2 = LeakyReLU(0.2)(up2)
    # up2 = Activation('relu')(up2)
    # up2 = res_block(concat2, n_filters * 2)

    up3 = Conv2DTranspose(filters=n_filters, kernel_size=(3,3), strides=(2,2), padding=padding)(up2)
    att_g3 = attention_gate(x=conv1, signal=up3, n_filters=n_filters, padding='same')
    concat3 = Concatenate(axis=3)([up3, att_g3])
    up3 = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1, 1), padding=padding)(concat3)
    # up3 = InstanceNormalization()(up3)
    up3 = GroupNormalization(groups=16, axis=3, scale=False)(up3)
    # up3 = BatchNormalization(scale=False, axis=3)(up3)
    up3 = LeakyReLU(0.2)(up3)
    # up3 = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1, 1), padding=padding)(up3)
    up3 = AtrousConvolution2D(n_filters, 3, 3, atrous_rate=(2, 2), border_mode='same')(up3)
    # up3 = InstanceNormalization()(up3)
    up3 = GroupNormalization(groups=16, axis=3, scale=False)(up3)
    # up3 = BatchNormalization(scale=False, axis=3)(up3)
    up3 = LeakyReLU(0.2)(up3)
    # up3 = Activation('relu')(up3)
    # up3 = res_block(concat3, n_filters)

    outputs = Conv2D(out_ch, kernel_size=(1, 1), strides=(1, 1), padding='same')(up3)
    outputs = Activation('sigmoid')(outputs)

    g = Model(inputs, outputs, name=name)

    return g

def unet(img_size, n_filters, name='unet'):
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'

    inputs = Input((img_height, img_width, img_ch))
    conv1 = Conv2D(n_filters, (k, k), padding=padding)(inputs)
    conv1 = Conv2D(n_filters, (k, k), padding=padding)(conv1)
    pool1 = AveragePooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding)(pool1)
    conv2 = InstanceNormalization(scale=False, axis=3)(conv2)
    # conv2 = LeakyReLU(0.2)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding)(conv2)
    conv2 = InstanceNormalization(scale=False, axis=3)(conv2)
    # conv2 = LeakyReLU(0.2)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = AveragePooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding)(pool2)
    conv3 = InstanceNormalization(scale=False, axis=3)(conv3)
    # conv3 = LeakyReLU(0.2)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding)(conv3)
    conv3 = InstanceNormalization(scale=False, axis=3)(conv3)
    # conv3 = LeakyReLU(0.2)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = AveragePooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding)(pool3)
    conv4 = InstanceNormalization(scale=False, axis=3)(conv4)
    # conv4 = LeakyReLU(0.2)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding)(conv4)
    conv4 = InstanceNormalization(scale=False, axis=3)(conv4)
    # conv4 = LeakyReLU(0.2)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = AveragePooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding)(pool4)
    conv5 = InstanceNormalization(scale=False, axis=3)(conv5)
    # conv5 = LeakyReLU(0.2)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding)(conv5)
    conv5 = InstanceNormalization(scale=False, axis=3)(conv5)
    # conv5 = LeakyReLU(0.2)(conv5)
    conv5 = Activation('relu')(conv5)

    up1 = Conv2DTranspose(filters=8 * n_filters, kernel_size=(3, 3), strides=(2, 2), padding=padding)(conv5)
    concat1 = Concatenate(axis=3)([up1, conv4])
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding)(concat1)
    conv6 = InstanceNormalization(scale=False, axis=3)(conv6)
    # conv6 = LeakyReLU(0.2)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding)(conv6)
    conv6 = InstanceNormalization(scale=False, axis=3)(conv6)
    # conv6 = LeakyReLU(0.2)(conv6)
    conv6 = Activation('relu')(conv6)

    up2 = Conv2DTranspose(filters=4 * n_filters, kernel_size=(3, 3), strides=(2, 2), padding=padding)(conv6)
    concat2 = Concatenate(axis=3)([up2, conv3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding)(concat2)
    conv7 = InstanceNormalization(scale=False, axis=3)(conv7)
    # conv7 = LeakyReLU(0.2)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding)(conv7)
    conv7 = InstanceNormalization(scale=False, axis=3)(conv7)
    # conv7 = LeakyReLU(0.2)(conv7)
    conv7 = Activation('relu')(conv7)

    up3 = Conv2DTranspose(filters=2 * n_filters, kernel_size=(3, 3), strides=(2, 2), padding=padding)(conv7)
    concat3 = Concatenate(axis=3)([up3, conv2])
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding)(concat3)
    conv8 = InstanceNormalization(scale=False, axis=3)(conv8)
    # conv8 = LeakyReLU(0.2)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding)(conv8)
    conv8 = InstanceNormalization(scale=False, axis=3)(conv8)
    # conv8 = LeakyReLU(0.2)(conv8)
    conv8 = Activation('relu')(conv8)

    up4 = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding=padding)(conv8)
    concat4 = Concatenate(axis=3)([up4, conv1])
    conv9 = Conv2D(n_filters, (k, k), padding=padding)(concat4)
    conv9 = InstanceNormalization(scale=False, axis=3)(conv9)
    # conv9 = LeakyReLU(0.2)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k), padding=padding)(conv9)
    conv9 = InstanceNormalization(scale=False, axis=3)(conv9)
    # conv9 = LeakyReLU(0.2)(conv9)
    conv9 = Activation('relu')(conv9)

    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(conv9)

    g = Model(inputs, outputs, name=name)

    return g

def expand_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat

def attention_gate(x, signal, n_filters, padding='same'):

    x_ = Conv2D(filters=n_filters, kernel_size=(1,1), strides=(1,1), padding=padding)(x)
    # x_ = InstanceNormalization()(x_)

    g = Conv2D(filters=n_filters, kernel_size=(1,1), strides=(1,1), padding=padding)(signal)
    # g = AtrousConvolution2D(n_filters, 3, 3, atrous_rate=(2, 2), border_mode='same')(signal)
    # g = AtrousConvolution2D(n_filters, 3, 3, atrous_rate=(2, 2), border_mode='same')(signal)
    # g = InstanceNormalization()(g)

    add_x = Add()([x_, g])
    # act_x = LeakyReLU(0.2)(add_x)
    # add_x = InstanceNormalization()(add_x)
    add_x = GroupNormalization(groups=16, axis=3, scale=False)(add_x)
    act_x = Activation('relu')(add_x)

    conv_x= Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='same')(act_x)
    conv_x = InstanceNormalization()(conv_x)
    # conv_x = GroupNormalization(groups=16, axis=3, scale=False)(conv_x)
    sig_x = Activation('sigmoid')(conv_x)
    # up_x = UpSampling2D(size=(2,2))(sig_x)
    # up_x = Conv2DTranspose(filters=n_filters, kernel_size=(3,3), strides=(2,2), padding='same')(sig_x)
    sig_x = expand_as(sig_x, n_filters)

    out = Multiply()([sig_x, x])
    out = Conv2D(n_filters, kernel_size=(1,1),strides=(1,1),padding=padding)(out)
    # out = InstanceNormalization()(out)
    out = GroupNormalization(groups=16, axis=3, scale=False)(out)

    return out

def attention_generator(img_size, n_filters=32, name='agn'):

    img_height, img_width = img_size[0], img_size[1]
    img_ch = 3
    out_ch = 1
    kernel_size = (3,3)
    strides = (2,2)
    padding = 'same'
    inputs = Input((img_height, img_width, img_ch))

    conv1 = Conv2D(filters=n_filters, kernel_size=(3,3), strides=(1,1), padding='same')(inputs)
    conv1 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    down1 = AveragePooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(filters=n_filters*2, kernel_size=kernel_size, strides=(1, 1), padding=padding)(down1)
    conv2 = InstanceNormalization()(conv2)
    conv2 = LeakyReLU(0.2)(conv2)
    conv2 = Conv2D(filters=n_filters*2, kernel_size=kernel_size, strides=(1, 1), padding=padding)(conv2)
    conv2 = InstanceNormalization()(conv2)
    conv2 = LeakyReLU(0.2)(conv2)
    down2 = AveragePooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(filters=n_filters*4, kernel_size=kernel_size, strides=(1, 1), padding=padding)(down2)
    conv3 = InstanceNormalization()(conv3)
    conv3 = LeakyReLU(0.2)(conv3)
    conv3 = Conv2D(filters=n_filters*4, kernel_size=kernel_size, strides=(1, 1), padding=padding)(conv3)
    conv3 = InstanceNormalization()(conv3)
    conv3 = LeakyReLU(0.2)(conv3)
    down3 = AveragePooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters=n_filters*8, kernel_size=kernel_size, strides=(1, 1), padding=padding)(down3)
    conv4 = InstanceNormalization()(conv4)
    conv4 = LeakyReLU(0.2)(conv4)
    conv4 = Conv2D(filters=n_filters*8, kernel_size=kernel_size, strides=(1, 1), padding=padding)(conv4)
    conv4 = InstanceNormalization()(conv4)
    conv4 = LeakyReLU(0.2)(conv4)

    up1 = Conv2DTranspose(filters=n_filters*8, kernel_size=kernel_size, strides=strides, padding=padding)(conv4)
    att_g1 = attention_gate(x=conv3, signal=conv4, n_filters=n_filters*4, padding='same')
    concat1 = Concatenate(axis=3)([up1, att_g1])
    up1 = Conv2D(filters=n_filters*4, kernel_size=kernel_size, strides=(1, 1), padding=padding)(concat1)
    up1 = InstanceNormalization()(up1)
    up1 = LeakyReLU(0.2)(up1)
    up1 = Conv2D(filters=n_filters*4, kernel_size=kernel_size, strides=(1, 1), padding=padding)(up1)
    up1 = InstanceNormalization()(up1)
    up1 = LeakyReLU(0.2)(up1)

    up2 = Conv2DTranspose(filters=n_filters * 4, kernel_size=kernel_size, strides=strides, padding=padding)(up1)
    att_g2 = attention_gate(x=conv2, signal=up1, n_filters=n_filters * 2, padding='same')
    concat2 = Concatenate(axis=3)([up2, att_g2])
    up2 = Conv2D(filters=n_filters * 2, kernel_size=kernel_size, strides=(1, 1), padding=padding)(concat2)
    up2 = InstanceNormalization()(up2)
    up2 = LeakyReLU(0.2)(up2)
    up2 = Conv2D(filters=n_filters * 2, kernel_size=kernel_size, strides=(1,1), padding=padding)(up2)
    up2 = InstanceNormalization()(up2)
    up2 = LeakyReLU(0.2)(up2)

    up3 = Conv2DTranspose(filters=n_filters * 2, kernel_size=kernel_size, strides=strides, padding=padding)(up2)
    att_g3 = attention_gate(x=conv1, signal=up2, n_filters=n_filters, padding='same')
    concat3 = Concatenate(axis=3)([up3, att_g3])
    up3 = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1, 1), padding=padding)(concat3)
    up3 = InstanceNormalization()(up3)
    up3 = LeakyReLU(0.2)(up3)
    up3 = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1,1), padding=padding)(up3)
    up3 = InstanceNormalization()(up3)
    up3 = LeakyReLU(0.2)(up3)

    outputs = Conv2D(out_ch, kernel_size=(1,1), strides=(1,1), padding='same')(up3)
    outputs = Activation('sigmoid')(outputs)

    g = Model(inputs, outputs, name=name)

    return g

def recurrent_block(x, n_filters, t=2):

    padding='same'

    x_ = Conv2D(filters=n_filters, kernel_size=(1, 1), strides=(1, 1), padding=padding)(x)

    for i in range(t):
        if i == 0:
            x1 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), padding=padding)(x)
            x1 = InstanceNormalization()(x1)
            x1 = LeakyReLU(0.2)(x1)

        a = Add()([x_, x1])
        x1 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), padding=padding)(a)
        x1 = InstanceNormalization()(x1)
        x1 = LeakyReLU(0.2)(x1)

    return x1

def r2_generator(img_size, n_filters, name='r2_g'):

    # set image specifics
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]

    inputs = Input((img_height, img_width, img_ch))

    r_conv1 = recurrent_block(inputs, n_filters, t=2)
    # r_conv1 = recurrent_block(r_conv1, n_filters, t=2)
    down1 = MaxPooling2D(pool_size=(2,2))(r_conv1)

    r_conv2 = recurrent_block(down1, n_filters*2, t=2)
    # r_conv2 = recurrent_block(r_conv2, n_filters*2, t=2)
    down2 = MaxPooling2D(pool_size=(2, 2))(r_conv2)

    r_conv3 = recurrent_block(down2, n_filters * 4, t=2)
    # r_conv3 = recurrent_block(r_conv3, n_filters * 4, t=2)
    down3 = MaxPooling2D(pool_size=(2, 2))(r_conv3)

    r_conv4 = recurrent_block(down3, n_filters * 8, t=2)
    # r_conv4 = recurrent_block(r_conv4, n_filters * 8, t=2)
    down4 = MaxPooling2D(pool_size=(2, 2))(r_conv4)

    r_conv5 = recurrent_block(down4, n_filters * 16, t=2)
    # r_conv5 = recurrent_block(r_conv5, n_filters * 16, t=2)

    up1 = Conv2DTranspose(filters=n_filters * 8, kernel_size=(3, 3), strides=(2, 2), padding='same')(r_conv5)
    concat1 = Concatenate(axis=3)([r_conv4, up1])
    r_conv6 = recurrent_block(concat1, n_filters*8, t=2)
    # r_conv6 = recurrent_block(r_conv6, n_filters * 8, t=2)

    up2 = Conv2DTranspose(filters=n_filters * 4, kernel_size=(3, 3), strides=(2, 2), padding='same')(r_conv6)
    concat2 = Concatenate(axis=3)([r_conv3, up2])
    r_conv7 = recurrent_block(concat2, n_filters * 4, t=2)
    # r_conv7 = recurrent_block(r_conv7, n_filters * 4, t=2)

    up3 = Conv2DTranspose(filters=n_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(r_conv7)
    concat3 = Concatenate(axis=3)([r_conv2, up3])
    r_conv8 = recurrent_block(concat3, n_filters * 2, t=2)
    # r_conv8 = recurrent_block(r_conv8, n_filters * 2, t=2)

    up4 = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(r_conv8)
    concat4 = Concatenate(axis=3)([r_conv1, up4])
    r_conv9 = recurrent_block(concat4, n_filters, t=2)
    # r_conv9 = recurrent_block(r_conv9, n_filters, t=2)

    outputs = Conv2D(out_ch, (1, 1), padding='same', activation='sigmoid')(r_conv9)

    g = Model(inputs, outputs, name=name)
    return g

def atrous_unet(img_size, n_filters, name='atrous_unet'):
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'

    inputs = Input((img_height, img_width, img_ch))
    conv1 = Conv2D(n_filters, (k, k), padding=padding)(inputs)
    pool1 = AveragePooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding)(pool1)
    # conv2 = InstanceNormalization(scale=False, axis=3)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    # conv2 = LeakyReLU(0.2)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding)(conv2)
    # conv2 = InstanceNormalization(scale=False, axis=3)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    # conv2 = LeakyReLU(0.2)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = AveragePooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding)(pool2)
    # conv3 = InstanceNormalization(scale=False, axis=3)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    # conv3 = LeakyReLU(0.2)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding)(conv3)
    # conv3 = InstanceNormalization(scale=False, axis=3)(conv3)
    # conv3 = LeakyReLU(0.2)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = AveragePooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding)(pool3)
    # conv4 = InstanceNormalization(scale=False, axis=3)(conv4)
    # conv4 = LeakyReLU(0.2)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = AtrousConvolution2D(8*n_filters, k, k, atrous_rate=(2,2), border_mode='same')(conv4)
    # conv4 = InstanceNormalization(scale=False, axis=3)(conv4)
    # conv4 = LeakyReLU(0.2)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = AveragePooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding)(pool4)
    # conv5 = InstanceNormalization(scale=False, axis=3)(conv5)
    # conv5 = LeakyReLU(0.2)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = AtrousConvolution2D(16*n_filters, k, k, atrous_rate=(2,2), border_mode='same')(conv5)
    # conv5 = InstanceNormalization(scale=False, axis=3)(conv5)
    # conv5 = LeakyReLU(0.2)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    up1 = Conv2DTranspose(filters=8*n_filters, kernel_size=(4,4), strides=(2,2), padding=padding)(conv5)
    concat1 = Concatenate(axis=3)([up1, conv4])
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding)(concat1)
    # conv6 = InstanceNormalization(scale=False, axis=3)(conv6)
    # conv6 = LeakyReLU(0.2)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = AtrousConvolution2D(8*n_filters, k, k, atrous_rate=(2,2), border_mode='same')(conv6)
    # conv6 = InstanceNormalization(scale=False, axis=3)(conv6)
    # conv6 = LeakyReLU(0.2)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    up2 = Conv2DTranspose(filters=4 * n_filters, kernel_size=(4, 4), strides=(2, 2), padding=padding)(conv6)
    concat2 = Concatenate(axis=3)([up2, conv3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding)(concat2)
    # conv7 = InstanceNormalization(scale=False, axis=3)(conv7)
    # conv7 = LeakyReLU(0.2)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = AtrousConvolution2D(4*n_filters, k, k, atrous_rate=(2,2), border_mode='same')(conv7)
    # conv7 = InstanceNormalization(scale=False, axis=3)(conv7)
    # conv7 = LeakyReLU(0.2)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    up3 = Conv2DTranspose(filters=2 * n_filters, kernel_size=(4, 4), strides=(2, 2), padding=padding)(conv7)
    concat3 = Concatenate(axis=3)([up3, conv2])
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding)(concat3)
    # conv8 = InstanceNormalization(scale=False, axis=3)(conv8)
    # conv8 = LeakyReLU(0.2)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = AtrousConvolution2D(2*n_filters, k, k, atrous_rate=(2,2), border_mode='same')(conv8)
    # conv8 = InstanceNormalization(scale=False, axis=3)(conv8)
    # conv8 = LeakyReLU(0.2)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    up4 = Conv2DTranspose(filters=n_filters, kernel_size=(4, 4), strides=(2, 2), padding=padding)(conv8)
    concat4 = Concatenate(axis=3)([up4, conv1])
    conv9 = Conv2D(n_filters, (k, k), padding=padding)(concat4)
    # conv9 = InstanceNormalization(scale=False, axis=3)(conv9)
    # conv9 = LeakyReLU(0.2)(conv9)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = AtrousConvolution2D(n_filters, k, k, atrous_rate=(2,2), border_mode='same')(conv9)
    # conv9 = InstanceNormalization(scale=False, axis=3)(conv9)
    # conv9 = LeakyReLU(0.2)(conv9)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(conv9)

    g = Model(inputs, outputs, name=name)

    return g

def feature_extractor(x, n_filters, isfirst=False):

    if isfirst:
        conv1 = Conv2D(filters=n_filters, kernel_size=(1,1), strides=(1,1), padding='same')(x)
        conv1 = Conv2D(filters=n_filters, kernel_size=(3,3), strides=(1,1), padding='same')(conv1)
    else:
        conv1 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    conv2 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    conv3 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv2)
    conv4 = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv3)

    out = Concatenate(axis=3)([conv1, conv2, conv3,conv4])

    return out

def discriminator(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (pixel GAN)
    """

    # set image specifics
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]

    inputs = Input((img_height, img_width, img_ch + out_ch))

    # conv1 = Conv2D(n_filters, kernel_size=(3, 3), padding="same")(inputs)
    # conv1 = LeakyReLU(0.2)(conv1)
    #
    # conv2 = Conv2D(2 * n_filters, kernel_size=(3, 3), padding="same")(conv1)
    # conv2 = LeakyReLU(0.2)(conv2)
    #
    # conv3 = Conv2D(4 * n_filters, kernel_size=(3, 3), padding="same")(conv2)
    # conv3 = LeakyReLU(0.2)(conv3)
    #
    # outputs = Conv2D(out_ch, kernel_size=(1, 1), padding="same")(conv3)

    conv1 = feature_extractor(inputs, n_filters, isfirst=True)
    conv1 = LeakyReLU(0.2)(conv1)
    conv2 = feature_extractor(conv1, n_filters)
    conv2 = LeakyReLU(0.2)(conv2)
    conv3 = feature_extractor(conv2, n_filters)
    conv3 = LeakyReLU(0.2)(conv3)
    outputs = Concatenate(axis=3)([inputs, conv1, conv2, conv3])
    outputs = Conv2D(out_ch, kernel_size=(1, 1), padding="same")(outputs)

    outputs = Activation('sigmoid')(outputs)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                           K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])

    return d, d.layers[-1].output_shape[1:]

def GAN(g, d, img_size, n_filters_g, n_filters_d, alpha_recip, init_lr, name='gan'):
    """
    GAN (that binds generator and discriminator)
    """
    img_h, img_w = img_size[0], img_size[1]

    img_ch = 3
    seg_ch = 1

    fundus = Input((img_h, img_w, img_ch))
    vessel = Input((img_h, img_w, seg_ch))

    fake_vessel = g(fundus)
    fake_pair = Concatenate(axis=3)([fundus, fake_vessel])

    gan = Model([fundus, vessel], d(fake_pair), name=name)

    def gan_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)

        vessel_flat = K.batch_flatten(vessel)
        fake_vessel_flat = K.batch_flatten(fake_vessel)

        L_seg = objectives.binary_crossentropy(vessel_flat, fake_vessel_flat)

        return L_adv + alpha_recip*L_seg

    gan.compile(optimizer=RMSprop(lr=init_lr), loss=gan_loss, metrics=['accuracy'])

    return gan

