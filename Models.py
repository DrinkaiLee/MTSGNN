from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Permute
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten, Concatenate
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l1_l2


def MTSGNN(nb_classes=12, Chans=8, Samples=256, 
           dropoutRate=0.5, kernLength=256, F1=96, D=1):
    input1 = Input(shape=(Chans, Samples, 1))
    
    # 256
    layer_mt_1 = Conv2D(int(F1/4), (1, int(kernLength)), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    layer_mt_1 = BatchNormalization()(layer_mt_1)
    # 128
    layer_mt_2 = Conv2D(int(F1/4), (1, int(kernLength/2)), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    layer_mt_2 = BatchNormalization()(layer_mt_2)
    # 64
    layer_mt_3 = Conv2D(int(F1/4), (1, int(kernLength/4)), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    layer_mt_3 = BatchNormalization()(layer_mt_3)
    layer_mt_4 = Conv2D(int(F1/4), (1, int(kernLength/8)), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    layer_mt_4 = BatchNormalization()(layer_mt_4)

    layer_mt = Concatenate(axis=-1)([layer_mt_1,layer_mt_2,layer_mt_3,layer_mt_4])
    layer_s = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(layer_mt)           # (1, Samples, F1*D)
    layer_s = BatchNormalization()(layer_s)                                         # (1, Samples, F1*D)
    layer_s = Activation('elu')(layer_s)                                            # (1, Samples, F1*D)
    layer_s = AveragePooling2D((1, 4))(layer_s)                                     # (1, Samples//4, F1*D)
    layer_s = Dropout(dropoutRate)(layer_s)                                         # (1, Samples//4, F1*D)

    layer_g = SeparableConv2D(nb_classes, (1, 16),
                             use_bias=False, padding='same')(layer_s)               # (1, Samples//4, nb_classes)
    layer_g = BatchNormalization()(layer_g)                                         # (1, Samples//4, nb_classes)
    layer_g = Activation('elu')(layer_g)                                            # (1, Samples//4, nb_classes)
    layer_g = AveragePooling2D((1, Samples/4))(layer_g)                             # (1, 1, nb_classes)

    flatten = Flatten(name='flatten')(layer_g)                                      # (nb_classes)

    softmax = Activation('softmax', name='softmax')(flatten)                        # (nb_classes)

    return Model(inputs=input1, outputs=[softmax])


def EEGNet_v1(nb_classes, Chans=64, Samples=128, regRate=0.0001,
               dropoutRate=0.25, kernels=[(2, 32), (8, 4)], strides=(2, 4)):
    """ Keras Implementation of EEGNet v1 """
    input_main = Input((Chans, Samples, 1))
    layer1 = Conv2D(16, (Chans, 1), input_shape=(Chans, Samples, 1),
                    kernel_regularizer=l1_l2(l1=regRate, l2=regRate))(input_main)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('elu')(layer1)
    layer1 = Dropout(dropoutRate)(layer1)

    permute_dims = 2, 1, 3
    permute1 = Permute(permute_dims)(layer1)

    layer2 = Conv2D(4, kernels[0], padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                    strides=strides)(permute1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('elu')(layer2)
    layer2 = Dropout(dropoutRate)(layer2)

    layer3 = Conv2D(4, kernels[1], padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                    strides=strides)(layer2)
    layer3 = BatchNormalization()(layer3)
    layer3 = Activation('elu')(layer3)
    layer3 = Dropout(dropoutRate)(layer3)

    flatten = Flatten(name='flatten')(layer3)

    dense = Dense(nb_classes, name='dense')(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def EEGNet_v3(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet v3 """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)
