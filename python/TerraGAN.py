import keras.layers as kl   # Keras Layers
import keras.models as km  # Keras Models
import keras.initializers as ki
import keras.backend as K   # Keras Backend
import numpy as np          # Numpy


class DataLoader:
    def __init__(self, params):
        # TODO ----------------


class TerraGAN:
    def __init__(self, params):
        # TODO ----------------

        self.patchShape = [64, 64, 3]
        self.outputShape = [1]
        self.inputShape = [2]

        self.dFilters = 32
        self.numPatches = 8

# BUILD
# --------------------------

    def buildGAN():
        # TODO ----------------

    def buildGenerator(self):
        UNET = True
        def conLayer(name, layerInput, numFilters, kernelSize = 4, bn=True){

            #  Convolutional Layer
            c = kl.Conv2D(numFilters, kernelSize,
                name=name + '_conv',
                strides=(2, 2),
                padding='same',
                kernel_initializer=ki.RandomNormal(0.0, 0.02)
                )(layerInput)

            # Batch Normalization Layer
            if (bn) c = kl.BatchNormalization(
                name=name + '_bnorm',
                axis=3,
                epsilon=1e-5,
                momentum=0.1,
                gamma_initializer=ki.RandomNormal(1.0, 0.02)
                )(c)  #?? Should I turn on training here

            # Leaky ReLU Layer
            c = kl.LeakyReLU(
                name=name + '_lrelu',
                alpha=0.2
                )(c)

            return c
            
        def deconLayer(name, inputLayer, skipLayer, numFilters, kernelSize=4, dropoutRate=0.5)

            d = kl.UpSampling2D(name=name + '_upsample', size=(2, 2))(inputLayer)
            
            d = kl.Conv2D(numFilters, kernelSize, name=name + '_conv', strides=(1, 1), padding='same', kernel_initializer = ki.RandomNormal(0.0, 0.02))(d)

            d = kl.BatchNormalization(name=name + '_bnorm', axis=3, epsilon=1e-5, momentum=0.1, gamma_initializer=ki.RandomNormal(1.0, 0.02))(d)

            if (dropoutRate) d = kl.Dropout(dropoutRate, name=name + '_dropout')(d)

            if (UNET) d = kl.Concatenate(name=name + '_unet', axis=3)([d, skipLayer])(d)

            d = kl.ReLU(name=name + '_relu')(d)

            return d

            
        kernel_initializer = ki.RandomNormal(0.0, 0.02)
        gamma_initializer = ki.RandomNormal(1.0, 0.02)

        #  -------------------------------
        #  ENCODER
        #  C64-C128-C256-C512-C512-C512-C512-C512
        #  1 layer block = Conv - BN - LeakyRelu
        #  -------------------------------

        inputLayer = kl.InputLayer(name='G_input', input_shape=self.inputShape)

        c1 = conLayer('G1c', inputLayer, self.gFilters, 4, False)
        c2 = conLayer('G2c', c1, self.gFilters * 2)
        c3 = conLayer('G3c', c2, self.gFilters * 4)
        c4 = conLayer('G4c', c3, self.gFilters * 8)
        c5 = conLayer('G5c', c4, self.gFilters * 8)
        c6 = conLayer('G6c', c5, self.gFilters * 8)
        c7 = conLayer('G7c', c6, self.gFilters * 8)
        c8 = conLayer('G8c', c7, self.gFilters * 8)

        # -------------------------------
        #  DECODER
        #  CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128 
        #  1 layer block = Conv - Upsample - BN - DO - Relu
        #  also adds skip connections (Concatenate). Takes input from previous layer matching encoder layer
        # -------------------------------
        d1 = deconLayer('G1d', c8, c7 self.gFilters * 8) #?? ADD DROPOUT SWITCH OFF
        d2 = deconLayer('G2d', d1, c6 self.gFilters * 8)
        d3 = deconLayer('G3d', d2, c5 self.gFilters * 8)
        d4 = deconLayer('G4d', d3, c5 self.gFilters * 8, dropout=False)
        d5 = deconLayer('G5d', d4, c3 self.gFilters * 4, dropout=False)
        d6 = deconLayer('G6d', d5, c2 self.gFilters * 2, dropout=False)
        d7 = deconLayer('G6d', d6, c1 self.gFilters * 0, dropout=False)

        d8 = kl.UpSampling2D(name='G7d_upsample', size - (2, 2))(d7)

        outputLayer = kl.Conv2D(this.outputShape[2], (4, 4), name='G_output', strides=(1, 1), activation='tanh', padding='same', kernel_initializer=ki.RandomNormal(0.0, 0.02))(d8)
        
        return km.Model(input=inputLayer, output=outputLayer)




        # TODO ----------------

    def buildDiscriminator(self): #?? READY TO TEST
    #  -------------------------------
     #  DISCRIMINATOR
     #  C64-C128-C256-C512-C512-C512 (for 256x256)
     #  otherwise, it scales from 64
     #  1 layer block = Conv - BN - LeakyRelu
     #  -------------------------------
        def dLayer(name, inputLayer, numFilters=32, kernelSize=4, batchNorm=True):
            d = kl.Conv2D(numFilters, kernelSize, name=name + '_conv', strides=(2, 2), padding='same', kernel_initializer = ki.RandomNormal(0.0, 0.02))(inputLayer)  #??
            
            if (bn) d = kl.BatchNormalization(name=name + '_bnorm', axis=3, epsilon=1e-5, momentum=0.1, gamma_initializer=ki.RandomNormal(1.0, 0.02))(d)  #??
            
            d = kl.LeakyReLU(name=name+'_lrelu', alpha=0.2)(d)
            return d

        # INPUT LAYER
        inputLayer = kl.InputLayer(name='D_input', input_shape=self.patchShape)

        # NUMBER OF FILTERS
        numConv = int(np.floor(np.log(self.outputShape[1]) / np.log(2)))
        filterList = [self.dFilters * min(8, (2 ** i)) for i in range(numConv)]

        # CONV LAYER 1
        d = dLayer('D1', inputLayer, self.dFilters 4, False)(inputLayer)

        # CONV 2 - CONV N
        for i, filterSize in enumerate(filtersList[1:]):
            name = 'D{}'.format(i + 2)
            d = dLayer(name, d, filterSize)(d)

        # BUILD PATCH GAN
        # generate a list of inputs for the different patches to the network
        inputList = [kl.InputLayer(shape=self.patchShape, name="PD_input_%s" % i) for i in range(self.numPatches)]

        # get an activation
        xFlat = kl.Flatten()(d)
        x = kl.Dense(2, activation='softmax', name="PD_dense")(xFlat)

        patchGAN = km.Model(input=[inputLayer], output=[x, xFlat], name="Patch_Discriminator_Model")

        # generate individual losses for each patch
        x = [patchGAN(patch)[0] for patch in inputList]
        xMbd = [patchGAN(patch)[1] for patch in inputList]

        # merge layers if have multiple patches (aka perceptual loss)
        if len(x) > 1:
            x = kl.Concatenate(name="merged_features")(x)
        else:
            x = x[0]

        # merge mbd if needed
        if len(x_mbd) > 1:
            xMbd = kl.Concatenate(name="merged_features_mbd")(xMbd)
        else:
            xMbd = xMbd[0]

        numKernels = 100
        dimPerKernel = 5

        def mbd(x):
            diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
            absDiffs = K.sum(K.abs(diffs), 2)
            x = K.sum(K.exp(-absDiffs), 2)
            return x

        def lambdaOutput(inputShape):
            return inputShape[:2]

        xMbd = kl.Dense(numKernels*dimPerKernel, use_bias=False)(xMbd)
        xMbd = kl.Reshape([numKernels, dimPerKernel])(xMbd)
        xMbd = kl.Lambda(mbd, output_shape=lambdaOutput)(xMbd)
        x = kl.Concatenate()([x, xMbd])
        x = kl.Dense(2, name='D_output', activation='softmax')(x)

        discriminator = km.Model(input=inputList, output=[x])
        return discriminator

# TRAINING
# --------------------------

    def loadData():
        # TODO ----------------

    def train():
        # TODO ----------------

    def trainEpoch():
        # TODO ----------------

    def trainBatch():
        # TODO ----------------

    def trainDisriminator():
        # TODO ----------------

    def trainGenerator():
        # TODO ----------------

# TESTING
# --------------------------

# INFERENCE
# --------------------------

# UTILS
# --------------------------

    def saveModel():
        # TODO ----------------

    def exportModel():
        # TODO ----------------

# DEPRICATED
# --------------------------
