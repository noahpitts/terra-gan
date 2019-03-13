import keras.layers as kl           # Keras Layers
import keras.models as km           # Keras Models
import keras.initializers as ki     # Keras Initializers
import keras.optimizers as ko       # Keras Optimizers
import keras.backend as K           # Keras Backend
import numpy as np                  # Numpy


class DataLoader:
    def __init__(self, params):
        # TODO ----------------
        x = 2 


class TerraGAN:
    def __init__(self, buildArch=True,
        inputHeight = 256, inputWidth = 256, inputChannels = 1,
        outputHeight = 256, outputWidth = 256, outputChannels = 1,
        patchDim = [64, 64], gFilters = 32, dFilters = 32, UNET = True,
        gFilterConvMult = [0, 1, 2, 4, 8, 8, 8, 8, 8],
        gFilterDeconvMult = [0, 8, 8, 8, 8, 4, 2, 0],
        learningRate = 1E-4, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08,
        randomSeed = 2019, dataLoader = False)

        # Initialize the Model Architecture Parameters
        # --------------------------
        initializeParams(inputHeight, inputWidth, inputChannels,outputHeight, outputWidth, outputChannels, patchDim, gFilters, dFilters, UNET, gFilterConvMult, gFilterDeconvMult, learningRate, beta1, beta2, epsilon, randomSeed, dataLoader)

        # Build the Model Architecture
        # --------------------------
        if (buildArch) buildArchitecture(summary=True)

# BUILD MODEL
# --------------------------
    def initializeParams(self, inputHeight, inputWidth, inputChannels,outputHeight, outputWidth, outputChannels, patchDim, gFilters, dFilters, UNET, gFilterConvMult, gFilterDeconvMult, learningRate, beta1, beta2, epsilon, randomSeed, dataLoader):
        # Input shape [rows, columns, channels]
        # ------------------------
        self.inputHeight = inputHeight
        self.inputWidth = inputWidth
        self.inputChannels = inputChannels
        self.inputShape = [inputHeight, inputWidth, inputChannels]

        # Output shape [rows, columns, channels]
        # ------------------------
        self.outputHeight = outputHeight
        self.outputWidth = outputWidth
        self.outputChannels = outputChannels
        self.outputShape = [outputHeight, outputWidth, outputChannels]

        # Number of Generator and Discriminator Filters
        # ------------------------
        self.UNET = UNET
        self.gFilters = gFilters
        self.gFilterConvMult = gFilterConvMult
        self.gFilterDeconvMult = gFilterDeconvMult
        self.dFilters = dFilters
        

        # Discriminator Patches [rows, cols]
        # ------------------------
        self.patchDim = patchDim
        self.numPatches = (self.outputShape[0] / self.patchDim[0]) * (self.outputShape[1] / self.patchDim[1])
        self.patchShape = [self.patchDim[0], self.patchDim[1], self.outputShape[2]]

        # Define Optimizers
        # ------------------------
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.dOptimizer = ko.Adam(lr=self.learningRate, beta_1=self.beta1, beta_2=slef.beta2, epsilon=self.epsilon)
        self.gOptimizer = ko.Adam(lr=self.learningRate, beta_1=self.beta1, beta_2=slef.beta2, epsilon=self.epsilon)

        # Define Initializers
        # ------------------------
        self.randomSeed = randomSeed
        self.convInitializer = ki.RandomNormal(0.0, 0.02, self.randomSeed)
        self.bnormInitializer = ki.RandomNormal(1.0, 0.02, self.randomSeed)

    def buildArch(self, summary=False):
        # Build and Compile Discriminator
        # ----------------------
        self.discriminator = self.buildDiscriminator()
        self.discriminator.compile(self.dOptimizer, loss='binary_crossentropy')
        self.discriminator.trainable = False

        # Build and Compile Generator
        # ----------------------
        self.generator = self.buildGenerator()
        self.generator.compile( self.dOptimizer, loss='mean_absolute_error' )

        # Build and Compile DCGAN
        # ----------------------
        self.dcgan = self.buildDCGAN()
        self.dcgan.compile(self.gOptimizer, loss=['mean_absolute_error', 'binary_crossentropy'], loss_weights=[1E2, 1])

        # Log Model Summary
        # ----------------------
        if (summary) self.summary()

    def buildDCGAN():
        # TODO - Comment
        genInput = kl.InputLayer(name='DCGAN_input', input_shape=self.inputShape)
        genOutput = self.generator(genInput)












        # TODO ----------------
        return km.Model()

    def buildGenerator(self):

        def conLayer(name, layerInput, numFilters, kernelSize = 4, bn=True){

            #  Convolutional Layer
            c = kl.Conv2D(numFilters, kernelSize,
                name=name + '_conv',
                strides=(2, 2),
                padding='same',
                kernel_initializer=self.convInitializer
                )(layerInput)

            # Batch Normalization Layer
            if (bn) c = kl.BatchNormalization(
                name=name + '_bnorm',
                axis=3,
                epsilon=1e-5,
                momentum=0.1,
                gamma_initializer=self.bnormInitializer
                )(c)  #?? Should I turn on training here

            # Leaky ReLU Layer
            c = kl.LeakyReLU(
                name=name + '_lrelu',
                alpha=0.2
                )(c)

            return c
            
        def deconLayer(name, inputLayer, skipLayer, numFilters, kernelSize=4, dropoutRate=0.5)
            # Upsampling Layer
            d = kl.UpSampling2D(
                name=name + '_upsample', 
                size=(2, 2)
                )(inputLayer)
            
            # Convolutional Layer
            d = kl.Conv2D(numFilters, kernelSize, 
                name=name + '_conv', 
                strides=(1, 1), 
                padding='same', 
                kernel_initializer = self.convInitializer
                )(d)

            # Batch Normalization Layer
            d = kl.BatchNormalization(
                name=name + '_bnorm', 
                axis=3, 
                epsilon=1e-5, 
                momentum=0.1, 
                gamma_initializer=self.bnormInitializer
                )(d)

            # Dropout Layer
            if (dropoutRate) d = kl.Dropout(dropoutRate, 
                name=name + '_dropout'
                )(d)

            # Concatination (skip connections) Layer
            if (self.UNET) d = kl.Concatenate(
                name=name + '_unet', 
                axis=3
                )([d, skipLayer])(d)

            # Leaky ReLU Layer
            d = kl.ReLU(
                name=name + '_relu'
                )(d)

            return d

        #  -------------------------------
        #  ENCODER
        #  C64-C128-C256-C512-C512-C512-C512-C512
        #  1 layer block = Conv - BN - LeakyRelu
        #  -------------------------------

        inputLayer = kl.InputLayer(name='G_input', input_shape=self.inputShape)

        c1 = conLayer('G1c', inputLayer, self.gFilters * self.gFilterConvMult[1], 4, False)
        c2 = conLayer('G2c', c1, self.gFilters * self.gFilterConvMult[2])
        c3 = conLayer('G3c', c2, self.gFilters * self.gFilterConvMult[3])
        c4 = conLayer('G4c', c3, self.gFilters * self.gFilterConvMult[4])
        c5 = conLayer('G5c', c4, self.gFilters * self.gFilterConvMult[5])
        c6 = conLayer('G6c', c5, self.gFilters * self.gFilterConvMult[6])
        c7 = conLayer('G7c', c6, self.gFilters * self.gFilterConvMult[7])
        c8 = conLayer('G8c', c7, self.gFilters * self.gFilterConvMult[8])

        # -------------------------------
        #  DECODER
        #  CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128 
        #  1 layer block = Conv - Upsample - BN - DO - Relu
        #  also adds skip connections (Concatenate). Takes input from previous layer matching encoder layer
        # -------------------------------
        d1 = deconLayer('G1d', c8, c7 self.gFilters * self.gFilterDeconvMult[1]) #?? ADD DROPOUT SWITCH OFF
        d2 = deconLayer('G2d', d1, c6 self.gFilters * self.gFilterDeconvMult[2])
        d3 = deconLayer('G3d', d2, c5 self.gFilters * self.gFilterDeconvMult[3])
        d4 = deconLayer('G4d', d3, c5 self.gFilters * self.gFilterDeconvMult[4], dropout=False)
        d5 = deconLayer('G5d', d4, c3 self.gFilters * self.gFilterDeconvMult[5], dropout=False)
        d6 = deconLayer('G6d', d5, c2 self.gFilters * self.gFilterDeconvMult[6], dropout=False)
        d7 = deconLayer('G6d', d6, c1 self.gFilters * self.gFilterDeconvMult[7], dropout=False)

        d8 = kl.UpSampling2D(name='G7d_upsample', size - (2, 2))(d7)

        outputLayer = kl.Conv2D(this.outputShape[2], (4, 4), name='G_output', strides=(1, 1), activation='tanh', padding='same', kernel_initializer=ki.RandomNormal(0.0, 0.02))(d8)
        
        return km.Model(input=inputLayer, output=outputLayer)

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
