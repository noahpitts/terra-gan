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
    
    def __init__(self, buildArch=True, inputHeight = 256, inputWidth = 256, inputChannels = 1, outputHeight = 256, outputWidth = 256, outputChannels = 1, patchDim = [256, 256], gFilters = 32, dFilters = 32, UNET = True, gFilterConvMult = [0, 1, 2, 4, 8, 8, 8, 8, 8], gFilterDeconvMult = [0, 8, 8, 8, 8, 4, 2, 1], learningRate = 1E-4, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08, randomSeed = 2019, dataLoader = False):

        # Initialize the Model Architecture Parameters
        # --------------------------
        self.initializeParams(inputHeight, inputWidth, inputChannels,outputHeight, outputWidth, outputChannels, patchDim, gFilters, dFilters, UNET, gFilterConvMult, gFilterDeconvMult, learningRate, beta1, beta2, epsilon, randomSeed, dataLoader)

        # Build the Model Architecture
        # --------------------------
        # if (buildArch) 
        self.buildArch(summary=True)

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
        self.numPatches = int((self.outputShape[0] / self.patchDim[0]) * (self.outputShape[1] / self.patchDim[1]))
        self.patchShape = [self.patchDim[0], self.patchDim[1], self.outputShape[2]]

        # Define Optimizers
        # ------------------------
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.dOptimizer = ko.Adam(lr=self.learningRate, beta_1=self.beta1, beta_2=self.beta2, epsilon=self.epsilon)
        self.gOptimizer = ko.Adam(lr=self.learningRate, beta_1=self.beta1, beta_2=self.beta2, epsilon=self.epsilon)

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
        self.discriminator.summary()

        # Build and Compile Generator
        # ----------------------
        self.generator = self.buildGenerator()
        self.generator.compile( self.dOptimizer, loss='mean_absolute_error' )
        self.generator.summary()

        # Build and Compile DCGAN
        # ----------------------
        self.dcgan = self.buildDCGAN()
        self.dcgan.compile(self.gOptimizer, loss=['mean_absolute_error', 'binary_crossentropy'], loss_weights=[1E2, 1])
        self.dcgan.summary()

        # Log Model Summary
        # ----------------------
        # self.summary()

    def buildDCGAN(self):
        genInput = kl.Input(name='DCGAN_input', shape=self.inputShape)
        genOutput = self.generator(genInput)

        # chop the generated image into patches
        h = self.inputShape[0]
        w = self.inputShape[1]
        ph = self.patchShape[0]
        pw = self.patchShape[0]

        rowIndexList = [(i * ph, (i + 1) * ph) for i in range(int(h / ph))]
        colIndexList = [(i * pw, (i + 1) * pw) for i in range(int(w / pw))]

        genPatchList = []
        for rowIndex in rowIndexList:
            for colIndex in colIndexList:
                xPatch = kl.Lambda(lambda z: z[:, rowIndex[0]:rowIndex[1], colIndex[0]:colIndex[1], :], output_shape=self.inputShape)(genOutput)
                genPatchList.append(xPatch)

        # measure loss from patches of the image (not the actual image)
        dcganOutput = self.discriminator(genPatchList)

        # DCGAN Model
        return km.Model(name='DCGAN_model', inputs=[genInput], outputs=[genOutput, dcganOutput])

    def buildGenerator(self):
        # UNET Generator 
        # [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]

        # Convolution-BatchNorm-ReLU layer with k ﬁlters
        def conLayer(name, layerInput, numFilters, kernelSize=(4,4), bn=True):

            #  Convolutional Layer
            c = kl.Conv2D(numFilters, kernelSize,
                name=name + '_conv',
                strides=(2, 2),
                padding='same',
                kernel_initializer=self.convInitializer
                )(layerInput)

            # Batch Normalization Layer
            if (bn): c = kl.BatchNormalization(
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
            
        # Convolution-BatchNorm-Dropout-ReLUlayer with a dropout rate of 50%
        def deconLayer(name, inputLayer, skipLayer, numFilters, kernelSize=(4,4), dropout=0.5):
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
            if (dropout): d = kl.Dropout(dropout, 
                name=name + '_dropout'
                )(d)

            # Concatination (skip connections) Layer
            if (self.UNET): d = kl.Concatenate(
                name=name + '_unet', 
                axis=3
                )([d, skipLayer])

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

        #  Image input
        inputLayer = kl.Input(name='G_input', shape=self.inputShape)

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
        d1 = deconLayer('G1d', c8, c7, self.gFilters * self.gFilterDeconvMult[1])
        d2 = deconLayer('G2d', d1, c6, self.gFilters * self.gFilterDeconvMult[2])
        d3 = deconLayer('G3d', d2, c5, self.gFilters * self.gFilterDeconvMult[3])
        d4 = deconLayer('G4d', d3, c4, self.gFilters * self.gFilterDeconvMult[4], dropout=False)
        d5 = deconLayer('G5d', d4, c3, self.gFilters * self.gFilterDeconvMult[5], dropout=False)
        d6 = deconLayer('G6d', d5, c2, self.gFilters * self.gFilterDeconvMult[6], dropout=False)
        d7 = deconLayer('G7d', d6, c1, self.gFilters * self.gFilterDeconvMult[7], dropout=False)

        d8 = kl.UpSampling2D(name='G8d_upsample', size=(2, 2))(d7)

        outputLayer = kl.Conv2D(self.outputChannels, (4, 4), name='G_output', strides=(1, 1), activation='tanh', padding='same', kernel_initializer=ki.RandomNormal(0.0, 0.02))(d8)
        
        return km.Model(inputs=[inputLayer], outputs=[outputLayer])

    def buildDiscriminator(self): #?? READY TO TEST
    #  -------------------------------
     #  DISCRIMINATOR
     #  C64-C128-C256-C512-C512-C512 (for 256x256)
     #  otherwise, it scales from 64
     #  1 layer block = Conv - BN - LeakyRelu
     #  -------------------------------
        def dLayer(name, inputLayer, numFilters=32, kernelSize=4, batchNorm=True):
            d = kl.Conv2D(numFilters, kernelSize, 
                name=name + '_conv', 
                strides=(2, 2), 
                padding='same', 
                kernel_initializer = self.convInitializer
                )(inputLayer)
            
            if (batchNorm): d = kl.BatchNormalization(
                name=name + '_bnorm', 
                axis=3, 
                epsilon=1e-5, 
                momentum=0.1, 
                gamma_initializer=self.bnormInitializer
                )(d)
            
            d = kl.LeakyReLU(
                name=name+'_lrelu', 
                alpha=0.2
                )(d)

            return d

        # INPUT LAYER
        inputLayer = kl.Input(name='D_input', shape=self.patchShape)

        # NUMBER OF FILTERS
        numConv = int(np.floor(np.log(self.outputShape[1]) / np.log(2)))
        filterList = [self.dFilters * min(8, (2 ** i)) for i in range(numConv)]

        # CONV LAYER 1
        d = dLayer('D1', inputLayer, self.dFilters, 4, False)

        # CONV 2 - CONV N
        for i, filterSize in enumerate(filterList[1:]):
            name = 'D{}'.format(i + 2)
            d = dLayer(name, d, filterSize)

        # BUILD PATCH GAN
        # generate a list of inputs for the different patches to the network
        inputList = [kl.Input(shape=self.patchShape, name="PD_input_%s" % i) for i in range(self.numPatches)]

        # get an activation
        xFlat = kl.Flatten()(d)
        x = kl.Dense(2, activation='softmax', name="PD_dense")(xFlat)

        patchGAN = km.Model(inputs=[inputLayer], outputs=[x, xFlat], name="Patch_Discriminator_Model")

        # generate individual losses for each patch
        x = [patchGAN(patch)[0] for patch in inputList]
        xMbd = [patchGAN(patch)[1] for patch in inputList]

        # merge layers if have multiple patches (aka perceptual loss)
        if len(x) > 1:
            x = kl.Concatenate(name="merged_features")(x)
        else:
            x = x[0]

        # merge mbd if needed
        if len(xMbd) > 1:
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

        discriminator = km.Model(name='D_model', inputs=inputList, outputs=[x])

        return discriminator

    # TRAINING
    # --------------------------

    def loadData(self):
        # TODO ----------------
        x = 2

    def train(self):
        # TODO ----------------
        x = 2

    def trainEpoch(self):
        # TODO ----------------
        x = 2

    def trainBatch(self):
        # TODO ----------------
        x = 2

    def trainDisriminator(self):
        # TODO ----------------
        x = 2

    def trainGenerator(self):
        # TODO ----------------
        x = 2
        

    # TESTING
    # --------------------------

    # INFERENCE
    # --------------------------

    # UTILS
    # --------------------------
    def summary(self):
        # TODO ----------------
        x = 2

    def saveModel(self):
        # TODO ----------------
        x = 2

    def exportModel(self):
        # TODO ----------------
        x = 2

    # DEPRICATED
    # --------------------------

terraGAN = TerraGAN()
