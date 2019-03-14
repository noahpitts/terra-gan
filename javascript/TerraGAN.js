const tf = require('@tensorflow/tfjs');
const Jimp = require('jimp');

// Terra GAN Model Builder
class TerraGAN {
    constructor(params, buildArch=true) {
        this.initializeParams(params);
        if (buildArch) this.buildArch(true);
    }

    // BUILD
    //---------------------

    initializeParams(params) {

        // Default Params
        // ------------------------
        const defaultParams = {
            inputHeight: 256,
            inputWidth: 256,
            inputChannels: 1,
            outputHeight: 256,
            outputWidth: 256,
            outputChannels: 1,
            patchDim: [256, 256],
            gFilters: 32,
            dFilters: 32,
            UNET: true,
            gFilterConvMult: [0, 1, 2, 4, 8, 8, 8, 8, 8],
            gFilterDeconvMult: [0, 8, 8, 8, 8, 4, 2, 1],
            learningRate: 1E-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-08,
            randomSeed: 2019,
            dataLoader: false
        };

        // Input shape [rows, columns, channels]
        // ------------------------
        this.inputHeight = (params.inputHeight !== undefined) ? params.inputHeight : defaultParams.inputHeight;
        this.inputWidth = (params.inputWidth !== undefined) ? params.inputWidth : defaultParams.inputWidth;
        this.inputChannels = (params.inputChannels !== undefined) ? params.inputChannels : defaultParams.inputChannels;
        this.inputShape = (!params.inputShape !== undefined) ? [this.inputHeight, this.inputWidth, this.inputChannels] : params.inputShape;
        this.inputChannels = this.inputShape[2];

        // Output shape [rows, columns, channels]
        // ------------------------
        this.outputHeight = (params.outputHeight !== undefined) ? params.outputHeight : defaultParams.outputHeight;
        this.outputWidth = (params.outputWidth !== undefined) ? params.outputWidth : defaultParams.outputWidth;
        this.outputChannels = (params.outputChannels !== undefined) ? params.outputChannels : defaultParams.outputChannels;
        this.outputShape = (!params.outputShape !== undefined) ? [this.outputHeight, this.outputWidth, this.outputChannels] : params.outputShape;
        this.outputChannels = this.outputShape[2];

        // Number of Generator and Discriminator Filters
        // ------------------------
        this.UNET = true;
        this.gFilters = (params.gFilters !== undefined) ? params.gFilters : defaultParams.gFilters;
        this.gFilterConvMult = (params.gFilterConvMult !== undefined) ? params.gFilterConvMult : defaultParams.gFilterConvMult;
        this.gFilterDeconvMult = (params.gFilterDeconvMult !== undefined) ? defaultParams.gFilterDeconvMult : defaultParams.gFilterDeconvMult;
        this.dFilters = (params.dFilters !== undefined) ? params.dFilters : defaultParams.dFilters;

        // Discriminator Patches [rows, cols]
        // ------------------------
        this.patchDim = (params.patchDim !== undefined) ? params.patchDim : defaultParams.patchDim;
        this.numPatches = (this.outputShape[0] / this.patchDim[0]) * (this.outputShape[1] / this.patchDim[1]);
        this.patchShape = [this.patchDim[0], this.patchDim[1], this.outputShape[2]];

        // Define Optimizers
        // ------------------------
        this.learningRate = (params.learningRate !== undefined) ? params.learningRate : defaultParams.learningRate;
        this.beta1 = (params.beta1 !== undefined) ? params.beta1 : defaultParams.beta1;
        this.beta2 = (params.beta2 !== undefined) ? params.beta2 : defaultParams.beta2;
        this.epsilon = (params.epsilon !== undefined) ? params.epsilon : defaultParams.epsilon;
        this.gOptimizer = tf.train.adam(this.learningRate, this.beta1, this.beta2, this.epsilon);
        this.dOptimizer = tf.train.adam(this.learningRate, this.beta1, this.beta2, this.epsilon);

        // Define Initializers
        // ------------------------
        // TODO - FIX THIS FOR PARAM CONTROL
        this.randomSeed = (params.randsomSeed !== undefined) ? params.randsomSeed : defaultParams.randsomSeed;
        this.convInitializer = tf.initializers.randomNormal({mean: 0.0, stddev: 0.02});
        this.bnormInitializer = tf.initializers.randomNormal({mean: 1.0, stddev: 0.02});

    }

    buildArch(summary=false) {
        // Build and Compile Discriminator
        // ----------------------
        this.discriminator = this.buildDiscriminator();
        this.discriminator.compile({ optimizer: this.dOptimizer, loss: 'binaryCrossentropy' });
        this.discriminator.trainable = false;

        // Build and Compile Generator
        // ----------------------
        this.generator = this.buildGenerator();
        this.generator.compile({ optimizer: this.dOptimizer, loss: 'meanAbsoluteError' });

        // Build and Compile DCGAN
        // ----------------------
        this.dcgan = this.buildDCGAN();
        this.dcgan.compile({ optimizer: this.gOptimizer, loss: ['meanAbsoluteError', 'binaryCrossentropy'], lossWeights: [1E2, 1] })

        // Log Model Summary
        // ----------------------
        if (summary) this.summary();
    }

    buildDCGAN() {
        const genInput = tf.input({ shape: this.inputShape, name: 'DCGAN_input' });
        const genOutput = this.generator.apply(genInput);

        const h = this.inputShape[0];
        const w = this.inputShape[1];
        const ph = this.patchShape[0];
        const pw = this.patchShape[1];

        // chop the generated image into patches
        const rowIndexList = [];
        for (let i = 0; i < Math.floor(h / ph); i++) {
            rowIndexList.push(i * ph);
        };

        const colIndexList = [];
        for (let i = 0; i < Math.floor(w / pw); i++) {
            colIndexList.push(i * pw);
        };

        const genPatchList = [];
        for (let rowIndex of rowIndexList) {
            for (let colIndex of colIndexList) {
                // TODO - Update Slice when tfjs reslease support for Lambda Layers
                let SLC = new Slice({ slice: [rowIndex, colIndex, ph, pw] });
                let xPatch = SLC.apply(genOutput);
                genPatchList.push(xPatch);
            }
        }

        // measure loss from patches of the image (not the actual image)
        const dcganOutput = this.discriminator.apply(genPatchList);

        // DCGAN Model
        return tf.model({name: 'DCGAN_model', inputs: genInput, outputs: [genOutput, dcganOutput] });
    }

    buildGenerator() {
        // UNET Generator 
        // [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]

        // Convolution-BatchNorm-ReLU layer with k ﬁlters
        function convLayer(name, inputLayer, numFilters, kernelSize = [4, 4], bn = true) {

            // Convolutional Layer
            let c = tf.layers.conv2d({
                name: name + '_conv',
                filters: numFilters,
                kernelSize: kernelSize,
                strides: [2, 2],
                padding: 'same',
                kernelInitializer: tf.initializers.randomNormal({mean: 0.0, stddev: 0.02})
                
            }).apply(inputLayer);

            // Batch Normalization Layer
            if (bn) c = tf.layers.batchNormalization({
                name: name + '_bnorm',
                axis: 3,
                epsilon: 1e-5,
                momentum: 0.1,
                gammaInitializer: tf.initializers.randomNormal({mean: 1.0, stddev: 0.02})
            }).apply(c);

            // Leaky ReLU Layer
            c = tf.layers.leakyReLU({
                name: name + '_lrelu',
                alpha: 0.2
            }).apply(c);

            return c;
        }

        // Convolution-BatchNorm-Dropout-ReLUlayer with a dropout rate of 50%
        function deconvLayer(name, inputLayer, skipLayer, numFilters, dropout_rate = 0.5, kernelSize = [4, 4]) {
            // Upsampling Layer
            let d = tf.layers.upSampling2d({
                name: name + '_upsample',
                size: [2, 2]
            }).apply(inputLayer);

            // Convolutional Layer
            d = tf.layers.conv2d({
                name: name + '_conv',
                filters: numFilters,
                kernelSize: kernelSize,
                strides: [1, 1],
                padding: 'same',
                kernelInitializer: tf.initializers.randomNormal({mean: 0.0, stddev: 0.02})
            }).apply(d);

            // Batch Normalization Layer
            d = tf.layers.batchNormalization({
                name: name + '_bnorm',
                axis: 3,
                epsilon: 1e-5,
                momentum: 0.1,
                gammaInitializer: tf.initializers.randomNormal({mean: 1.0, stddev: 0.02})
            }).apply(d);

            // Dropout Layer
            if (dropout_rate) d = tf.layers.dropout({
                name: name + '_dropout',
                rate: dropout_rate
            }).apply(d);

            // Concatination (skip connections) Layer
            // if (this.UNET) 
            d = tf.layers.concatenate({
                name: name + '_unet',
                axis: 3
            }).apply([d, skipLayer]);

            // Leaky ReLU Layer
            d = tf.layers.reLU({
                name: name + '_relu'
            }).apply(d);

            return d;
        }

        // -------------------------------
        // ENCODER
        // C64-C128-C256-C512-C512-C512-C512-C512
        // 1 layer block = Conv - BN - LeakyRelu
        // -------------------------------

        // Image input
        const inputLayer = tf.input({ shape: this.inputShape, name: 'G_input' });

        const c1 = convLayer('G1c', inputLayer, this.gFilters * this.gFilterConvMult[1], [4, 4], false);     // default: C64
        const c2 = convLayer('G2c', c1, this.gFilters * this.gFilterConvMult[2]);                  // default: C128
        const c3 = convLayer('G3c', c2, this.gFilters * this.gFilterConvMult[3]);                  // default: C256
        const c4 = convLayer('G4c', c3, this.gFilters * this.gFilterConvMult[4]);                  // default: C512
        const c5 = convLayer('G5c', c4, this.gFilters * this.gFilterConvMult[5]);                  // default: C512
        const c6 = convLayer('G6c', c5, this.gFilters * this.gFilterConvMult[6]);                  // default: C512
        const c7 = convLayer('G7c', c6, this.gFilters * this.gFilterConvMult[7]);                  // default: C512
        const c8 = convLayer('G8c', c7, this.gFilters * this.gFilterConvMult[8]);                  // default: C512

        //-------------------------------
        // DECODER
        // CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128 
        // 1 layer block = Conv - Upsample - BN - DO - Relu
        // also adds skip connections (Concatenate). Takes input from previous layer matching encoder layer
        //-------------------------------

        const d1 = deconvLayer('G1d', c8, c7, this.gFilters * this.gFilterDeconvMult[1]);                     // default: C512
        const d2 = deconvLayer('G2d', d1, c6, this.gFilters * this.gFilterDeconvMult[2]);                     // default: C512
        const d3 = deconvLayer('G3d', d2, c5, this.gFilters * this.gFilterDeconvMult[3]);                     // default: C512
        const d4 = deconvLayer('G4d', d3, c4, this.gFilters * this.gFilterDeconvMult[4], false);      // default: C512
        const d5 = deconvLayer('G5d', d4, c3, this.gFilters * this.gFilterDeconvMult[5], false);      // default: C256
        const d6 = deconvLayer('G6d', d5, c2, this.gFilters * this.gFilterDeconvMult[6], false);      // default: C128
        const d7 = deconvLayer('G7d', d6, c1, this.gFilters * this.gFilterDeconvMult[7], false);      // default: C64

        const d8 = tf.layers.upSampling2d({
            name: 'G8d_upsample',
            size: [2, 2]
        }).apply(d7);


        const outputLayer = tf.layers.conv2d({
            name: 'G_output',
            kernelSize: [4,4],
            filters: this.outputChannels,
            strides: [1, 1],
            activation: 'tanh',
            padding: 'same',
            kernelInitializer: tf.initializers.randomNormal({mean: 0.0, stddev: 0.02})
        }).apply(d8);

        // return Model(input, output)
        return tf.model({name: 'G_model', inputs: inputLayer, outputs: outputLayer });
    }

    buildDiscriminator() {

        // -------------------------------
        // DISCRIMINATOR
        // C64-C128-C256-C512-C512-C512 (for 256x256)
        // otherwise, it scales from 64
        // 1 layer block = Conv - BN - LeakyRelu
        // -------------------------------
        function dLayer(name, inputLayer, numFilters, kernelSize = [4, 4], bn = true) {
            let d = tf.layers.conv2d({
                name: name + '_conv',
                kernelSize: kernelSize,
                filters: numFilters,
                strides: [2, 2],
                padding: 'same',
                kernelInitializer: tf.initializers.randomNormal({mean: 0.0, stddev: 0.02})
            }).apply(inputLayer);

            if (bn) d = tf.layers.batchNormalization({
                name: name + '_bnorm',
                axis: 3,
                epsilon: 1e-5,
                momentum: 0.1,
                gammaInitializer: tf.initializers.randomNormal({mean: 1.0, stddev: 0.02})
            }).apply(d);

            d = tf.layers.leakyReLU({
                name: name + '_lrelu',
                alpha: 0.2
            }).apply(d);

            return d;
        }

        //  INPUT LAYER
        const inputLayer = tf.input({ name: 'D_input', shape: this.patchShape });

        const numConv = Math.floor(Math.log(this.outputShape[1] / Math.log(2)));
        const filterList = [];
        for (let i = 0; i < numConv; i++) {
            filterList.push(this.dFilters * Math.min(8, Math.pow(2, i)));
        }

        // CONV 1
        let d = dLayer('D1', inputLayer, this.dFilters, [4, 4], false);

        // CONV 2 - CONV N
        for (let i = 1; i < filterList.length; i++) {
            d = dLayer('D' + i + 1, d, filterList[i]);
        }

        // generate a list of inputs for the different patches to the network
        const inputList = [];
        for (let i = 0; i < this.numPatches; i++) {
            inputList.push(tf.input({ shape: this.patchShape, name: 'pD_input_' + i }));
        }

        // get an activation
        const xFlat = tf.layers.flatten().apply(d);
        let xx = tf.layers.dense({ units: 2, activation: 'softmax', name: 'D_dense' }).apply(xFlat);

        const patchGan = tf.model({ inputs: inputLayer, outputs: [xx, xFlat], name: 'patch_GAN' });

        // generate individual losses for each patch
        let x = [];
        let xMbd = [];
        for (let patch of inputList) {
            x.push(patchGan.apply(patch)[0]); //??
            xMbd.push(patchGan.apply(patch)[1]) //??
        }

        // merge layers if have multiple patches (aka perceptual loss)
        if (x.length > 1) {
            x = tf.layers.concatenate({ name: 'merged_features' }).apply(x);
        } else {
            x = x[0];
        }

        // merge mbd (mini batch discrimination) if needed -- https://arxiv.org/pdf/1606.03498.pdf
        if (xMbd.length > 1) {
            xMbd = tf.layers.concatenate({ name: 'merged_feature_mbd' }).apply(xMbd);
        } else {
            xMbd = xMbd[0];
        }

        const num_kernels = 100
        const dim_per_kernel = 5
        xMbd = tf.layers.dense({ units: num_kernels * dim_per_kernel, useBias: false, /*activation: 'none'*/ }).apply(xMbd);

        xMbd = tf.layers.reshape({ targetShape: [num_kernels, dim_per_kernel] }).apply(xMbd);

        // TODO - Replace MiniBatchDisc with a Lambda Layer in tfjs once implemented
        const MBD = new MiniBatchDisc({ name: 'mini_batch_discriminator' });
        xMbd = MBD.apply(xMbd);
        x = tf.layers.concatenate(1).apply([x, xMbd]);

        x = tf.layers.dense({ units: 2, activation: 'softmax', name: 'D_output' }).apply(x)
        const discriminator = tf.model({ name: 'D_model', inputs: inputList, outputs: x });

        return discriminator;
    }

    // TRAINING
    //---------------------

    loadData(params) {
        this.dataLoader = new DataLoader(params);
    }

    async train(params) {
        this.batchCounter = 0;

        const epochs = (params.epochs !== undefined) ? params.epochs : 100;
        const bpe = (params.batchesPerEpoch !== undefined) ? params.batchesPerEpoch : 1;
        const batchSize = (params.batchSize !== undefined) ? params.batchSize : [1, 1];

        const logOnBatch = (params.logOnBatch !== undefined) ? params.logOnBatch : true;
        const logOnEpoch = (params.logOnEpoch !== undefined) ? params.logOnEpoch : true;
        const saveOnEpoch = (params.saveOnEpoch !== undefined) ? params.saveOnEpoch : 10;

        const epochAvgStats = [];

        for (let e = 1; e <= epochs; e++) {
            let avgStats = await this.trainEpoch(bpe, batchSize, [logOnEpoch, logOnBatch]);

            epochAvgStats.push(avgStats);

            if (e % saveOnEpoch === 0) {
                await this.saveModel(e);
            }
        }
        // TODO Add Other shit to do per epoch
        // Print summary
        // Save file
    }

    async saveModel(e = 0) {

        const fil = this.dFilters + this.gFilters + '_'
        const modelName = this.modelName + fil + this.modelTag
        const savePath = 'file://' + params.modelDirectory + '/node/' + modelName;

        let saveResultsDis = await this.discriminator.save(savePath + '/model_dis');
        let saveResultsGen = await this.generator.save(savePath + '/model_gen_' + e);
        let saveResultsGan = await this.dcgan.save(savePath + '/model_gan');

        console.log('MODEL SAVED TO: ' + savePath);
    }

    async trainEpoch(bpe, batchSize, log) {

        const epochStats = [];
        for (let b = 1; b <= bpe; b++) {
            let stats = await this.trainBatch(batchSize[0], batchSize[1], log[1]);
            epochStats.push(stats);
        }

        const epochAvgStats = [0, 0, 0, 0];
        let b = 0;
        for (let i = 0; i < epochAvgStats.length; i++) {
            for (b = 0; b < epochStats.length; b++) {
                epochAvgStats[i] = epochAvgStats[i] + epochStats[b][i];
            }
            epochAvgStats[i] = epochAvgStats[i] / b;
        }

        if (log[0]) console.log('EPOCH FINISHED...'); //TODO FINISH EPOCH LOG

        return epochAvgStats;
    }

    async trainBatch(discBatchSize, genBatchSize, logBatchStats) {
        // TODO: Consider clocking the batch
        //??  start_time = datetime.datetime.now()
        // this.batchStart = 

        let discLoss, genLoss;

        // Train the discriminator on the Batch and return the Loss
        discLoss = await this.trainDisBatch(discBatchSize);

        // Train the discriminator on the Batch and return the Loss
        genLoss = await this.trainGenBatch(genBatchSize);

        // Increment the batch Counter
        this.batchCounter++;

        //TODO - log batch time
        let loss = [discLoss, Math.min(genLoss[0]), Math.min(genLoss[1]), Math.min(genLoss[2])];

        if (logBatchStats) console.log('[D logloss: ' + loss[0] + '] [G total: ' + loss[1] + ' L1 (mae): ' + loss[2] + ' logloss: ' + loss[3] + ']');

        return loss;
    }

    async trainDisBatch(batchSize) {
        // TODO ->>> USE TF TIDY instead of disposeing all manually
        // generate a batch of data and feed to the discriminator
        // some images that come out of here are real and some are fake

        // Tensors to dispose of later
        let inputA, inputB, inputD, result0, result1, patches, result;

        // Get the shape of the Input/Output Data
        const shapeA = [batchSize, this.dataLoader.height, this.dataLoader.width, this.dataLoader.numChannelsA];
        const shapeB = [batchSize, this.dataLoader.height, this.dataLoader.width, this.dataLoader.numChannelsB];

        // Create a batch to train the Discriminator
        let batch = await this.dataLoader.loadBatch(batchSize);

        // Discriminator Input 4D Tensors
        inputA = tf.tensor4d(batch.A, shapeA);    // Truth
        inputB = tf.tensor4d(batch.B, shapeB);    // Representation


        //?? Consider make this a random selection of trasining on real vs fake images ?????
        if (this.batchCounter % 2 === 0) {

            // generate fake image
            inputD = this.generator.predictOnBatch(inputB);

            // each image will produce a 1x2 vector for the results (aka is fake or not)
            result0 = tf.ones([inputD.shape[0], 1]);
            result1 = tf.zeros([inputD.shape[0], 1]);



            //?? OPTIONAL IMPELEMENT LABEL - FLIPPING

            // label_flipping = 0 # some val [0.0 - 1.0)
            // if label_flipping > 0:
            // p = np.random.binomial(1, label_flipping)
            // if p > 0:
            //     y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

        } else {

            // generate real image
            inputD = inputA;

            // each image will produce a 1x2 vector for the results (aka is fake or not)
            result0 = tf.zeros([inputD.shape[0], 1]);
            result1 = tf.ones([inputD.shape[0], 1]);

            //?? OPTIONAL IMPELEMENT LABEL - SMOOTHING
            //     if label_smoothing:
            //     y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
            // else:
            //     # these are real images
            //     y_disc[:, 1] = 1

            //?? OPTIONAL IMPELEMENT LABEL - FLIPPING
            // if label_flipping > 0:
            // p = np.random.binomial(1, label_flipping)
            // if p > 0:
            //     y_disc[:, [0, 1]] = y_disc[:, [1, 0]]


        }

        // patches is image patches for each image in the batch
        patches = this.extractPatches(inputD);
        // result is a 1x2 vector for each image. (means fake or not)
        result = tf.concat([result0, result1], 1);

        // Update the discriminator
        let dLoss = await this.discriminator.trainOnBatch(patches, result);

        // Dispose of all tensors
        inputA.dispose(); inputB.dispose(); inputD.dispose(); result0.dispose(); result1.dispose(); result.dispose();

        for (let p of patches) {
            p.dispose();
        }

        return dLoss;
    }

    async trainGenBatch(batchSize) {
        // Get the shape of the Input/Output Data
        const shapeA = [batchSize, this.inputHeight, this.inputWidth, this.inputChannels];
        const shapeB = [batchSize, this.outputHeight, this.outputWidth, this.outputChannels];

        // Tensors to dispose of later
        let genInputA, genInputB, genResult0, genResult1, genResult;

        // Create a batch to train the Discriminator
        let batch = await this.dataLoader.loadBatch(batchSize);

        // Generator Input 4D Tensors
        genInputA = tf.tensor4d(batch.A, shapeA);    // Truth
        genInputB = tf.tensor4d(batch.B, shapeB);    // Representation

        genResult0 = tf.zeros([batchSize, 1]);
        genResult1 = tf.ones([batchSize, 1]);
        genResult = tf.concat([genResult0, genResult1], 1);

        // Freeze the discriminator
        // this.discriminator.trainable = false;

        let gLoss = await this.dcgan.trainOnBatch(genInputB, [genInputA, genResult])

        // Unfreeze the discriminator
        // this.discriminator.trainable = true;

        // Dispose All Tensors
        genInputA.dispose(), genInputB.dispose(); genResult0.dispose(); genResult1.dispose(); genResult.dispose();

        return gLoss;
    }

    // TESTING
    //---------------------

    // INFERENCE
    //---------------------

    // TODO
    predict() {

    }


    // UTILS
    //---------------------

    // TODO
    save() {

    }

    extractPatches(inputD) {
        const inputHeight = inputD.shape[1];
        const inputWidth = inputD.shape[2];
        const patchHeight = this.patchDim[0];
        const patchWidth = this.patchDim[1];

        // console.log(inputHeight, inputWidth, patchHeight, patchWidth); // TAKEOUT

        const patches = [];
        for (let y = 0; y < inputHeight; y += patchHeight) {
            for (let x = 0; x < inputWidth; x += patchWidth) {
                let patch = tf.slice4d(inputD, [0, x, y, 0], [inputD.shape[0], patchWidth, patchHeight, inputD.shape[3]]);
                patches.push(patch);
            }
        }
        return patches;
    }

    summary() {
        // Returns a summary of the GAN NN structure to the console

        console.log(tf.getBackend());
        console.log('\n\n--------------------------------------------------------------------------------------------------\n-----------GENERATOR SUMMARY----------||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n--------------------------------------------------------------------------------------------------\n');

        this.generator.summary();

        console.log('\n\n--------------------------------------------------------------------------------------------------\n---------DISCRIMINATOR SUMMARY--------||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n--------------------------------------------------------------------------------------------------\n');

        this.discriminator.summary();

        console.log('\n\n--------------------------------------------------------------------------------------------------\n-------------DCGAN SUMMARY------------||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n--------------------------------------------------------------------------------------------------\n');

        this.dcgan.summary();
    }

    // DEPRICATED
    //---------------------
    _buildDiscriminator() {

        // def d_layer(layer_input, filters, f_size = 4, bn = True):
        // """Discriminator layer"""
        // d = Conv2D(filters, kernel_size = f_size, strides = 2, padding = 'same')(layer_input)
        // d = LeakyReLU(alpha = 0.2)(d)
        // if bn:
        //     d = BatchNormalization(momentum = 0.8)(d)
        // return d



        function dLayer(layer_input, filters, f_size = 4, bn = true) {
            let d = tf.layers.conv2d({
                kernelSize: f_size,
                filters: filters,
                strides: 2,
                // activation: 'relu',
                // kernelInitializer: 'VarianceScaling', 
                padding: 'same'
            }).apply(layer_input);

            d = tf.layers.leakyReLU({
                alpha: 0.2
            }).apply(d);

            if (bn) d = tf.layers.batchNormalization({ momentum: 0.8 }).apply(d);

            return d;
        }

        // img_A = Input(shape = self.img_shape)
        // img_B = Input(shape = self.img_shape)
        let input1 = tf.input({ shape: this.inputShape });
        let input2 = tf.input({ shape: this.inputShape });

        // # Concatenate image and conditioning image by channels to produce input
        // combined_imgs = Concatenate(axis = -1)([img_A, img_B])
        let combinedImgs = tf.layers.concatenate().apply([input1, input2]);

        // d1 = d_layer(combined_imgs, self.df, bn = False)
        // d2 = d_layer(d1, self.df * 2)
        // d3 = d_layer(d2, self.df * 4)
        // d4 = d_layer(d3, self.df * 8)

        const d1 = dLayer(combinedImgs, this.dFilters);
        const d2 = dLayer(d1, this.dFilters * 2);
        const d3 = dLayer(d2, this.dFilters * 4);
        const d4 = dLayer(d3, this.dFilters * 8);

        // validity = Conv2D(1, kernel_size = 4, strides = 1, padding = 'same')(d4)
        const output = tf.layers.conv2d({
            // inputShape: [28, 28, 1],
            kernelSize: 4,
            filters: 1,
            strides: 1,
            padding: 'same'
        }).apply(d4);

        // return Model([img_A, img_B], validity)
        return tf.model({ inputs: [input1, input2], outputs: output });
    }

    async _trainBatch(batchSize = 1) {
        // start_time = datetime.datetime.now()
        // let fakeA, dLossReal, dLossFake, dLoss, gLoss;
        let batch = await this.dataLoader.loadBatch(batchSize);
        let batchA = batch[0];
        let batchB = batch[1];

        // return tf.tidy(async () => {
        const valid = tf.ones([batchSize, this.dPatch, this.dPatch, 1]);
        const fake = tf.zeros([batchSize, this.dPatch, this.dPatch, 1]);

        const shapeA = [batchSize, this.dataLoader.patchSize, this.dataLoader.patchSize, this.dataLoader.numChannelsA];
        const shapeB = [batchSize, this.dataLoader.patchSize, this.dataLoader.patchSize, this.dataLoader.numChannelsB];

        // Input 4D Tensors
        const inputA = tf.tensor4d(batchA, shapeA);
        const inputB = tf.tensor4d(batchB, shapeB)

        // Train Discriminator
        // ---------------------

        // Condition on B and generate a translated version
        const fakeA = this.generator.predictOnBatch(inputB);

        // Train the discriminators(original images = real / generated = Fake)
        const dLossReal = await this.discriminator.trainOnBatch([inputA, inputB], valid);
        console.log(dLossReal);

        const dLossFake = await this.discriminator.trainOnBatch([fakeA, inputB], fake);
        console.log(dLossFake);

        const dLoss = 0.5 * tf.add(dLossReal, dLossFake); // TODO: FIX THIS
        console.log(dLoss);

        // Train Generator
        // ---------------------

        // Train the generators
        const gLoss = await this.combined.trainOnBatch([inputA, inputB], [valid, inputA]);
        console.log(gLoss);


        console.log('[D loss: ' + dLoss[0] + ', acc: ' + (100 * dLoss[1]) + '] [G loss: ' + gLoss[0] + ']');
        // return {disLoss: dLoss, genLoss: gLoss}


        // });
    }

    async _train(data, epochs, batchSize = 1, sampleInterval = 50) {
        // start_time = datetime.datetime.now()
        let fakeA, dLossReal, dLossFake, dLoss, gLoss;

        // # Adversarial loss ground truths
        // valid = np.ones((batch_size,) + self.disc_patch)
        // fake = np.zeros((batch_size,) + self.disc_patch)
        const valid = tf.ones([batchSize, this.dPatch, this.dPatch, 1]); //??
        const fake = tf.zeros([batchSize, this.dPatch, this.dPatch, 1]); //??


        // for epoch in range(epochs):
        //     for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
        for (let epoch in epochs) {
            let batch = await this.dataLoader.loadBatch(batchSize);

            let batchA = batch[0];
            let batchB = batch[1];
        }

        // # ---------------------
        // #  Train Discriminator
        // # ---------------------

        // # Condition on B and generate a translated version
        // fake_A = self.generator.predict(imgs_B)
        fakeA = this.generator.predict(imgPair.b);

        // # Train the discriminators(original images = real / generated = Fake)
        // d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
        // d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
        // d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        dLossReal = await this.discriminator.trainOnBatch([imgPair.a, imgPair.b], valid);
        dLossFake = await this.discriminator.trainOnBatch([fakeA, imgPair.b], fake);
        dLoss = 0.5 * tf.addStrict(dLossReal, dLossFake);

        // # -----------------
        // #  Train Generator
        // # -----------------

        // # Train the generators
        // g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
        gLoss = await this.combined.trainOnBatch([imgPair.a, imgPair.b], [valid, imgPair.a]);

        // elapsed_time = datetime.datetime.now() - start_time
        // # Plot the progress
        console.log('[Epoch ' + epoch + '/' + epochs + '] [Batch ' + batch_i + '/' + self.data_loader.n_batches + '] [D loss: ' + d_loss[0] + ', acc: ' + (100 * d_loss[1]) + '] [G loss: ' + g_loss[0] + '] time: ' + elapsed_time);
        // # If at save interval => save generated image samples
        // if batch_i % sample_interval == 0:
        //     self.sample_images(epoch, batch_i)

        if (batch_i % sampleInterval == 0) {
        }

    }

    static _preprocess(image) {
        return tf.tidy(() => {

            //convert to a tensor 
            const tensor = tf.browser.fromPixels(image).toFloat();

            //resize 
            const resized = tf.image.resizeBilinear(tensor, [256, 256]);

            //normalize 
            const offset = tf.scalar(127.5);
            const normalized = resized.div(offset).sub(tf.scalar(1.0));

            //We add a dimension to get a batch shape 
            // const batched = normalized.expandDims(0);
            return normalized;

        });
    }

    static _postprocess(tensor, w, h) {
        return tf.tidy(() => {

            //normalization factor  
            const scale = tf.scalar(0.5);

            //unnormalize and sqeeze 
            const squeezed = tensor.squeeze().mul(scale).add(scale);

            //resize to canvas size 
            const resized = tf.image.resizeBilinear(squeezed, [w, h]);

            return resized;
        });
    }

    _predict(image) {
        return tf.tidy(() => {

            //get the prediction
            const gImg = model.predict(preprocess(imgData));

            //post process
            const postImg = postprocess(gImg, 512, 512);
            return postImg;

        });
    }

    // TODO maybe not needed
    static _getBinomial(n, p) {
        const flip = false;
        if (p > 0.5) {
            p = 1 - p;
            flip = true
        }

        const log_q = Math.log(1.0 - p);
        let x = 0;
        let sum = 0;

        while (true) {
            sum += Math.log(Math.random()) / (n - x);
            if (sum < log_q) {
                return x;
            }
            x++;
        }
    }
}

class DataLoader {
    // class DataLoader():
    constructor(params) {

        // TODO: Error on no root location provided or set a default once server is up
        this.datasetDirectory = params.datasetDirectory;

        // DATSET SELECTION PARAMS
        this.location = (params.location !== undefined) ? params.location : 'yosemite';
        this.width = (params.width !== undefined) ? params.width : 256;
        this.height = (params.height !== undefined) ? params.height : 256;
        this.scale = (params.scale !== undefined) ? params.scale : 8;
        this.dataset = (params.dataset !== undefined) ? params.dataset : 'train';
        this.channels = (params.channels !== undefined) ? params.channels : [['grid8bin'], ['topo']];
        this.numSamples = (params.numSamples !== undefined) ? params.numSamples : 400;
        this.dataType = (params.dataType !== undefined) ? params.dataType : '.png';

        this.datasetPath = this.datasetDirectory + '/' + this.location + '/' + this.height + 'x' + this.width + '/' + this.scale + '/' + this.dataset + '/';

        this.numChannelsA = this.channels[0].length;
        this.numChannelsB = this.channels[1].length;

        // Construct Paths for Input A (Truth) and Input B (Representation)
        this.datasetPathsA = [];
        this.datasetPathsB = [];


        for (let i = 1; i <= this.numSamples; i++) {

            let inputChannelPathsA = [];
            for (let channelNameA of this.channels[0]) {
                inputChannelPathsA.push(this.datasetPath + channelNameA + '/' + i + this.dataType);
            }
            this.datasetPathsA.push(inputChannelPathsA);

            let inputChannelPathsB = [];
            for (let channelNameB of this.channels[1]) {
                inputChannelPathsB.push(this.datasetPath + channelNameB + '/' + i + this.dataType);
            }
            this.datasetPathsB.push(inputChannelPathsB);
        }

        // Create Model Name String (Used when saving the Model on training)
        const loc = this.location + '_';
        const size = this.height + 'x' + this.width + '_';
        const scale = this.scale + '_';
        let inputChannels = '';

        for (let ich of this.channels[0]) {
            inputChannels = inputChannels + ich + '_';
        }
        let outputChannels = '';
        for (let och of this.channels[1]) {
            outputChannels = outputChannels + och + '_';
        }
        this.modelName = loc + size + scale + inputChannels + 'to_' + outputChannels

    }

    async loadChannel(path) {
        const image = await Jimp.read(path);
        const imageArray = new Array(image.bitmap.width * image.bitmap.height);
        const offset = 127.5;
        for (let i = 0, j = 0; i < imageArray.length; i++) {
            imageArray[i] = image.bitmap.data[j] / offset - 1.0;
            j += 4
        }

        return imageArray;
    }

    async loadChannels(paths) {
        const numChannels = paths.length;
        const images = []

        // Get all of the image channels
        for (let i = 0; i < numChannels; i++) {
            images[i] = await Jimp.read(paths[i]);
        }

        const size = images[0].bitmap.width * images[0].bitmap.width;
        const offset = 127.5; // ?? This is for 8-bit offset

        // Normalize and interleave image channels
        const channelArray = new Array(size * numChannels);
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < numChannels; j++) {
                channelArray[numChannels * i + j] = images[j].bitmap.data[i * 4] / offset - 1.0;
            }
        }

        return channelArray;
    }

    //  DATASET PATH                            (root URL to Datasets)
    //  |---datasets.json                       (JSON describing dataset)
    //  |---DATASET NAME                        (location name?)
    //  |   |---.rawdata                        (blob of raw data for dataset)
    //  |   |   |---nw034.img                   (random data blob)
    //  |   |   |---nw035.img                   (random data blob)
    //  |   |---<DATASET TYPE>                  (train/test/val)
    //  |   |   |---<DATASET CHANNEL/S NAME>    (bitmap[1], rgb[3], rgba[4]);
    //  |   |   |   |---1.jpg                   (numbers should pair across all channels per name and type)
    //  |   |   |   |---2.jpg
    //  |   |   |   |---3.jpg
    //  |   |   |---<DATASET CHANNEL/S NAME>    (bitmap[1], rgb[3], rgba[4]);
    //  |   |       |---1.jpg                   (numbers should pair across all channels per name and type)
    //  |   |       |---2.jpg
    //  |   |       |---3.jpg
    //  |   |---<DATASET TYPE>                  (train/test/val)
    //  |       |---<DATASET CHANNEL/S NAME>    (bitmap[1], rgb[3], rgba[4]);
    //  |       |   |---1.jpg                   (numbers should pair across all channels per name and type)
    //  |       |   |---2.jpg
    //  |       |   |---3.jpg
    //  |       |---<DATASET CHANNEL/S NAME>    (bitmap[1], rgb[3], rgba[4]);
    //  |           |---1.jpg                   (numbers should pair across all channels per name and type)
    //  |           |---2.jpg
    //  |           |---3.jpg
    //  .scripts                            (python scripts for generating data patches)
    //  |---genDataset.py                   (pyhton script)

    async loadBatch(batchSize = 1) {

        let batchArrayA = [];
        let batchArrayB = [];

        for (let b = 0; b < batchSize; b++) {
            let index = Math.floor(Math.random() * this.numSamples);
            let nextA = await this.loadChannels(this.datasetPathsA[index]);
            let nextB = await this.loadChannels(this.datasetPathsB[index]);

            batchArrayA = batchArrayA.concat(nextA);
            batchArrayB = batchArrayB.concat(nextB);
        }

        return { A: batchArrayA, B: batchArrayB };
    }

}

// This custom layer is written in a way that can be saved and loaded.
class MiniBatchDisc extends tf.layers.Layer {
    constructor(config) {
        super(config);
    }

    build(inputShape) {
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1]];
    }

    call(inputs, kwargs) {
        let input = inputs;
        if (Array.isArray(input)) {
            input = input[0];
        }
        this.invokeCallHook(inputs, kwargs);

        return tf.tidy(() => {
            const diffs = tf.sub(tf.expandDims(input, 3), tf.expandDims(tf.transpose(input, [1, 2, 0]), 0));
            const abs_diffs = tf.sum(tf.abs(diffs), 2);
            const x = tf.sum(tf.exp(tf.neg(abs_diffs)), 2);
            return x
        });
    }

    getConfig() {
        const config = super.getConfig();
        // Object.assign(config, {alpha: this.alpha});
        return config;
    }

    static get className() {
        return 'MiniBatchDisc';
    }
}

class Slice extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.slice = config.slice;
    }

    build(inputShape) {

    }

    call(inputs, kwargs) {
        let input = inputs;
        if (Array.isArray(input)) {
            input = input[0];
        }
        this.invokeCallHook(inputs, kwargs);

        return tf.tidy(() => {
            const shp = input.shape;
            return tf.slice(input, [0, this.slice[0], this.slice[1], 0], [shp[0], this.slice[2], this.slice[3], shp[3]]);
        });
    }

    getConfig() {
        const config = super.getConfig();
        Object.assign(config, { slice: this.slice });
        return config;
    }


    static get className() {
        return 'Slice';
    }
}

// Regsiter the custom layer, so TensorFlow.js knows what class constructor to call when deserializing an saved instance of the custom layer.
tf.serialization.registerClass(MiniBatchDisc);
tf.serialization.registerClass(Slice);


// terraGAN = new TerraGAN({});
module.exports = { TerraGAN, DataLoader };