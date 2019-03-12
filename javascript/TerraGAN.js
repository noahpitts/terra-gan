const tf = require('@tensorflow/tfjs-node-gpu');
const Jimp = require('jimp');

// Terra GAN Model Builder
class TerraGAN {
    // // TODO - ENTER FULL DISCRIPTION OF THE CLASS HERE
    constructor(params) {

        // ------------------------
        // Input shape [rows, columns, channels]
        this.inputHeight = (params.inputHeight) ? params.inputHeight : 256;
        this.inputWidth = (params.inputWidth) ? params.inputWidth : 256;
        this.inputChannels = (params.inputChannels) ? params.inputChannels : 1;
        this.inputShape = (!params.inputShape) ? [this.inputHeight, this.inputWidth, this.inputChannels] : params.inputShape;
        this.inputChannels = this.inputShape[2];

        // ------------------------
        // Output shape [rows, columns, channels]
        this.outputHeight = (params.outputHeight) ? params.outputHeight : 256;
        this.outputWidth = (params.outputWidth) ? params.outputWidth : 256;
        this.outputChannels = (params.outputChannels) ? params.outputChannels : 1;
        this.outputShape = (!params.outputShape) ? [this.outputHeight, this.outputWidth, this.outputChannels] : params.outputShape;
        this.outputChannels = this.outputShape[2];

        // ------------------------
        // Discriminator Patches [rows, cols]
        this.patchDim = (params.patchDim) ? params.patchDim : [64, 64];
        this.numPatches = (this.outputShape[0] / this.patchDim[0]) * (this.outputShape[1] / this.patchDim[1]);
        this.patchShape = [this.patchDim[0], this.patchDim[1], this.outputShape[2]];

        // ------------------------
        // Number of Generator and Discriminator Filters
        this.genFilters = (params.genFilters) ? params.genFilters : 32;
        this.disFilters = (params.disFilters) ? params.disFilters : 32;

        // ------------------------
        // Define Optimizers
        this.learningRate = (params.learningRate) ? params.learningRate : 1E-4;
        this.beta1 = (params.beta1) ? params.beta1 : 0.9;
        this.beta2 = (params.beta2) ? params.beta2 : 0.999;
        this.epsilon = (params.epsilon) ? params.epsilon : 1e-08;
        this.adamGAN = tf.train.adam(this.learningRate, this.beta1, this.beta2, this.epsilon);
        this.adamDisc = tf.train.adam(this.learningRate, this.beta1, this.beta2, this.epsilon);

        // ----------------------
        // Build PatchDiscriminator - the patch gan averages loss across sub patches of the image
        this.discriminator = this.buildPatchDiscriminator();

        // ---------------------
        // Compile Discriminator
        // this.discriminator.trainable = true;
        this.discriminator.compile({ optimizer: this.adamDisc, loss: 'binaryCrossentropy' });
        // this.discriminator.summary();
        this.discriminator.trainable = false; // disable training while we put it through the GAN

        // ----------------------
        // Build Generator - Our generator is an AutoEncoder with U-NET skip connections
        this.generator = this.buildGenerator();

        // -------------------------
        // Compile Generator
        this.generator.compile({ loss: 'meanAbsoluteError', optimizer: this.adamDisc });

        // ----------------------
        // Build DCGAN
        this.dcgan = this.buildDCGAN();

        // ---------------------
        // Compile DCGAN
        this.dcgan.compile({ optimizer: this.adamGAN, loss: ['meanAbsoluteError', 'binaryCrossentropy'], lossWeights: [1E2, 1] }); //?? Loss Weights ??

        // ---------------------
        // Load Data on Class Construction
        this.dataLoader = (params.loadData) ? new DataLoader(params.loadData) : params.loadData;
        this.batchCounter = 0;

        // ---------------------
        // Log Terra GAN summary
        if (params.summary) this.summary();
    }

    //---------------------
    // STRUCTURE
    //---------------------

    buildDCGAN() {

        // 1. Generate an image with the generator
        // 2. break up the generated image into patches
        // 3. feed the patches to a discriminator to get the avg loss across all patches
        //     (i.e is it fake or not)
        // 4. the DCGAN outputs the generated image and the loss

        const genInput = tf.input({ shape: this.inputShape, name: 'DCGAN_input' });

        //  generated image model from the generator
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
                let SLC = new Slice({ slice: [rowIndex, colIndex, ph, pw] });
                let xPatch = SLC.apply(genOutput);
                genPatchList.push(xPatch);
            }
        }

        // measure loss from patches of the image (not the actual image)
        const dcganOutput = this.discriminator.apply(genPatchList);

        // actually turn into keras model
        const DCGAN = tf.model({ inputs: genInput, outputs: [genOutput, dcganOutput], name: 'DCGAN' });

        return DCGAN;
    }

    buildGenerator() {
        const UNET = true;
        // UNET Generator 
        // [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]

        // Generator does the following:
        // 1. Takes in an image
        // 2. Generates an image from this image

        // Convolution-BatchNorm-ReLU layer with k ﬁlters.
        function conv2d(layer_input, filters, f_size = 4, bn = true) {

            // Convolutional Layer
            let d = tf.layers.conv2d({ kernelSize: f_size, filters: filters, strides: 2, padding: 'same' }).apply(layer_input);

            // Batch Normalization Layer
            if (bn) d = tf.layers.batchNormalization({ momentum: 0.8 }).apply(d);

            // Leaky ReLU Layer
            d = tf.layers.leakyReLU({ alpha: 0.2 }).apply(d);

            return d;
        }

        // Convolution-BatchNorm-Dropout-ReLUlayer with a dropout rate of 50%
        function deconv2d(layer_input, skip_input, filters, f_size = 4, dropout_rate = 0.5) {
            //Upsampling Layer
            let u = tf.layers.upSampling2d({ size: [2, 2] }).apply(layer_input);

            // Convolutional Layer
            u = tf.layers.conv2d({ kernelSize: f_size, filters: filters, strides: 1, padding: 'same' }).apply(u);

            // Batch Normalization Layer
            u = tf.layers.batchNormalization({ momentum: 0.8 }).apply(u);

            // Dropout Layer
            if (dropout_rate) u = tf.layers.dropout({ rate: dropout_rate }).apply(u);

            // Concatination (skip connections) Layer
            if (UNET) u = tf.layers.concatenate().apply([u, skip_input]);

            // Leaky ReLU Layer
            u = tf.layers.reLU().apply(u);

            return u;
        }

        // -------------------------------
        // ENCODER
        // C64-C128-C256-C512-C512-C512-C512-C512
        // 1 layer block = Conv - BN - LeakyRelu
        // -------------------------------

        // Image input
        const input = tf.input({ shape: this.inputShape, name: 'unet_input' });

        const d1 = conv2d(input, this.genFilters, 4, false); // default: C64
        const d2 = conv2d(d1, this.genFilters * 2);          // default: C128
        const d3 = conv2d(d2, this.genFilters * 4);          // default: C256
        const d4 = conv2d(d3, this.genFilters * 8);          // default: C512
        const d5 = conv2d(d4, this.genFilters * 8);          // default: C512
        const d6 = conv2d(d5, this.genFilters * 8);          // default: C512
        const d7 = conv2d(d6, this.genFilters * 8);          // default: C512
        const d8 = conv2d(d7, this.genFilters * 8);          // default: C512

        //-------------------------------
        // DECODER
        // CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128 
        // 1 layer block = Conv - Upsample - BN - DO - Relu
        // also adds skip connections (Concatenate). Takes input from previous layer matching encoder layer
        //-------------------------------

        const u1 = deconv2d(d8, d7, this.genFilters * 8);    // default: C512
        const u2 = deconv2d(u1, d6, this.genFilters * 16);   // default: C1024
        const u3 = deconv2d(u2, d5, this.genFilters * 16);   // default: C1024
        const u4 = deconv2d(u3, d4, this.genFilters * 16);   // default: C1024
        const u5 = deconv2d(u4, d3, this.genFilters * 16);   // default: C1024
        const u6 = deconv2d(u5, d2, this.genFilters * 8);    // default: C512
        const u7 = deconv2d(u6, d1, this.genFilters * 4);    // default: C256

        // ?? Why no C128 Layer Here

        // After the last layer in the decoder, a convolution is applied
        // to map to the number of output channels (3 in general,
        // except in colorization, where it is 2), followed by a Tanh
        // function.
        const u8 = tf.layers.upSampling2d({ size: [2, 2] }).apply(u7);
        const output = tf.layers.conv2d({
            name: 'unet_output',
            kernelSize: 4,
            filters: this.outputChannels,
            strides: 1,
            activation: 'tanh',
            padding: 'same'
        }).apply(u8);

        // return Model(input, output)
        return tf.model({ inputs: input, outputs: output });
    }

    buildPatchDiscriminator() {

        // -------------------------------
        // DISCRIMINATOR
        // C64-C128-C256-C512-C512-C512 (for 256x256)
        // otherwise, it scales from 64
        // 1 layer block = Conv - BN - LeakyRelu
        // -------------------------------
        function dLayer(layer_input, filters, f_size = 4, bn = true) {
            let d = tf.layers.conv2d({
                kernelSize: f_size,
                filters: filters,
                strides: 2,
                padding: 'same'
            }).apply(layer_input);

            if (bn) d = tf.layers.batchNormalization({ momentum: 0.8 }).apply(d);

            d = tf.layers.leakyReLU({ alpha: 0.2 }).apply(d);

            return d;
        }

        const stride = 2;
        const inputLayer = tf.input({ shape: this.patchShape });
        // this.disFilters

        const numConv = Math.floor(Math.log(this.outputShape[1] / Math.log(2)));
        const filterList = [];
        for (let i = 0; i < numConv; i++) {
            filterList.push(this.disFilters * Math.min(8, Math.pow(2, i)));
        }

        // CONV 1
        let d = dLayer(inputLayer, this.disFilters, 4, false);

        // CONV 2 - CONV N
        for (let i = 1; i < filterList.length; i++) {
            d = dLayer(d, filterList[i]);
        }

        // generate a list of inputs for the different patches to the network
        const inputList = [];
        for (let i = 0; i < this.numPatches; i++) {
            inputList.push(tf.input({ shape: this.patchShape, name: 'patch_input_' + i }));
        }

        // get an activation
        const xFlat = tf.layers.flatten().apply(d);
        let xx = tf.layers.dense({ units: 2, activation: 'softmax', name: 'dis_dense' }).apply(xFlat);

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

        const MBD = new MiniBatchDisc({ name: 'mini_batch_discriminator' });
        xMbd = MBD.apply(xMbd);
        x = tf.layers.concatenate(1).apply([x, xMbd]);

        const xOut = tf.layers.dense({ units: 2, activation: 'softmax', name: 'disc_output' }).apply(x)
        const discriminator = tf.model({ inputs: inputList, outputs: xOut, name: 'discriminator_nn' });

        return discriminator;
    }

    //---------------------
    // TRAINING
    //---------------------

    loadData(params) {
        this.dataLoader = new DataLoader(params);
    }

    async train(params) {
        const epochs = (params.epochs !== undefined) ? params.epochs : 100;
        const bpe = (params.batchesPerEpoch !== undefined) ? params.batchesPerEpoch : 1;
        const batchSize = (params.batchSize !== undefined) ? params.batchSize : [1, 1];

        const logOnBatch = (params.logOnBatch !== undefined) ? params.logOnBatch : true;
        const logOnEpoch = (params.logOnEpoch !== undefined) ? params.logOnEpoch : true;
        const saveOnEpoch = (params.saveOnEpoch !== undefined) ? params.saveOnEpoch : 10;

        const epochAvgStats = [];



        // TODO - Finish this
        
        for (let e = 1; e <= epochs; e++) {
            let avgStats = await this.trainEpoch(bpe, batchSize, [logOnEpoch, logOnBatch]);

            epochAvgStats.push(avgStats);

            if (e % saveOnEpoch === 0) {
                let saveResults1 = await this.discriminator.save(savePath + '/model_dis');
                let saveResults2 = await this.generator.save(savePath + '/model_gen_' + e);
                let saveResults3 = await this.dcgan.save(savePath + '/model_gan');
                console.log('MODEL SAVED TO: ' + savePath);
            }
        }


        // Add Other shit to do per epoch
        // Print summary
        // Save file
    }

    async saveModel(params) {

        // TODO - finish this
        const e = (params.epoch !== undefined) params
        // Saveing the Model
        const fil = this.disFilters + this.genFilters + '_'
        const modelName = this.dataset.modelName + fil + modelTag
        const savePath = 'file://' + params.modelDirectory + '/node/' + modelName;
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
        discLoss = await this.trainDiscBatch(discBatchSize);

        // Train the discriminator on the Batch and return the Loss
        genLoss = await this.trainGenBatch(genBatchSize);

        // Increment the batch Counter
        this.batchCounter++;

        //TODO - log batch time
        let loss = [discLoss, Math.min(genLoss[0]), Math.min(genLoss[1]), Math.min(genLoss[2])];

        if (logBatchStats) console.log('[D logloss: ' + loss[0] + '] [G total: ' + loss[1] + ' L1 (mae): ' + loss[2] + ' logloss: ' + loss[3] + ']');

        return loss;
    }

    async trainDiscBatch(batchSize) {
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
        const shapeA = [batchSize, this.dataLoader.height, this.dataLoader.width, this.dataLoader.numChannelsA];
        const shapeB = [batchSize, this.dataLoader.height, this.dataLoader.width, this.dataLoader.numChannelsB];

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

    //---------------------
    // INFERENCE
    //---------------------

    // TODO
    predict() {

    }

    //---------------------
    // HELPERS
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

    //---------------------
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

        const d1 = dLayer(combinedImgs, this.disFilters);
        const d2 = dLayer(d1, this.disFilters * 2);
        const d3 = dLayer(d2, this.disFilters * 4);
        const d4 = dLayer(d3, this.disFilters * 8);

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
        const inputChannels = '';

        for (let ich of this.dataset.channels[0]) {
            inputChannels = inputChannels + ich + '_';
        }
        const outputChannels = '';
        for (let och of this.dataset.channels[1]) {
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

//??-MAIN TEST-------------------------------------------------------------------------
const terraGAN = new TerraGAN({
    inputShape: [256, 256, 1],
    outputShape: [256, 256, 1],
    patchDim: [64, 64],
    genFilters: 32,
    disFilters: 32,
    loadData: false,
    summary: true,
});

terraGAN.loadData({
    datasetDirectory: './src/img',
    location: 'yosemite',
    width: 256,
    height: 256,
    scale: 8,
    dataset: 'train',
    channels: [['topo'], ['grid_8_bin']],
    numSamples: 1000,
    dataType: '.png'
});

terraGAN.train({
    modelTag: 'trial1',
    modelDirectory: 'models',
    epochs: 800,
    batchesPerEpoch: 100,
    batchSize: [1, 1],
    logOnBatch: true,
    logOnEpoch: true,
    saveOnEpoch: 10
});
//??-MAIN TEST-------------------------------------------------------------------------


//?? MISC TESTING---------------------------------------------------------
// p2p.discriminator.summary();
// p2p.generator.summary();

// console.log(p2p.dataLoader.datasetPathsA[0].length);
// for (let i = 0; i < epochs; i++) {
//     p2p.trainBatch(1);
// }
// let y = tf.zeros([4, 32]);
// let x = tf.ones([4, 100, 5]);

// const diffs = tf.sub(tf.expandDims(x, 3), tf.expandDims(tf.transpose(x, [1, 2, 0]), 0));
// const abs_diffs = tf.sum(tf.abs(diffs), 2);
// x = tf.sum(tf.exp(tf.neg(abs_diffs)), 2);

// let z = tf.concat([y, x], 1);
// console.log(y.shape);
// console.log(x.shape);
// console.log(z.shape);
// const fakeA = tf.ones([16, 256, 256, 1]);
// const fake1 = tf.ones([fakeA.shape[0], 1]);
// const fake2 = tf.zeros([fakeA.shape[0], 1]);
// const fake = tf.concat([fake1, fake2], 1);

// console.log(fake.shape);
// fake.print();
//?? MISC TESTING---------------------------------------------------------


