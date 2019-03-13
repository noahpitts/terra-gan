const { TerraGAN, DataLoader } = require('./TerraGAN.js');

const datasetDir = '../terra-datasets/public'
const modelTempDir = '../terra-models/temp'
// const modelExportDir = '../terra-models/public'

// ??-MAIN TEST LOOP-----


const terraGAN = new TerraGAN({
    dFilters: 32,
    gFilters: 32
});

terraGAN.loadData({
    datasetDirectory: datasetDir,
    location: 'yosemite',
    width: 256,
    height: 256,
    scale: 8,
    dataset: 'train',
    channels: [['topo'], ['grid8bin']],
    numSamples: 1000,
    dataType: '.png'
});

terraGAN.train({
    modelTag: 'trial1',
    modelDirectory: modelTempDir,
    epochs: 800,
    batchesPerEpoch: 100,
    batchSize: [2, 2],
    logOnBatch: true,
    logOnEpoch: true,
    saveOnEpoch: 10
});


// ??-MAIN TEST LOOP-----