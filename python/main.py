import TerraGAN

# Model and Dataset Directory
datasetDir = '../terra-datasets/temp'
modelDir = '../terra-models/temp'

# ??-MAIN TEST-------------------------------------------------------------------------


terraGAN = TerraGAN(
    inputShape = [256, 256, 1],
    outputShape = [256, 256, 1],
    patchDim = [64, 64],
    genFilters = 32,
    disFilters = 32,
    loadData = False,
    summary=True
    )

terraGAN.loadData(
    datasetDirectory = datasetDir,
    location = 'yosemite',
    width = 256,
    height = 256,
    scale = 8,
    dataset = 'train',
    channels = [['topo'], ['grid_8_bin']],
    numSamples = 1000,
    dataType = '.png'
)

terraGAN.train({
    modelTag: 'trial1',
    modelDirectory: modelDir,
    epochs: 800,
    batchesPerEpoch: 100,
    batchSize: [1, 1],
    logOnBatch: true,
    logOnEpoch: true,
    saveOnEpoch: 10
});


# ??-MAIN TEST-------------------------------------------------------------------------

