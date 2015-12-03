--[[!
   Training classifier for foreground(contour)/background pixels using
   patches around the pixels.
--]]

-- should be able to change this to double if wanted
torch.setdefaulttensortype("torch.FloatTensor")

require 'nn'
require 'nne'
require 'nnx'
require 'dp'
require 'dpnn'
require 'rnn'
require 'math'
require 'xlua'
require 'optim'
require 'image'
require 'lfs'

-- Cuda
require 'cutorch'
require 'cunn'

-- My files
require 'updation.help_funcs'
require 'coco.coco'

op = xlua.OptionParser('%prog [options]')

-- Data
op:option{'-d', '--datadir', action='store', dest='datadir',
          help='path to datadir', default=""}
op:option{'-o', '--outdir', action='store', dest='outdir',
          help='path to outdir', default=""}
op:option{'--height', action='store', dest='height', help='height',default=224}
op:option{'--width', action='store', dest='width', help='height',default=224}
op:option{'--crop', action='store_true', dest='crop',
          help='Crop heightxwidth from the input image.', default=false}
op:option{'--noOfTrainSamples', action='store', dest='noOfTrainSamples',
          help='Number of train samples.', default=2391476}
op:option{'--noOfValidSamples', action='store', dest='noOfValidSamples',
          help='Number of train samples.', default=1153660}
op:option{'--maxFnLen', action='store', dest='maxFnLen',
          help='Max filename length.', default=150}

-- Model
op:option{'--dropOuts', action='store', dest='dropOuts',
          help='Feat map drouputs', default='{0, 0, 0, 0}'}
op:option{'--featMaps', action='store', dest='featMaps',
          help='Convolution FeatMaps', default='{32, 48, 64, 128}'}
op:option{'--filtSizes', action='store', dest='filtSizes',
          help='Convolution Filter sizes', default='{3, 3, 3, 3}'}
op:option{'--filtStrides', action='store', dest='filtStrides',
          help='Convolution Filter sizes', default='{1, 1, 1, 1}'}
op:option{'--poolSizes', action='store', dest='poolSizes',
          help='Pooling sizes', default='{2, 2, 2, 2}'}
op:option{'--poolStrides', action='store', dest='poolStrides',
          help='Pooling strides', default='{2, 2, 2, 2}'}
op:option{'--useBatchNorm', action='store_true', dest='useBatchNorm',
          help='Use batch normalization.', default=false}
op:option{'--nonLinearity', action='store', dest='nonLinearity',
          help='non linearity', default='ReLU'}
op:option{'--nHiddens', action='store', dest='nHiddens', help='nHiddens',
          default=512}
op:option{'--iheight', action='store', dest='iheight', help='iheight',
          default=56}
op:option{'--iwidth', action='store', dest='iwidth', help='iwidth',
          default=56}

-- Learning
op:option{'--batchSize', action='store', dest='batchSize',
          help='Batch Size.',default=32}
op:option{'--epochs', action='store', dest='epochs',
          help='epochs',default=10000}
op:option{'--learningRate', action='store', dest='learningRate',
          help='Learning rate',default=0.001}
op:option{'--momentum', action='store', dest='momentum',
          help='Momentum',default=0.9}
op:option{'--learningRateDecay', action='store', dest='learningRateDecay',
          help='Learning rate decay',default=0.00005}
op:option{'--normGradient', action='store_true', dest='normGradient',
          help='Normalize gradients.', default=false}
op:option{'--earlyStopThresh', action='store', dest='earlyStopThresh',
          help='Early stop count',default=1000}

-- Use Cuda
op:option{'--useCuda', action='store_true', dest='useCuda', help='Use GPU',
          default=false}
op:option{'--deviceId', action='store', dest='deviceId', help='GPU device Id',
          default=1}

-- Command line arguments
opt = op:parse()
op:summarize()

datadir = opt.datadir
outdir = opt.outdir
cropHeight = tonumber(opt.height)
cropWidth = tonumber(opt.width)
crop = opt.crop
noOfTrainSamples = tonumber(opt.noOfTrainSamples)
noOfValidSamples = tonumber(opt.noOfValidSamples)
maxFnLen = tonumber(opt.maxFnLen)

-- Data directories
trainImagesDir = paths.concat(datadir, "train/images")
trainMasksDir = paths.concat(datadir, "train/masks")

validImagesDir = paths.concat(datadir, "valid/images")
validMasksDir = paths.concat(datadir, "valid/masks")

trainSamplesDictFile = paths.concat(outdir, "trainSamplesDict.t7")
validSamplesDictFile = paths.concat(outdir, "validSamplesDict.t7")
trainTargetsDictFile = paths.concat(outdir, "trainTargetsDict.t7")
validTargetsDictFile = paths.concat(outdir, "validTargetsDict.t7")

if paths.filep(trainSamplesDictFile) 
   and paths.filep(validSamplesDictFile)
   and paths.filep(trainTargetsDictFile)
   and paths.filep(validTargetsDictFile) then
   print("Loading training samples dictionary")
   trainSamplesDict = torch.load(trainSamplesDictFile)
   trainTargetsDict = torch.load(trainTargetsDictFile)
   print("Loading validation samples dictionary")
   validSamplesDict = torch.load(validSamplesDictFile)
   validTargetsDict = torch.load(validTargetsDictFile)
else
   print("Generating training dictionary")
   trainSamplesDict = torch.CharTensor(noOfTrainSamples, maxFnLen)
   trainSamplesPtr = torch.data(trainSamplesDict)
   trainTargetsDict = torch.CharTensor(noOfTrainSamples, maxFnLen)
   trainTargetsPtr = torch.data(trainTargetsDict)

   indx = 1
   for fp in lfs.dir(trainImagesDir) do
      startIndx, endIndx = fp:find('.jpg')
      if startIndx ~= nil then
         -- images
         fpSample = paths.concat(trainImagesDir, fp)
         ffi.copy(trainSamplesPtr + ((indx-1)*maxFnLen), fpSample)

         -- masks
         fpTarget = paths.concat(trainMasksDir, fp)
         ffi.copy(trainTargetsPtr + ((indx-1)*maxFnLen), fpTarget)

         indx = indx + 1
      end
   end
   trainSamplesDict = trainSamplesDict[{{1, indx-1}}]
   trainTargetsDict = trainTargetsDict[{{1, indx-1}}]

   print("Generating validation dictionary")
   validSamplesDict = torch.CharTensor(noOfValidSamples, maxFnLen)
   validSamplesPtr = torch.data(validSamplesDict)
   validTargetsDict = torch.CharTensor(noOfValidSamples, maxFnLen)
   validTargetsPtr = torch.data(validTargetsDict)

   indx = 1
   for fp in lfs.dir(validImagesDir) do
      startIndx, endIndx = fp:find('.jpg')
      if startIndx ~= nil then
         -- images
         fpSample = paths.concat(validImagesDir, fp)
         ffi.copy(validSamplesPtr + ((indx-1)*maxFnLen), fpSample)

         -- masks
         fpTarget = paths.concat(validMasksDir, fp)
         ffi.copy(validTargetsPtr + ((indx-1)*maxFnLen), fpTarget)

         indx = indx + 1
      end
   end
   validSamplesDict = validSamplesDict[{{1, indx-1}}]
   validTargetsDict = validTargetsDict[{{1, indx-1}}]

   -- Saving to disk
   torch.save(trainSamplesDictFile, trainSamplesDict)
   torch.save(validSamplesDictFile, validSamplesDict)
   torch.save(trainTargetsDictFile, trainTargetsDict)
   torch.save(validTargetsDictFile, validTargetsDict)
end

-- Training Data
trainData = {}
trainData.samples = trainSamplesDict
trainData.labels = trainTargetsDict
trainData.size = function() return trainData.labels:size()[1] end

-- Validation Data
validData = {}
validData.samples = validSamplesDict
validData.labels = validTargetsDict
validData.size = function() return validData.labels:size()[1] end

-- Input features
-- Model
sample = image.load(ffi.string(torch.data(trainSamplesDict[1])))
sampleSize = sample:size()
label = image.load(ffi.string(torch.data(trainTargetsDict[1])))
labelSize = label:size()

nFeats = sample:size(1)
height = sample:size(2)
width = sample:size(3)

-- Model
classes = {'Background', 'Foreground'}
confusion = optim.ConfusionMatrix(classes)

dropOuts = dp.returnString(opt.dropOuts)
featMaps = dp.returnString(opt.featMaps)
filtSizes = dp.returnString(opt.filtSizes)
filtStrides = dp.returnString(opt.filtStrides)
poolSizes = dp.returnString(opt.poolSizes)
poolStrides = dp.returnString(opt.poolStrides)
useBatchNorm = opt.useBatchNorm
nonLinearity = opt.nonLinearity
nHiddens = tonumber(opt.nHiddens)
iheight = tonumber(opt.iheight)
iwidth = tonumber(opt.iwidth)
useCuda = opt.useCuda
deviceId = tonumber(opt.deviceId)

model = nn.Sequential()
if crop then
   preprocessor = nn.Sequential()
   preprocessor:add(nn.SpatialUniformCrop(cropHeight, cropWidth))
end

usingPooling = false
for key, value in pairs(featMaps) do
   if key == 1 then
      -- model1
      model:add(nn.SpatialConvolution(nFeats, value,
                                      filtSizes[key], filtSizes[key],
                                      filtStrides[key], filtStrides[key],
                                      math.floor(filtSizes[key]/2),
                                      math.floor(filtSizes[key]/2)))
   else
      -- model1
      model:add(nn.SpatialConvolution(featMaps[key-1], value,
                                      filtSizes[key], filtSizes[key],
                                      filtStrides[key], filtStrides[key],
                                      math.floor(filtSizes[key]/2),
                                      math.floor(filtSizes[key]/2)))
   end

   if useBatchNorm then
      print("Adding BatchNorm")
      model:add(nn.SpatialBatchNormalization(value))
   elseif dropOuts[key] ~= 0 then
      print("Adding SpatialDropout: " .. tostring(dropOuts[key]))
      model:add(nn.SpatialDropout(dropOuts[key]))
   end

   if poolSizes[key] ~= 0 then
      usingPooling = true
      print("Adding Pooling: PooSize: " .. tostring(poolSizes[key])..
            " Stride: " .. tostring(poolStrides[key]))
      model:add(nn.SpatialMaxPooling(poolSizes[key], poolSizes[key],
                                     poolStrides[key], poolStrides[key]))
   end
   -- model
   model:add(nn[nonLinearity]())
end

if batchNorm then
   sample:resize(1, nFeats, height, width)
end

if crop then sample = preprocessor:forward(sample) end
tempOp = model:forward(sample)
if batchNorm then
   opFeats = tempOp:size(2)
   rHeight = tempOp:size(3)
   rWidth = tempOp:size(4)
else
   opFeats = tempOp:size(1)
   rHeight = tempOp:size(2)
   rWidth = tempOp:size(3)
end

-- Recurrent layer
l = nn.Sequential()
l:add(nn.JoinTable(1, 3))
inputChannels = featMaps[#featMaps] + featMaps[#featMaps]
l:add(nn.SpatialConvolution(inputChannels, featMaps[#featMaps], 1, 1))
l:add(nn[nonLinearity]())
rl = nn.Recurrence(l, {featMaps[#featMaps], rHeight, rWidth}, 3)
rcnn = nn.Repeater(rl, 3)
model:add(rcnn)
model:add(nn.SelectTable(-1))

model:add(nn.Collapse(3))
model:add(nn.Linear(featMaps[#featMaps]*rHeight*rWidth, nHiddens))
model:add(nn.Linear(nHiddens, iheight*iwidth))
model:add(nn.Reshape(1, iheight, iwidth))

-- Resize to input image size
model:add(nn.SpatialScaling{oheight=cropHeight, owidth=cropWidth})

-- Criterion SpatialBinaryLogisticRegression
criterion = nn.SpatialBinaryLogisticRegression()

if useCuda then
   print("Using GPU:"..deviceId)
   cutorch.setDevice(deviceId)
   print("GPU set")
   preprocessor:cuda()
   model:cuda()
   print("Model copied to CUDA")
   criterion:cuda()
   print("Criterion copied to CUDA")
else
   print("Not using GPU")
end

-- Retrieve parameters and gradients
parameters, gradParameters = model:getParameters()

-- Optimizers: Using SGD [Stocastic Gradient Descent]
batchSize = tonumber(opt.batchSize)
epochs = tonumber(opt.epochs)
learningRate = tonumber(opt.learningRate)
momentum = tonumber(opt.momentum)
learningRateDecay = tonumber(opt.learningRateDecay)
normGradient = opt.normGradient
earlyStopThresh = tonumber(opt.earlyStopThresh)

print("Using Stocastic gradient descent")
optimState = {
               coefL1 = 0,
               coefL2 = 0,
               learningRate = learningRate,
               weightDecay = 0.0,
               momentum = momentum,
               learningRateDecay = learningRateDecay
             }
optimMethod = optim.sgd
print(optimState)

displayProgress = true
imageLevel = true
useBce = true
bceThresh = 0
best_test_accu = 0
best_test_model = nn.Sequential()
best_train_accu = 0
previous_train_accu = 0
best_train_model = nn.Sequential()
times = torch.Tensor(epochs)
earlyStopCount = 0

print(model)

for i=1,epochs do
   model:training()
   trainAccu, time = coco.model_train_batch(model, criterion, parameters,
                                      gradParameters, trainData, sampleSize,
                                      optimMethod,
                                      optimState, batchSize, i, confusion,
                                      trainLogger, normGradient, useCuda,
                                      useBce, displayProgress, bceThresh,
                                      imageLevel, labelSize, preprocessor)
   -- logging
   print(previous_train_accu, trainAccu)
   previous_train_accu = trainAccu

   --[[
   model:evaluate()
   model_test_gpu_batch(model, validData, sampleSize, batchSize, confusion,
                  deviceId, testLogger, setDevice, useBce, bceThresh,
                  displayProgress, imageLevel)

   testAccu = confusion.totalValid
   print(best_test_accu, testAccu)
   if best_test_accu < testAccu then
      earlyStopCount = 0
      best_test_accu = testAccu
      best_test_model = model:clone()
   else
      earlyStopCount = earlyStopCount + 1
   end
   print("EarlyStopCount: "..tostring(earlyStopCount))
   if earlyStopCount >= earlyStopThresh then
      print("Early stopping")
      break
   end
   --]]
end
