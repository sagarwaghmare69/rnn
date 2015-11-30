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
op:option{'--height', action='store', dest='height', help='height',default=256}
op:option{'--width', action='store', dest='width', help='height',default=304}
op:option{'--noOfTrainSamples', action='store', dest='noOfTrainSamples',
          help='Number of train samples.', default=28000000}
op:option{'--noOfValidSamples', action='store', dest='noOfValidSamples',
          help='Number of train samples.', default=16000000}

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
height = tonumber(opt.height)
width = tonumber(opt.width)
noOfTrainSamples = tonumber(opt.noOfTrainSamples)
noOfValidSamples = tonumber(opt.noOfValidSamples)

-- Input features
nFeats = 3
height = 224 
width = 224

batch = torch.rand(32, nFeats, height, width)

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

tempOp = model:forward(batch)
opFeats = tempOp:size(2)
rHeight = tempOp:size(3)
rWidth = tempOp:size(4)

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
model:add(nn.SpatialScaling{oheight=height, owidth=width})

-- Criterion SpatialBinaryLogisticRegression
criterion = nn.SpatialBinaryLogisticRegression()

if useCuda then
   print("Using GPU:"..deviceId)
   cutorch.setDevice(deviceId)
   print("GPU set")
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
imageLevel = false
bceThresh = 0.5
best_test_accu = 0
best_test_model = nn.Sequential()
best_train_accu = 0
previous_train_accu = 0
best_train_model = nn.Sequential()
times = torch.Tensor(epochs)
earlyStopCount = 0

print(model)

-- Sample forward
trainInputs = batch:cuda()
trainTargets = torch.rand(32, 1, height, width)
trainTargets = trainTargets:cuda()
outputs = model:forward(trainInputs)
f = criterion:forward(outputs, trainTargets)

df_do = criterion:backward(outputs, trainTargets)
model:backward(trainInputs, df_do)
