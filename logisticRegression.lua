-- lua programming for logistic regression just want to use optim packages

require('nn')
require('optim')
require('xlua')

-- parse command line options
parser = torch.CmdLine()
parser:text()
parser:text()
parser:text('Train a Logistic Regression with torch')
parser:text()
parser:text('Options')
parser:option('-algo', 'LBFGS', 'which optimization algorithm to use: LBFGS/SGD/CG')
parser:option('-iter', 1000, 'the number of maximum iterations for train LBFGS or SGD')
parser:option('-train', 'train.t7', 'training data for mnist data')
parser:option('-test', 'test.t7', 'test data for mnist data')
parser:option('-output', 'output.model', 'output model')
parser:option('-batch', 20, 'mini batch size')
ret = parser:parse(arg)

if ret.algo == 'SGD' then
   optimMethod = optim.sgd
   optim_params = {
      learningRate = 1e-3,
      learningRateDecay = 1e-4,
      weightDecay = 0,
      momentum = 0
   }
   print('Use Optimization algorithm SGD')

elseif ret.algo == 'LBFGS' then
   optimMethod = optim.lbfgs
   optim_params = {
      -- lineSearch = optim.lswolfe,
      learningRate = 1e-3,
      maxIter = ret.iter,
      nCorrection = 10
      -- verbose = true
   }
   print('Use Optimization algorithm LBFGS')

elseif ret.algo == 'CG' then
   optimMethod = optim.cg
   optim_params = {
      maxIter = ret.iter
   }
   print('Use Optimization algorithm CG')

else
   print 'Error, must be (LBFGS|SGD)'   
   os.exit()
end

if ret.algo == 'SGD' and ret.iter < 100 then
   print('Warning: Well, small number of iterations is not enough for SGD')
end

-- processing each parameters
batchSize = ret.batch
print('Batch size for training is ' .. batchSize )
print('Maximum Iterations ' .. ret.iter )
print('Training data is ' .. ret.train )
print('Testing data is ' .. ret.test)
print('Model will be saved as ' .. ret.output)

classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}

-- record the accuracy for each iteration
trainLogger = optim.Logger( paths.concat( paths.cwd(), 'train.log' ) )
testLogger  = optim.Logger( paths.concat( paths.cwd(), 'test.log' ) )

-- confusion matrix
confusion = optim.ConfusionMatrix( classes )

-- load data
train = torch.load( ret.train, 'ascii')
test  = torch.load( ret.test, 'ascii')

X = train.data
Y = train.labels

-- define the model
s = X:size()
inputSize = s[3]*s[4]
outputSize = 10
sampleSize = s[1]

print("input size is  " .. inputSize .. "\n")
print("output size is " .. outputSize .. "\n")

model = nn.Sequential()
model:add( nn.Reshape(inputSize) )
model:add( nn.Linear(inputSize, outputSize) )
model:add( nn.LogSoftMax() )

-- define loss function
criterion = nn.ClassNLLCriterion()

-- try to train the model
param, derivative = model:getParameters()

epochs = 1e2

for i=1,epochs do
   
   -- value to record the current loss
   current_loss = 0
   print('start epochs = ' .. i)

   local time = sys.clock()
   confusion:zero()

   for j = 1, sampleSize, batchSize do

      inputs = {}
      targets = {}

      startIdx = j
      endIdx = math.min(j + batchSize, sampleSize)

      for t=startIdx, endIdx do
         table.insert(inputs,  X[t])
         table.insert(targets, Y[t])
      end

      feval = function ( weight )
   
         if param ~= weight then
            param:copy(weight)
         end
   
         derivative:zero()
         local loss_sample = 0

         for t = 1, batchSize do
            local sample = torch.Tensor(inputSize,1)
            sample:copy(inputs[t])

            local label  = targets[t]

            local model_output = model:forward(sample)
            local loss_iter = criterion:forward(model_output, label)
            local criterion_derivative = criterion:backward(model.output, label)
            model:backward(sample, criterion_derivative )

            -- get statistics about the error
            confusion:add(model_output, label)
            loss_sample = loss_sample + loss_iter
         end

         loss_sample = loss_sample / batchSize
         derivative:div(#inputs)
         return loss_sample, derivative
      end
      
      __, fs= optimMethod(feval, param, optim_params)

      current_loss = current_loss + fs[1]
      xlua.progress(j, s[1])
   end

   trainLogger:add{['errrate']=confusion.totalValid*100}
   print(confusion)

   time = sys.clock() - time
   time = time / s[1]

   print('each sample cost ' .. (time*1000) .. ' ms')

   current_loss = current_loss / s[1]
   print('epoch = ' .. i ..
            ' of ' .. epochs ..
            ' current loss = ' .. current_loss )

   if confusion.totalValid*100 >= 100 then
      break
   end
end

trainLogger:style{['errrate']='-'}
trainLogger:plot()

