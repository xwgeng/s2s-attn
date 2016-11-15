require 'paths'
require 'optim'

local maker = require 'data_maker'

local tablex = require 'pl.tablex'

local Trainer = torch.class('Trainer')

function Trainer:__init(model)
	self.model = model
	self.params, self.grad_params = self.model:parameters()
end

function Trainer:train(epoch, train, opt)
	local timer = torch.Timer()

	local tIter = 0
	local tLoss = {}

	train:reset()
	if opt.curriculum < epoch then
		train:shuffle()
	end
	
	local nbOfshard = train:nshard()
	for i = 1, nbOfshard do
		local shard = train:next()
		if opt.cuda then shard:cuda() end
		if opt.reverse then shard:reverse() end
		local nbOfbatch = shard:nbatch()
		local lOfbatch = shard:lbatch()
		local nbOfnonzero = shard:nonzero()
		for j = 1, nbOfbatch do
			timer:reset()
			local feval = self.model:trainb(opt, unpack(shard[j]))
			local _, loss = optim[opt.optim](feval, self.params, opt.optim_config)
			tLoss[#tLoss + 1] = loss[1] * lOfbatch[j] / nbOfnonzero[j]
			tIter = tIter + 1
			if tIter % opt.nprint == 0 then
				print(string.format(
					'%3d/%d/%d/%d (epoch %d), err = %6.4e, grad = %6.4e, time = %.4fs',
					j, nbOfbatch, i, nbOfshard, epoch, tLoss[#tLoss],
					self.grad_params:norm() / self.params:norm(),
					timer:time().real
				))
			end
		end
	end

	local loss = tablex.reduce('+', tLoss)
	loss = loss / tIter
	return loss
end

function Trainer:eval(epoch, valid, opt)
	local timer = torch.Timer()

	local vIter = 0
	local vLoss = {}
	
	valid:reset()
	local nbOfshard = valid:nshard()
	for i = 1, nbOfshard do
		local shard = valid:next()
		if opt.cuda then shard:cuda() end
		if opt.reverse then shard:reverse() end
		local nbOfbatch = shard:nbatch()
		local lOfbatch = shard:lbatch()
		local nbOfnonzero = shard:nonzero()
		for j = 1, nbOfbatch do
			timer:reset()
			local loss = self.model:evalb(unpack(shard[j]))
			vLoss[#vLoss + 1] = loss * lOfbatch[j] / nbOfnonzero[j]
			vIter = vIter + 1
		end
	end

	local loss = tablex.reduce('+', vLoss)
	loss = loss / vIter
	return loss
end

function Trainer:run(train, valid, opt)
	local tLosses = {}
	local vLosses = {}

	local lr = opt.learningRate
	local shrink_factor = opt.shrink_factor
	local shrink_multiplier = opt.shrink_multiplier

	local timer = torch.Timer()

	for i = 1, opt.nepoch do
		timer:reset()
		local lr = opt.optim_config.learningRate
		local tLoss = self:train(i, train, opt)
		print(string.format(
			'=>[epoch %d] training loss = %6.4e, lr = %.4f, time = %.4fs',
			i, tLoss, lr, timer:time().real
		))
		
		timer:reset()

		collectgarbage()
		
		local vLoss = self:eval(i, valid, opt)
		print(string.format(
			'=>[epoch %d] valid loss = %6.4e, time = %.4fs',
			i, vLoss, timer:time().real
		))

		collectgarbage()
		
		local name = string.format(
			'model-%s-%s-epoch%.2f-t%.4e-v%.4e-%s.t7',
			opt.name, torch.type(self.model), i, tLoss, vLoss, 
			os.date('%Y%m%d[%H%M]')
		)
		self.model:save(paths.concat(opt.save, name))

		if opt.optim == 'sgd' and #vLosses > 1 and 
			vLosses[#vLosses] > vLoss * opt.shrink_multiplier
		then
			lr = lr / opt.shrink_factor
			lr = math.max(lr, opt.minLearningRate)
			opt.optim_config.learningRate = lr
		end

		if opt.anneal and i > opt.start_epoch then
			lr = lr - (opt.learningRate - opt.minLearningRate) / opt.saturate_epoch
			lr = math.max(lr, opt.minLearningRate)
			opt.optim_config.learningRate = lr
		end

		tLosses[#tLosses + 1] = tLoss
		vLosses[#vLosses + 1] = vLoss
	end
	local name = string.format('loss-%s-%s-nepoch%.2f-%s.t7',
		opt.name, torch.type(self.model), opt.nepoch, os.date('%Y%m%d[%H%M]')
	)
	torch.save(paths.concat(opt.save, name), {tLosses, vLosses})
end
