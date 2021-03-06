require 'paths'

require 'beam'
require 'greedy'

local maker = require 'data_maker'

local Tester = torch.class('Tester')

function Tester:__init(model, sdict, tdict)
	self.model = model
	self.sdict = sdict
	self.tdict = tdict
end

function Tester:test(test, opt)
	local score = 0
	local nbOfwords = 0
	local tIter = 0

	local timer = torch.Timer()

	local path = paths.concat(
		opt.output,
		string.format(
			'out-%s-%s-best-%s',
			opt.name, torch.type(self.model),os.date('%Y%m%d[%H%M]')
		)
	) 
	local npath = paths.concat(
		opt.output,
		string.format(
			'out-%s-%s-nbest-%s',
			opt.name, torch.type(self.model),os.date('%Y%m%d[%H%M]')
		)
	)
	local fbest = assert(io.open(path, 'w'))
	local fnbest = nil
	if opt.search == 'beam' and opt.beam_size > 1 then
		fnbest = assert(io.open(npath, 'w'))
	end

	local nbOfshard = test:nshard()
	for i = 1, nbOfshard do
		local shard = test:next()
		if opt.cuda then shard:cuda() end
		if opt.reverse then shard:reverse() end
		local nbOfbatch = shard:nbatch()
		for j = 1, nbOfbatch do
			timer:reset()

			local src, tgt, lab, pos, ix = unpack(shard[j])
			local output = self.strategy:search(opt, src, pos)
			local best_score, best_tgt, nbest_score, nbest_tgt = unpack(output)
			local best_tgt = maker.convert_ix(best_tgt, self.tdict, true)
			for i = 1, #best_tgt do
				score = score + best_score[i]
				nbOfwords = nbOfwords + #best_tgt[i]
				fbest:write(ix[i], '\t', best_tgt[i], '\n')
			end
			if fnbest then
				local nbest_tgt =  maker.convert_ix(nbest_tgt, self.tdict, true)
				for k = 1, #nbest_tgt do
					fnbest:write(ix[1], '\t', k, '\t', nbest_tgt[k], '\n')
				end
				fnbest:write('\n')
			end

			tIter = tIter + 1
			if tIter % opt.nprint == 0 then 
				print(string.format(
					'batch = %3d, nOfbatch = %d, shard = %d, nOfshard = %d, ' ..  
					'time = %.4fs',
					j, nbOfbatch, i, nbOfshard, timer:time().real
				))
			end
		end
	end
	fbest:close()
	if fnbest then fnbest:close() end

	print(string.format('=>best output = %s', path))
	print(string.format('=>nbest output = %s', npath))

	return score, nbOfwords
end

function Tester:run(test, opt)
	if opt.search == 'greedy' then
		self.strategy = Greedy(self.model)
	else
		self.strategy = Beam(self.model)
	end
	test:reset()

	if not paths.dirp(opt.output) then
		os.execute('mkdir -p ' .. opt.output)
	end
	
	local timer = torch.Timer()
	local score, nbOfwords = self:test(test, opt)
	print(string.format(
		'=>average score = %6.4e, ppl = %6.4e, time = %.4fs',
		score / nbOfwords, math.exp(-score / nbOfwords), timer:time().real
	))
end
