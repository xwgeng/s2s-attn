local tablex = require 'pl.tablex'
local maker = require 'data_maker'

local greedy = torch.class('Greedy')

function greedy:__init(model)
	self.model = model
end

function greedy:search(opt, src, pos)
	local batch_size = src:size(2)
	local tgt = torch.Tensor(opt.tgt_seq_len + 1, batch_size):typeAs(src)
	local score = torch.Tensor(batch_size):typeAs(src):zero()
	tgt:fill(opt.tgt_pad)
	tgt[1]:fill(opt.tgt_eos)

	local generator = self.model:test(opt, src, pos)
	local prob, ix = nil, nil
	for t = 1, opt.tgt_seq_len do	
		local pred = generator:step(tgt[t])
		local out = pred:clone()
		out:select(2, opt.tgt_pad):fill(-math.huge)
		prob, ix = out:max(2)
		score:add(prob:view(-1))
		tgt[t + 1]:copy(ix:view(-1))

		if tgt:eq(opt.tgt_eos):sum(1)
			:ge(2):sum() >= batch_size then
			break
		end
	end

	return {score, tgt:t()}
end
