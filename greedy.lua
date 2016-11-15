local tablex = require 'pl.tablex'
local maker = require 'data_maker'

local greedy = torch.class('Greedy')

function greedy:__init(model, opt)
	self.model = model
	self.tgt_seq_len = opt.tgt_seq_len
	self.tgt_eos = opt.tgt_eos
	self.tgt_pad = opt.tgt_pad
end

function greedy:search(src, pos)
	local batch_size = src:size(2)
	local tgt = torch.Tensor(self.tgt_seq_len + 1, batch_size):typeAs(src)
	local score = torch.Tensor(batch_size):typeAs(src):zero()
	tgt:fill(self.tgt_pad)
	tgt[1]:fill(self.tgt_eos)

	local generator = self.model:test(src, pos)
	local prob, ix = nil, nil
	for t = 1, self.tgt_seq_len do	
		local pred = generator:step(tgt[t])
		local out = pred:clone()
		out:select(2, self.tgt_pad):fill(-math.huge)
		prob, ix = out:max(2)
		score:add(prob:view(-1))
		tgt[t + 1]:copy(ix:view(-1))

		if tgt:eq(self.tgt_eos):sum(1)
			:ge(2):sum() >= batch_size then
			break
		end
	end

	return {score, tgt:t()}
end
