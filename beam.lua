local tablex = require 'pl.tablex'

local beam = torch.class('Beam')

function beam:__init(model)
	self.model = model
end

function beam:search(opt, src, pos)
	local src = src:expand(src:size(1), opt.beam_size) 
	local pos = pos:expand(pos:size(1), opt.beam_size)

	local tgt = torch.Tensor(opt.tgt_seq_len + 1, opt.beam_size):typeAs(src)
	tgt:fill(opt.tgt_pad)
	tgt[1]:fill(opt.tgt_eos)

	local candidate_tgt, candidate_score = nil, nil

	local generator = self.model:test(opt, src, pos)
	local score, ix = nil, nil
	for t = 1, opt.tgt_seq_len do	
		local pred = generator:step(tgt[t])
		local out = pred:clone()
		out:select(2, opt.tgt_pad):fill(-math.huge)
		if t ~= 1 then out:add(score:view(-1, 1):expandAs(out)) end
		if t == 1 then out = out[1] end
		local col = out:size(out:nDimension())
		out = out:view(-1)	
		score, ix = out:topk(opt.beam_size, 1, true, true)
		local ix_row = ix:clone():long():add(-1):div(col):add(1)
		local ix_col = ix:clone():long():add(-1):remainder(col):add(1)
		tgt = tgt:index(2,  ix_row:typeAs(ix))	
		tgt[t + 1]:copy(ix_col:typeAs(ix))

		local state = generator:getState()
		tablex.transform(
			function(v) return v:index(1, ix_row) end, state
		)
		generator:setState(state)

		if tgt[t + 1][1] == opt.tgt_eos then
			break
		else
			local cand = score[tgt[t + 1]:eq(opt.tgt_eos)] 
			if cand:nElement() ~= 0 then
				cand_score, idx = cand:max(1)	
				if not best_score or cand_score[1] > best_score then
					best_score = cand_score[1]
					best_tgt = tgt:select(2, idx[1]):clone()
				end
			end
		end
	end

	if opt.strategy or not best_tgt then
		best_score = score:narrow(1, 1, 1)
		best_tgt = tgt:narrow(2, 1, 1):clone()
	end

	local output = nil
	if opt.nbest then
		output = {best_score, best_tgt:t(), score, tgt:t()}
	else
		output = {best_score, best_tgt:t()}
	end
	return output
end
