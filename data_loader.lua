require 'paths'

local tablex = require 'pl.tablex'
local _ = require 'moses'

local loader = torch.class('DataLoader')

function loader:__init(config)
	self.batch_size = config.batch_size
	self.src_seq_len = config.src_seq_len
	self.tgt_seq_len = config.tgt_seq_len
	self.src_pad = config.src_pad
	self.tgt_pad = config.tgt_pad
	self.src_eos = config.src_eos
	self.tgt_eos = config.tgt_eos
	self.shuff = config.shuff

	self.thresh = config.thresh

	self.path = config.path
	self.label = config.label
	self.data = torch.load(paths.concat(self.path, self.label .. '.t7'))
	self.data = tablex.filter(
		self.data, function(v) return (v.tgt:nElement() >= self.thresh) end
	)

	self.nbOfshards = #self.data
	self.shard_ix  = 0
end

function loader:reset()
	self.shard_ix = 0
end

function loader:shuffle()
	self.data = _.shuffle(self.data)
end

function loader:nshard()
	return self.nbOfshards
end

function loader:next()
	self.shard_ix = self.shard_ix + 1
	self.shard_ix = (self.shard_ix - 1) % self.nbOfshards + 1
	self.shard = self.data[self.shard_ix]

	local in_src = self.shard.src
	local in_tgt = self.shard.tgt
	local in_offset = self.shard.offset
	local in_ix = self.shard.ix

	local perm = nil
	if self.label == 'train' and self.shuff then
		perm = torch.randperm(in_offset:nElement())
	else
		perm = torch.range(1, in_offset:nElement())
	end

	local out_src = {}
	local out_tgt = {}
	local out_lab = {}
	local out_pos = {}
	local out_ix = {}
	local out_lOfbatch = {}
	local out_nbOfnonzero = {}
	
	local src_len = math.min(in_src:size(2), self.src_seq_len)
	for i = 1, in_offset:nElement(), self.batch_size do
		local j = math.min(i + self.batch_size - 1, in_offset:nElement())
		local batch_size = j - i + 1
		local b_src = torch.LongTensor(src_len, batch_size)
		local b_tgt = torch.LongTensor(self.tgt_seq_len, batch_size)
		local b_lab = torch.LongTensor(self.tgt_seq_len, batch_size)
		local b_pos = torch.range(1, src_len):view(-1, 1):expandAs(b_src)
		local b_ix = torch.LongTensor(batch_size):zero()
		b_src:fill(self.src_pad)
		b_tgt:fill(self.tgt_pad)
		b_lab:fill(self.tgt_pad)

		for ix = i, j do
			local px = perm[ix]
			local tgt_len = ((px < in_offset:nElement()) and in_offset[px + 1] 
				or in_tgt:nElement()) - in_offset[px]
			local lab_len = ((px < in_offset:nElement()) and in_offset[px + 1] 
				or in_tgt:nElement()) - in_offset[px] - 1
			tgt_len = math.min(self.tgt_seq_len, tgt_len)
			lab_len = math.min(self.tgt_seq_len, lab_len)
			b_src:select(2, ix - i + 1):narrow(1, 1, src_len):
				copy(in_src:select(1, px):narrow(1, 1, src_len))
			b_tgt:select(2, ix - i + 1):narrow(1, 1, tgt_len):
				copy(in_tgt:narrow(1, in_offset[px], tgt_len))
			b_lab:select(2, ix - i + 1):narrow(1, 1, lab_len):
				copy(in_tgt:narrow(1, in_offset[px] + 1, lab_len))
			if tgt_len < self.tgt_seq_len then
				b_tgt:select(2, ix - i + 1)[tgt_len + 1] = self.tgt_eos
			end
			if lab_len < self.tgt_seq_len then
				b_lab:select(2, ix - i + 1)[lab_len + 1] = self.tgt_eos
			end
			b_ix[ix - i + 1] = in_ix[px]
		end
		out_src[#out_src + 1] = b_src
		out_tgt[#out_tgt + 1] = b_tgt
		out_lab[#out_lab + 1] = b_lab
		out_pos[#out_pos + 1] = b_pos
		out_ix[#out_ix + 1] = b_ix
		out_lOfbatch[#out_lOfbatch + 1] = batch_size
		out_nbOfnonzero[#out_nbOfnonzero + 1] = b_tgt:clone():ne(self.tgt_pad)
													:sum()
	end
	return self:wrapper(
		out_src, out_tgt, out_lab, out_pos, out_ix,
		out_lOfbatch, out_nbOfnonzero,
		src_len, self.tgt_seq_len, #out_src
	)
end

function loader:wrapper(
	out_src, out_tgt, out_lab, out_pos, out_ix,
	out_lOfbatch, out_nbOfnonzero,
	src_len, tgt_len, nbOfbatch)
	local out = {
		src = out_src, tgt = out_tgt, lab = out_lab, pos = out_pos, ix = out_ix,
		lOfbatch = out_lOfbatch, nbOfnonzero = out_nbOfnonzero,
		src_len = src_len, tgt_len = tgt_len, nbOfbatch = nbOfbatch
	}
	out.__index = function(self, i)
		return {self.src[i], self.tgt[i], self.lab[i], self.pos[i], self.ix[i]}
	end
	
	setmetatable(out, out)

	function out:cuda()
		tablex.transform(function(v) return v:cuda() end, self.src)
		tablex.transform(function(v) return v:cuda() end, self.tgt)
		tablex.transform(function(v) return v:cuda() end, self.lab)
		tablex.transform(function(v) return v:cuda() end, self.pos)
		tablex.transform(function(v) return v:cuda() end, self.ix)
	end

	function out:nonzero()
		return self.nbOfnonzero
	end

	function out:lbatch()
		return self.lOfbatch
	end

	function out:reverse()
		tablex.transform(
			function(v) 
				local t = tablex.range(src_len - 1, 1, -1)
				t[#t + 1] = src_len
				return v:index(1, torch.LongTensor(t)) 
			end,
			self.src
		)
		tablex.transform(
			function(v) 
				local t = tablex.range(src_len - 1, 1, -1)
				t[#t + 1] = src_len
				return v:index(1, torch.LongTensor(t))
			end,
			self.pos
		)
	end

	function out:len()
		return self.src_len, self.tgt_len
	end

	function out:nbatch()
		return self.nbOfbatch
	end

	return out
end
