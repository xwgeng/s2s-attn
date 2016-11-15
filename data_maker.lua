require 'torch'

require 'math'
local tds = require 'tds'
local _ = require 'moses'

local maker = {}

function maker.dict(path,  min_freq)
	local UNK = '<unk>'
	local PAD = '<pad>'
	local BOS = '<s>'
	local EOS = '</s>'
	min_freq = min_freq or 0

	local w2c = {[UNK] = 0, [PAD] = 0, [BOS] = 0, [EOS] = 0}
	local i2w = {[1] = UNK, [2] = PAD, [3] = BOS, [4] = EOS}
	local w2i = {[UNK] = 1, [PAD] = 2, [BOS] = 3, [EOS] = 4}

	local f = assert(io.open(path, 'r'))
	local rawdata = nil 
	while true do
		rawdata = f:read()
		if rawdata == nil then break end
		rawdata = maker.clear_sent(rawdata)
		for w in rawdata:lower():gmatch('%S+') do
			if not w2c[w] then
				w2c[w] = 1
			else
				w2c[w] = w2c[w] + 1
			end
		end
		w2c[EOS] = w2c[EOS] + 1
	end
	f:close()
	
	for w, c in pairs(w2c) do
		if not w2i[w] then
			if  w2c[w] >= min_freq  then
				i2w[#i2w + 1] = w
				w2i[w] = #i2w
			else
				w2i[w] = w2i[UNK]
			end
		end
	end
	print('[the number of uique words is ' .. tablex.size(w2c) .. '.]')
	print('[the size of vocab is ' .. #i2w .. '.]')

	local vocab = {UNK = UNK, PAD = PAD, BOS = BOS, EOS = EOS}
	vocab.w2c = w2c
	vocab.i2w = i2w
	vocab.w2i = w2i
	vocab.min_freq = min_freq

	return vocab
end

function maker.clear_sent(s)
	s = s:gsub('\t', '')
	s = s:gsub('^%s+', '')
	s = s:gsub('%s+$', '')
	s = s:gsub('%s+', ' ')
	return s
end

function maker.filter(spath, tpath, dict)
	local sf = assert(io.open(spath, 'r'))
	local tf = assert(io.open(tpath, 'w'))

	local line = nil
	local words = nil
	while true do
		line = sf:read()
		if line == nil then break end
		line = maker.clear_sent(line)
		words = line:split(' ')
		tablex.transform(
			function(v) return dict.i2w[dict.w2i[v] or dict.w2i[dict.UNK]] end,
			words
		)
		tf:write(table.concat(words, ' '), '\n')
	end
	
	sf:close()
	tf:close()
end

function maker.convert_ix(ix, dict, filter)
	local t = (type(ix) == 'table') and ix or ix:long():totable()
	local str = {}
	local elem = nil
	for i = 1, #t do
		if type(t[i]) == 'number' then
			str[i] = dict.i2w[t[i]]
			elem = 'number'
		elseif type(t[i] == 'table') then
			str[i] = maker.convert_ix(t[i], dict, filter)
			elem = 'table'
		end
	end

	if elem == 'number' then
		if filter then
			local ix1 = tablex.find(str,dict.EOS)
			ix1 = ix1 and (ix1 + 1) or ix1
			local ix2 = tablex.find(str,dict.EOS, ix1)
			ix2 = ix2 and (ix2 - 1) or ix2
			str = tablex.sub(str, ix1, ix2)
		end	
		return table.concat(str, ' ')
	else
		return str
	end
end

function maker.convert_sent(sent, dict, sep)
	local words = sent:split(' ')	
	local data = {}
	for i, w in ipairs(words) do
		local ix = dict.w2i[w] or dict.w2i[dict.UNK]
		data[#data + 1] = ix
	end
	if sep == 'src' then
		table.insert(data, dict.w2i[dict.EOS])
	else
		table.insert(data, 1, dict.w2i[dict.EOS])
	end
	assert(#words + 1 == #data)
	return torch.LongTensor(data)
end

function maker.convert(spath, tpath, sdict, tdict)
	local sf = assert(io.open(spath, 'r'))
	local tf = assert(io.open(tpath, 'r'))

	local src_sents = tds.Vec()
	local tgt_sents = tds.Vec()
	local tgt_lens = {}

	local bins = {lens = {}, ixs = {}}

	local sline, tline = nil, nil
	local swords, twords = nil, nil
	local max_tgt_len = 0
	local max_src_len = 0
	while true do
		sline = sf:read()
		tline = tf:read()

		if not (sline and tline) then break end

		sline = maker.clear_sent(sline)
		tline = maker.clear_sent(tline)

		swords = sline:split(' ')
		twords = tline:split(' ')

		src_sents[#src_sents + 1] = sline
		tgt_sents[#tgt_sents + 1] = tline

		local len = #swords + 1
		bins.lens[len] = (bins.lens[len] or 0) + 1
		bins.ixs[len] = bins.ixs[len] or {}
		table.insert(bins.ixs[len], #src_sents)
		tgt_lens[#tgt_lens + 1] = #twords + 1
		max_src_len = math.max(#swords + 1, max_src_len)
		max_tgt_len = math.max(#twords + 1, max_tgt_len)
	end
	sf:close()
	tf:close()

	assert(#src_sents == #tgt_sents)
	print('the number of sentence: ' .. #src_sents)
	print('the maximum of source sentence length: ' .. max_src_len)
	print('the maximum of target sentence length: ' .. max_tgt_len)

	local nlines, ntgts, nsrcs = 0, 0, 0

	local data = {} 
	for len, nb in pairs(bins.lens) do
		local src = torch.LongTensor(nb, len):zero()
		local tgt = torch.LongTensor(nb * max_tgt_len):zero()
		local offset = torch.LongTensor(nb):fill(1)
		local ix = shuff and _.shuffle(bins.ixs[len]) or bins.ixs[len]
		
		local ntgt = 0
		for i, sent_ix in ipairs(ix) do	
			local src_sent = src_sents[sent_ix]
			local tgt_sent = tgt_sents[sent_ix]

			src:select(1, i):copy(maker.convert_sent(src_sent, sdict, 'src'))	
			tgt:narrow(1, offset[i], tgt_lens[sent_ix]):copy(maker.convert_sent(tgt_sent, tdict, 'tgt'))
			if i < nb then
				offset[i + 1] = offset[i] + tgt_lens[sent_ix]
			end
			ntgt = ntgt + tgt_lens[sent_ix]
		end
		tgt = tgt:narrow(1, 1, ntgt):clone()	
		data[#data + 1] = {
			src = src, tgt = tgt, offset = offset, ix = torch.LongTensor(ix)
		}	

		nlines = nlines + #ix
		ntgts = ntgts + ntgt
		nsrcs = nsrcs + len * nb
	end

	print(string.format('nlines: %d, ntokens(src: %d, tgt: %d)', nlines, nsrcs, ntgts))
	return data
end

return maker
