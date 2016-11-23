local nninit = require 'nninit'

local tablex = require 'pl.tablex'
local stringx = require 'pl.stringx'

local models = require 'models_maker'
local recurrent = require 'recurrent'
local model_utils = require 'model_utils'
local maker = require 'data_maker'

require 'misc'

local Seq2seq = torch.class('Seq2seq')

function Seq2seq:__init(opt, encoder, decoder)
	self.opt = opt
	if encoder and decoder then
		self.encoder, self.decoder = encoder, decoder
	else
		self.encoder, self.decoder = self:create_networks(opt)
	end
	if opt.attn_net == 'conv' then
		self.params, self.grad_params = model_utils.combine_all_parameters(
			self.encoder.lookup, self.decoder.lookup, self.decoder.rnn_attn	
		)
	else
		self.params, self.grad_params = model_utils.combine_all_parameters(
			self.encoder.lookup, self.encoder.frnn, self.encoder.brnn,
			self.decoder.lookup, self.decoder.rnn_attn
		)
	end

	if not(encoder and decoder) then
		self.params:uniform(-0.1, 0.1)
	end

	print('the number of parameters is ' .. self.params:size(1))

	self:initialize_net(opt)
	self.enc_init_state, self.dec_init_state = self:initialize_state(opt)
end

function Seq2seq:initialize_net(opt)
	self.clones = {}
	self.clones.encoder = {}	
	for name, proto in pairs(self.encoder) do
		if name ~= 'lookup' then
			print('cloning encoder ' .. name)
			self.clones.encoder[name] = model_utils.clone_many_times(
				proto, opt.src_seq_len, not proto.parameters
			)
		else
			self.clones.encoder[name] = self.encoder.lookup
		end
	end

	self.clones.decoder = {}
	for name, proto in pairs(self.decoder) do
		if name ~= 'lookup' then
			print('cloning decoder ' .. name)
			self.clones.decoder[name] = model_utils.clone_many_times(
				proto, opt.tgt_seq_len, not proto.parameters
			)
		else
			self.clones.decoder[name] = self.decoder.lookup
		end
	end
end

function Seq2seq:initialize_state(opt)
	local enc_init_state = {}
	for L = 1, opt.nlayer do
		local h_init = torch.zeros(opt.batch_size, opt.enc_rnn_size)
		if opt.cuda then h_init = h_init:cuda() end
		table.insert(enc_init_state, h_init:clone())
		if opt.rnn == 'lstm' then
			table.insert(enc_init_state, h_init:clone())
		end
	end

	local dec_init_state = {}
	for L = 1, opt.nlayer do
		local h_init = torch.zeros(opt.batch_size, opt.dec_rnn_size)
		if opt.cuda then h_init = h_init:cuda() end
		table.insert(dec_init_state, h_init:clone())
		if opt.rnn == 'lstm' then
			table.insert(dec_init_state, h_init:clone())
		end
	end

	return enc_init_state, dec_init_state
end

function Seq2seq:create_networks(opt)
	local encoder = {}

	local m = nn.ParallelTable()
	m:add(nn.LookupTable(opt.src_vocab, opt.emb))
	m:add(nn.LookupTable(opt.src_pos, opt.emb))
	encoder.lookup = nn.Sequential():add(m):add(nn.CAddTable())
	
	if opt.attn_net ~= 'conv' then
		encoder.frnn = recurrent[opt.rnn](
			opt.emb, opt.enc_rnn_size, opt.nlayer, opt.dropout
		)
		encoder.brnn = recurrent[opt.rnn](
			opt.emb, opt.enc_rnn_size, opt.nlayer, opt.dropout
		)
	end

	local decoder = {}
	decoder.lookup = nn.LookupTable(opt.tgt_vocab, opt.emb)

	decoder.rnn_attn = (opt.rnn == 'lstm') and models.decoder_lstm_attn(opt) 
							or models.decoder_gru_attn(opt)
	
	decoder.criterion = nn.ClassNLLCriterion()
	
	if opt.cuda then
		for k, v in pairs(encoder) do v:cuda() end
		for k, v in pairs(decoder) do v:cuda() end
	end	

	return encoder, decoder
end

function Seq2seq:trainb(opt, src, tgt, lab, pos)
	return function(params)
		if params ~= self.params then
			self.params:copy(params)
		end
		self.grad_params:zero()

		local src_len = src:size(1)
		local tgt_len = tgt:size(1)
		local batch_size = src:size(2)

		local enc_init_state = clone_list(self.enc_init_state)
		local dec_init_state = clone_list(self.dec_init_state)
		if batch_size ~= self.batch_size then
			tablex.transform(
				function(v) return v:resize(batch_size, v:size(2)):zero() end,
				enc_init_state
			)
			tablex.transform(
				function(v) return v:resize(batch_size, v:size(2)):zero() end,
				dec_init_state
			)
		end

		local loss = 0

		-- forward pass

		-- encoder
		local enc_frnn_state = {[0] = clone_list(enc_init_state)}
		local enc_brnn_state = {[0] = clone_list(enc_init_state)}
		local enc_frnn_output = nil
		local enc_brnn_output = nil

		local enc_lookup = self.clones.encoder.lookup
		enc_lookup:training()
		enc_reps = enc_lookup:forward({src, pos})
		
		local context = {}
		if self.clones.encoder.frnn then
			for t = 1, src_len do
				local frnn = self.clones.encoder.frnn[t]		
				local brnn = self.clones.encoder.brnn[t]		
				frnn:training()
				brnn:training()
				enc_frnn_state[t] = frnn:forward(
					{enc_reps[t], unpack(enc_frnn_state[t - 1])}
				)
				enc_brnn_state[t] = brnn:forward(
					{enc_reps[src_len - t + 1], unpack(enc_brnn_state[t - 1])}
				)
				if type(enc_frnn_state[t]) ~= 'table' then
					enc_frnn_state[t] = {enc_frnn_state[t]}
					enc_brnn_state[t] = {enc_brnn_state[t]}
				end
			end
			enc_frnn_output = tablex.imap(
				function(v) 
					local h = v[#v]:clone()
					return h:view(1, unpack(h:size():totable())) end,
					enc_frnn_state
			)
			enc_brnn_output = tablex.imap(
				function(v) 
					local h = v[#v]:clone()
					return h:view(1, unpack(h:size():totable()))	
				end,
				enc_brnn_state
			)
			enc_frnn_output = torch.cat(enc_frnn_output, 1)
			enc_brnn_output = torch.cat(enc_brnn_output, 1)
			enc_brnn_output = enc_brnn_output:index(
				1, torch.range(src_len, 1, -1):long()
			)
			context = torch.add(enc_frnn_output, enc_brnn_output)
		else
			context = enc_reps
		end

		context = context:transpose(1, 2):contiguous()

		-- decoder
		local dec_rnn_state = {[0] = clone_list(dec_init_state)}
		if self.clones.encoder.frnn and
			opt.enc_rnn_size == opt.dec_rnn_size then
			dec_rnn_state[0] = tablex.imap2(
				torch.add, enc_frnn_state[src_len], enc_brnn_state[src_len]
			)
		end
		
		local dec_lookup = self.clones.decoder.lookup
		dec_lookup:training()
		dec_reps = dec_lookup:forward(tgt)

		local dec_preds = {}
		for t = 1, tgt_len do
			local rnn_attn = self.clones.decoder.rnn_attn[t]
			rnn_attn:training()
			dec_rnn_state[t] = rnn_attn:forward(
				{dec_reps[t], context, unpack(dec_rnn_state[t - 1])}
			)
			dec_preds[t] = table.remove(dec_rnn_state[t])

			local criterion = self.clones.decoder.criterion[t]
			local err = criterion:forward(dec_preds[t], lab[t])
			loss = loss + err
		end
		loss = loss / tgt_len

		-- backward pass
		
		-- decoder
		local dec_drnn_state = {[tgt_len] = clone_list(dec_init_state)}
		local dcontext = {}
		local dec_dreps = {}

		for t = tgt_len, 1, -1 do
			local criterion = self.clones.decoder.criterion[t]
			local dec_dpred = criterion:backward(dec_preds[t], lab[t])

			table.insert(dec_drnn_state[t], dec_dpred)

			local rnn_attn = self.clones.decoder.rnn_attn[t]
			dec_drnn_state[t-1] = rnn_attn:backward(
				{dec_reps[t], context, unpack(dec_rnn_state[t-1])}, 
				dec_drnn_state[t]
			)
			
			dec_dreps[t] = table.remove(dec_drnn_state[t-1], 1)
			dcontext[t] = table.remove(dec_drnn_state[t-1], 1)

			dec_dreps[t] = dec_dreps[t]:view(
				1, unpack(dec_dreps[t]:size():totable())
			)
		end
		dec_dreps = torch.cat(dec_dreps, 1)
		dcontext = tablex.reduce(torch.add, dcontext) 

		dec_lookup:backward(tgt, dec_dreps)

		-- encoder
		local enc_dfrnn_state = {[src_len] = clone_list(enc_init_state)}	
		local enc_dbrnn_state = {[src_len] = clone_list(enc_init_state)}
		if self.clones.encoder.frnn and 
			opt.enc_rnn_size == opt.dec_rnn_size then
			enc_dfrnn_state[src_len] = clone_list(dec_drnn_state[0])
			enc_dbrnn_state[src_len] = clone_list(dec_drnn_state[0])
		end
		
		dcontext = dcontext:transpose(1, 2):contiguous()

		local enc_dreps  = {}
		if self.clones.encoder.frnn then
			for t = src_len, 1, -1 do
				local frnn = self.clones.encoder.frnn[t]	
				local brnn = self.clones.encoder.brnn[t]	
				enc_dfrnn_state[t][#enc_dfrnn_state[t]]:add(dcontext[t])
				enc_dbrnn_state[t][#enc_dbrnn_state[t]]:add(dcontext[src_len-t+1])
				enc_dfrnn_state[t - 1] = frnn:backward(
					{enc_reps[t], unpack(enc_frnn_state[t - 1])},
					#enc_dfrnn_state[t] == 1 and 
						enc_dfrnn_state[t][1] or enc_dfrnn_state[t]
				)
				enc_dbrnn_state[t - 1] = brnn:backward(
					{enc_reps[src_len - t + 1], unpack(enc_brnn_state[t - 1])},
					#enc_dbrnn_state[t] == 1 and
					enc_dbrnn_state[t][1] or enc_dbrnn_state[t]
				)

				local dfrep = table.remove(enc_dfrnn_state[t - 1], 1)  
				local dbrep = table.remove(enc_dbrnn_state[t - 1], 1)
				dfrep = dfrep:view(1, unpack(dfrep:size():totable()))
				dbrep = dbrep:view(1, unpack(dbrep:size():totable()))
				enc_dreps[t] = enc_dreps[t] and enc_dreps[t]:add(dfrep) or dfrep
				enc_dreps[src_len - t + 1] = enc_dreps[src_len - t + 1] and
					enc_dreps[src_len - t + 1]:add(dbrep) or dbrep
			end
			enc_dreps = torch.cat(enc_dreps, 1)
		else
			enc_dreps = dcontext
		end
		
		enc_lookup:backward({src, pos}, enc_dreps)
		self.grad_params:div(tgt_len)
		self.grad_params:clamp(-opt.grad_clip, opt.grad_clip)
		
		return loss, self.grad_params 
	end
end

function Seq2seq:evalb(src, tgt, lab, pos)
	local src_len = src:size(1)
	local tgt_len = tgt:size(1)
	local batch_size = src:size(2)

	local enc_init_state = clone_list(self.enc_init_state)
	local dec_init_state = clone_list(self.dec_init_state)
	if batch_size ~= self.batch_size then
		tablex.transform(
			function(v) return v:resize(batch_size, v:size(2)):zero() end,
			enc_init_state
		)
		tablex.transform(
			function(v) return v:resize(batch_size, v:size(2)):zero() end,
			dec_init_state
		)
	end

	local loss = 0

	-- encoder
	local enc_frnn_state = {[0] = clone_list(enc_init_state)}
	local enc_brnn_state = {[0] = clone_list(enc_init_state)}
	local enc_frnn_output = nil
	local enc_frnn_output = nil

	local enc_lookup = self.clones.encoder.lookup
	enc_lookup:evaluate()
	enc_reps = enc_lookup:forward({src, pos})

	local context = {}
	if self.clones.encoder.frnn then
		for t = 1, src_len do
			local frnn = self.clones.encoder.frnn[t]		
			local brnn = self.clones.encoder.brnn[t]		
			frnn:evaluate()
			brnn:evaluate()
			enc_frnn_state[t] = frnn:forward(
				{enc_reps[t], unpack(enc_frnn_state[t - 1])}
			)
			enc_brnn_state[t] = brnn:forward(
				{enc_reps[src_len - t + 1], unpack(enc_brnn_state[t - 1])}
			)
			if type(enc_frnn_state[t]) ~= 'table' then
				enc_frnn_state[t] = {enc_frnn_state[t]}
				enc_brnn_state[t] = {enc_brnn_state[t]}
			end
		end
		enc_frnn_output = tablex.imap(
			function(v)
				local h = v[#v]:clone()
				return h:view(1, unpack(h:size():totable())) end,
				enc_frnn_state
		)
		enc_brnn_output = tablex.imap(
			function(v)
				local h = v[#v]:clone()
				return h:view(1, unpack(h:size():totable())) end,
				enc_brnn_state
		)
		enc_frnn_output = torch.cat(enc_frnn_output, 1)
		enc_brnn_output = torch.cat(enc_brnn_output, 1)
		enc_brnn_output = enc_brnn_output:index(
			1, torch.range(src_len, 1, -1):long()
		)
		context = torch.add(enc_frnn_output, enc_brnn_output)
	else
		context = enc_reps
	end
	context = context:transpose(1, 2):contiguous()

	-- decoder
	local dec_rnn_state = {[0] = clone_list(dec_init_state)}
	if self.clones.encoder.frnn and
		opt.enc_rnn_size == opt.dec_rnn_size then
		dec_rnn_state[0] = tablex.imap2(
			torch.add, enc_frnn_state[src_len], enc_brnn_state[src_len]
		)
	end
	
	local dec_lookup = self.clones.decoder.lookup
	dec_lookup:evaluate()
	dec_reps = dec_lookup:forward(tgt)

	local dec_preds = {}
	for t = 1, tgt_len do
		local rnn_attn = self.clones.decoder.rnn_attn[t]
		rnn_attn:evaluate()
		dec_rnn_state[t] = rnn_attn:forward(
			{dec_reps[t], context, unpack(dec_rnn_state[t-1])}
		)
		dec_preds[t] = table.remove(dec_rnn_state[t])

		local criterion = self.clones.decoder.criterion[t]
		local err = criterion:forward(dec_preds[t], lab[t])
		loss = loss + err
	end
	loss = loss / tgt_len
	return loss	
end

function Seq2seq:test(src, pos)
	local src_len = src:size(1)
	local batch_size = src:size(2)

	local enc_init_state = clone_list(self.enc_init_state)
	local dec_init_state = clone_list(self.dec_init_state)
	if batch_size ~= self.batch_size then
		tablex.transform(
			function(v) return v:resize(batch_size, v:size(2)):zero() end,
			enc_init_state
		)
		tablex.transform(
			function(v) return v:resize(batch_size, v:size(2)):zero() end,
			dec_init_state
		)
	end

	local loss = 0

	-- encoder
	local enc_frnn_state = {[0] = clone_list(enc_init_state)}
	local enc_brnn_state = {[0] = clone_list(enc_init_state)}
	local enc_frnn_output = nil
	local enc_brnn_output = nil

	local enc_lookup = self.clones.encoder.lookup
	enc_lookup:evaluate()
	enc_reps = enc_lookup:forward({src, pos})

	local context = {}
	if self.clones.encoder.frnn then
		for t = 1, src_len do
			local frnn = self.clones.encoder.frnn[t]		
			local brnn = self.clones.encoder.brnn[t]		
			frnn:evaluate()
			brnn:evaluate()
			enc_frnn_state[t] = frnn:forward(
				{enc_reps[t], unpack(enc_frnn_state[t - 1])}
			)
			enc_brnn_state[t] = brnn:forward(
				{enc_reps[src_len - t + 1], unpack(enc_brnn_state[t - 1])}
			)
			if type(enc_frnn_state[t]) ~= 'table' then
				enc_frnn_state[t] = {enc_frnn_state[t]}
				enc_brnn_state[t] = {enc_brnn_state[t]}
			end
		end
		enc_frnn_output = tablex.imap(
			function(v)
				local h = v[#v]:clone()
				return h:view(1, unpack(h:size():totable()))
			end,
			enc_frnn_state
		)
		enc_brnn_output = tablex.imap(
			function(v)
				local h = v[#v]:clone()
				return h:view(1, unpack(h:size():totable()))
			end,
			enc_brnn_state
		)
		enc_frnn_output = torch.cat(enc_frnn_output, 1)
		enc_brnn_output = torch.cat(enc_brnn_output, 1)
		enc_brnn_output = enc_brnn_output:index(
			1, torch.range(src_len, 1, -1):long()
		)
		local concat_h = self.clones.encoder.concat_h
		concat_h:evaluate()
		context = concat_h:forward({enc_frnn_output, enc_brnn_output})
	else
		context = enc_reps
	end
	context = context:transpose(1, 2):contiguous()

	-- generator
	local dec_rnn_state = clone_list(dec_init_state)
	if self.clones.encoder.frnn and
		opt.enc_rnn_size == opt.dec_rnn_size then
		local concat_last = self.clones.encoder.concat_last
		concat_last:evaluate()
		dec_rnn_state = concat_last:forward(
			{enc_frnn_state[src_len], enc_brnn_state[src_len]}
		)
	end

	local generator = { 
		decoder = self.decoder,
		context = context, dec_rnn_state = dec_rnn_state,
	}
	setmetatable(generator, generator)

	function generator:step(tgt)
		local dec_lookup = self.decoder.lookup
		dec_lookup:evaluate()
		local dec_rep = dec_lookup:forward(tgt)
		local rnn_attn = self.decoder.rnn_attn
		rnn_attn:evaluate()
		self.dec_rnn_state = rnn_attn:forward(
			{dec_rep, self.context, unpack(self.dec_rnn_state)}
		)
		local dec_pred = table.remove(self.dec_rnn_state)

		return dec_pred
	end

	function generator:getState()
		return clone_list(self.dec_rnn_state)
	end

	function generator:setState(state)
		self.dec_rnn_state = clone_list(state)
	end

	return generator
end

function Seq2seq:parameters()
	return self.params, self.grad_params
end

function Seq2seq.load(path)
	local net, opt = unpack(torch.load(path))
	local s2s = Seq2seq.new(opt, unpack(net))
	return s2s
end

function Seq2seq:save(path)
	torch.save(path, {{self.encoder, self.decoder}, self.opt})
end

