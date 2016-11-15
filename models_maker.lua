require 'nn'
require 'nngraph'

local tablex = require 'pl.tablex'

local recurrent = require 'recurrent'

local models = {}

function models.decoder_lstm_attn(opt)
	local inputs = {}
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()()) -- context
	for L = 1, opt.nlayer do
		table.insert(inputs, nn.Identity()()) -- prev_c[L]
		table.insert(inputs, nn.Identity()()) -- prev_h[L]
	end

	local outputs = {}
	local x = inputs[1]
	local prev_c = inputs[#inputs - 1]
	local prev_h = inputs[#inputs]
	
	local attn_h = nn.CAddTable(){
		nn.Linear(opt.emb, opt.dec_rnn_size)(x),
		nn.Linear(opt.dec_rnn_size, opt.dec_rnn_size)(prev_c),
		nn.Linear(opt.dec_rnn_size, opt.dec_rnn_size)(prev_h)
	}

	local attention = nil
	if opt.attn_net == 'conv' then
		attention = models.decoder_conv_attn(opt)
	else
		attention = models.decoder_mlp_attn(opt)
	end

	local attn = attention({attn_h, inputs[2]}):annotate{name = 'attn'}

	local input_size_L
	for L = 1, opt.nlayer do
		local prev_c = inputs[2 * L + 1]
		local prev_h = inputs[2 * (L + 1)]
		if L == 1 then
			x = inputs[1]
			input_size_L = opt.emb
		else
			x = outputs[2 * (L - 1)]
			if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x) end
			input_size_L = opt.dec_rnn_size	
		end

		local i2h = nn.Linear(input_size_L, 4 * opt.dec_rnn_size)(x)
						:annotate{name = 'i2h_' .. L}
		local h2h = nn.Linear(opt.dec_rnn_size, 4 * opt.dec_rnn_size)(prev_h)
						:annotate{name = 'h2h_' .. L}
		local c2h = nn.Linear(opt.enc_rnn_size, 4 * opt.dec_rnn_size)(attn)
						:annotate{name = 'c2h_' .. L}

		local all_input_sums = nn.CAddTable()(
			L == opt.nlayer and {i2h, h2h, c2h} or {i2h, h2h}
		)
		local reshaped = nn.Reshape(4, opt.dec_rnn_size)(all_input_sums)
		local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
		-- decode the gates
		local in_gate = nn.Sigmoid()(n1)
		local forget_gate = nn.Sigmoid()(n2)
		local out_gate = nn.Sigmoid()(n3)
		-- decode the write input
		local in_transform = nn.Tanh()(n4)
		-- perform the LSTM update
		local next_c = nn.CAddTable(){
			nn.CMulTable()({forget_gate, prev_c}),
			nn.CMulTable()({in_gate, in_transform})
		}
		-- gated cells form the output
		local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

		table.insert(outputs, next_c)
		table.insert(outputs, next_h)
	end

	local top_h = outputs[#outputs]
	attn = nn.Linear(opt.enc_rnn_size, opt.dec_rnn_size)(attn)
	attn = nn.Sigmoid()(attn)
	top_h = nn.JoinTable(2)({top_h, attn})
	if opt.dropout > 0 then top_h = nn.Dropout(opt.dropout)(top_h) end
	local proj = nn.Linear(2 * opt.dec_rnn_size, opt.tgt_vocab)(top_h)
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)

	return nn.gModule(inputs, outputs)
end

function models.decoder_gru_attn(opt)
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- x
	table.insert(inputs, nn.Identity()()) -- context
	for L = 1, opt.nlayer do
		table.insert(inputs, nn.Identity()()) -- prev_h[L]
	end

	local outputs = {}
	local x = inputs[1]
	local prev_h = inputs[#inputs]

	local attn_h = nn.CAddTable(){
		nn.Linear(opt.emb, opt.dec_rnn_size)(x),
		nn.Linear(opt.dec_rnn_size, opt.dec_rnn_size)(prev_h)
	}
	
	local attention = nil
	if opt.attn_net == 'conv' then
		attention = models.decoder_conv_attn(opt)
	else
		attention = models.decoder_mlp_attn(opt)
	end

	local attn = attention({attn_h, inputs[2]}):annotate{name = 'attn'}

	function new_input_sum(insize, xv, hv, L)
		local i2h = nn.Linear(insize, opt.dec_rnn_size)(xv)
					:annotate{name = 'i2h_' .. L}
		local h2h = nn.Linear(opt.dec_rnn_size, opt.dec_rnn_size)(hv)
					:annotate{name = 'h2h_' .. L}
		local c2h = nn.Linear(opt.dec_rnn_size, opt.dec_rnn_size)(attn)
					:annotate{naem = 'c2h_' .. L}
		return nn.CAddTable()(L == opt.nlayer and {i2h, h2h, c2h} or {i2h, h2h})
	end

	local input_size_L
	for L = 1, opt.nlayer do
		local prev_h = inputs[L + 2]
		if L == 1 then
			x = inputs[1]
			input_size_L = opt.emb
		else
			x = outputs[L - 1]
			if opt.dropout > 0 then x = nn.Dropout(opt.dropout)(x) end 
			input_size_L = opt.dec_rnn_size
		end
		
		-- forward the update and reset gates
		local update_gate = nn.Sigmoid()(
			new_input_sum(input_size_L, x, prev_h, L)
		):annotate{name = 'update_' .. L}
		local reset_gate = nn.Sigmoid()(
			new_input_sum(input_size_L, x, prev_h, L)
		):annotate{name = 'reset_' .. L}

		-- compute the candidate hidden state
		local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
		local p2 = nn.Linear(opt.dec_rnn_size, opt.dec_rnn_size)(gated_hidden)
		local p1 = nn.Linear(input_size_L, opt.dec_rnn_size)(x)
		local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1, p2}))

		-- compute the new interpolated hidden state, based on the update gate
		local zh = nn.CMulTable()({update_gate, hidden_candidate})
		local zhm1 = nn.CMulTable()({
			nn.AddConstant(1, false)(nn.MulConstant(-1, false)(update_gate)),
			prev_h
		})
		local next_h = nn.CAddTable()({zh, zhm1})

		table.insert(outputs, next_h)
	end

	local top_h = outputs[#outputs]
	attn = nn.Linear(opt.enc_rnn_size, opt.dec_rnn_size)(attn)
	attn = nn.Sigmoid()(attn)
	top_h = nn.JoinTable(2)({top_h, attn})
	if opt.dropout > 0 then top_h = nn.Dropout(opt.dropout)(top_h) end
	local proj = nn.Linear(2 * opt.dec_rnn_size, opt.tgt_vocab)(top_h)
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)

	return nn.gModule(inputs, outputs)
end

function models.decoder_conv_attn(opt)
	local inputs = {}
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())

	local target = nn.Linear(
		opt.dec_rnn_size, opt.enc_rnn_size, false)(inputs[1])
	
	local conv = nn.Sequential()
	conv:add(nn.View(1, -1, opt.enc_rnn_size):setNumInputDims(2))
	conv:add(nn.SpatialZeroPadding(0, 0, (opt.pool - 1) / 2, (opt.pool - 1) / 2))
	conv:add(nn.SpatialAveragePooling(1, opt.pool))
	conv:add(nn.View(-1, opt.enc_rnn_size):setNumInputDims(2))

	local context = inputs[2]
	conv_con = conv(context)

	local attn = nn.MM()({conv_con, nn.Replicate(1, 3)(target)})
	attn = nn.Sum(3)(attn)
	attn = nn.SoftMax()(attn)
	attn = nn.Replicate(1, 2)(attn)

	local c = nn.MM()({attn, context})
	c = nn.Sum(2)(c)
	
	local outputs = {c}

	return nn.gModule(inputs, outputs)
end

function models.decoder_mlp_attn(opt)
	local inputs = {}
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())

	local target = nn.Linear(
		opt.dec_rnn_size, opt.enc_rnn_size, false)(inputs[1])
	local context = inputs[2]
	local attn = nn.MM()({context, nn.Replicate(1, 3)(target)})

	attn = nn.Sum(3)(attn)
	attn = nn.SoftMax()(attn)
	attn = nn.Replicate(1, 2)(attn)

	local c = nn.MM()({attn, context})
	c = nn.Sum(2)(c)

	local outputs = {c}

	return nn.gModule(inputs, outputs)
end

return models
