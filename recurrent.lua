local recurrent = {}

function recurrent.lstm(
	input_size, rnn_size, n, dropout, output_size, context_size)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local context = nil
  if context_size then
	  table.insert(inputs, nn.Identity()())
	  context = inputs[#inputs]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local c2h = nil
	if context_size and L == n then
		c2h = nn.Linear(context_size, 4 * rnn_size)(context):annotate{name='c2h_'..L}
	end

    local all_input_sums = nn.CAddTable()(
		not (context_size and L == n) and {i2h, h2h} or {i2h, h2h, c2h}
	)

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  if output_size then
	-- set up the decoder
	local top_h = outputs[#outputs]
	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
	local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)
  end

  return nn.gModule(inputs, outputs)
end

function recurrent.gru(
	input_size, rnn_size, n, dropout, output_size, context_size)
  dropout = dropout or 0 
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local context = nil
  if context_size then
	  table.insert(inputs, nn.Identity()())
  end

  function new_input_sum(insize, xv, hv, cv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
	local c2h = nil
	if cv then c2h = nn.Linear(context_size, rnn_size)(cv) end
    return nn.CAddTable()(cv and {i2h, h2h, c2h} or {i2h, h2h})
  end


  local x, input_size_L
  local outputs = {}
  for L = 1,n do

    local prev_h = inputs[L+1]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

	if L == n and context_size then context = inputs[#inputs] end

    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(
		new_input_sum(input_size_L, x, prev_h, context)
	)
    local reset_gate = nn.Sigmoid()(
		new_input_sum(input_size_L, x, prev_h, context)
	)
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(rnn_size, rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end
  if output_size then
	-- set up the decoder
	local top_h = outputs[#outputs]
	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
	local proj = nn.Linear(rnn_size, output_size)(top_h)
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)
  end

nngraph.annotateNodes()
	nngraph.setDebug(false)
  return nn.gModule(inputs, outputs)
end

function recurrent.rnn(
	input_size, rnn_size, n, dropout, output_size, context_size)
  
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local context = nil
  if context_size then
	table.insert(inputs, nn.Identity()())
	context = inputs[#inputs]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    
    local prev_h = inputs[L+1]
    if L == 1 then 
      x = inputs[1]
      input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
	local c2h = nil
	if L == n and context_size then
		c2h = nn.Linear(context_size, rnn_size)(context)
	end
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h, c2h})

    table.insert(outputs, next_h)
  end

  if output_size then
  -- set up the decoder
  local top_h = outputs[#outputs]
	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
	local proj = nn.Linear(rnn_size, output_size)(top_h)
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)
  end

  return nn.gModule(inputs, outputs)
end

return recurrent
