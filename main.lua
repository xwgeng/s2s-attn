require 'torch'
require 'nn'
require 'paths'

require 'data_loader'
require 'seq2seq'
require 'trainer'

cmd = torch.CmdLine()
cmd:text()
cmd:text('train seq2seq with attention model')
cmd:text()
-- data
cmd:option('-data', 'data/nmt/data', 'path to the training data')
cmd:option('-src_dict', 'src.dict.t7', 'the filename of source dictionary')
cmd:option('-tgt_dict', 'tgt.dict.t7', 'the filename of target dictionary')
cmd:option('-thresh', 800, 'the minimum length of shard')
cmd:option('-reverse', true, 'reverse the source sequence')
cmd:option('-shuff', true, 'shuffle the sentences in trainging data or not')
cmd:option('-curriculum', 1, 'curriculum learning before this epoch')
-- model
cmd:option('-model', '', 'initialize the model')
cmd:option('-emb', 256, 'the dim of embedding')
cmd:option('-enc_rnn_size', 256, 'the number of hidden units')
cmd:option('-dec_rnn_size', 256, 'the number of hidden units')
cmd:option('-rnn', 'lstm', 'recurrent unit: rnn | gru | lstm')
cmd:option('-nlayer', 1, 'the number of layers')
cmd:option('-attn_net', 'conv', 'the network of attention: conv | mlp')
cmd:option('-pool', 5, 'the convolution window')
-- optimization
cmd:option('-optim', 'adam', 'the optimization algorithm')
cmd:option('-dropout', 0, 'dropout rate')
cmd:option('-learningRate', 1e-3, 'the learning rate')
cmd:option('-minLearningRate', 1e-4, 'the minimum learning rate')
cmd:option('-shrink_factor', 1.2, 'the shrink factor of learning rate')
cmd:option('-shrink_multiplier', 0.9999, 'the shrink multiplier')
cmd:option('-anneal', false, 'anneal the learning rate to minLearningRate')
cmd:option('-start_epoch', 0, 'learning rate decays to the minLearningRate')
cmd:option('-saturate_epoch', 30, 'learning rate decays to the minLearningRate')
cmd:option('-batch_size', 32, 'the size of mini-batch')
cmd:option('-src_seq_len', 52, 'the maximum length of source sequences')
cmd:option('-tgt_seq_len', 25, 'the maximum length of target sequences')
cmd:option('-grad_clip', 5, 'clip gradients at this value')
cmd:option('-nepoch', 40, 'the maximum of epoches')
-- bookkeeping
cmd:option('-save', 'backup/nmt', 'path to save the model')
cmd:option('-name', 'vanilla', 'the optional identifier of model')
cmd:option('-seed', 123, 'torch manual random number generator seed')
-- GPU
cmd:option('-cuda', true, 'whether or not use cuda')
cmd:option('-gpu', 0, 'which gpu to use. -1 = use cpu')
-- misc
cmd:option('-nprint', 10, 'the frequency of print')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

if opt.cuda and opt.gpu >=0 then
	require 'cutorch'
	require 'cunn'
	cutorch.setDevice(opt.gpu + 1)
	cutorch.manualSeed(opt.seed)
else
	opt.cuda = false
end

-- load dict
local src_dict = assert(torch.load(paths.concat(opt.data, opt.src_dict)))
local tgt_dict = assert(torch.load(paths.concat(opt.data, opt.tgt_dict)))
opt.src_vocab = #src_dict.i2w
opt.tgt_vocab = #tgt_dict.i2w 
opt.src_pos = 200
opt.src_dict = src_dict
opt.tgt_dict = tgt_dict

-- load datasets
local train = DataLoader({
	batch_size = opt.batch_size,
	src_seq_len = opt.src_seq_len,
	tgt_seq_len = opt.tgt_seq_len,
	src_pad = src_dict.w2i[src_dict.PAD],
	src_eos = src_dict.w2i[src_dict.EOS],
	tgt_pad = tgt_dict.w2i[tgt_dict.PAD],
	tgt_eos = tgt_dict.w2i[tgt_dict.EOS],
	thresh = opt.thresh,
	shuff = opt.shuff,
	path = opt.data,
	label = 'train',
})

local valid = DataLoader({
	batch_size = opt.batch_size,
	src_seq_len = opt.src_seq_len,
	tgt_seq_len = opt.tgt_seq_len,
	src_pad = src_dict.w2i[src_dict.PAD],
	src_eos = src_dict.w2i[src_dict.EOS],
	tgt_pad = tgt_dict.w2i[tgt_dict.PAD],
	tgt_eos = tgt_dict.w2i[tgt_dict.EOS],
	thresh = opt.thresh,
	shuff = opt.shuff,
	path = opt.data,
	label = 'valid',
})

-- create seq2seq model
local model = nil
if opt.model ~= '' then
	model = assert(Seq2seq.load(paths.concat(opt.save, opt.model)))
else
	model = Seq2seq(opt)
end

opt.optim_config = {}
opt.optim_config.learningRate = opt.learningRate	

-- create trainer
local trainer = Trainer(model)

trainer:run(train, valid, opt)
