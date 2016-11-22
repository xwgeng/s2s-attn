require 'torch'
require 'nn'
require 'paths'

require 'data_loader'
require 'seq2seq'
require 'tester'

cmd = torch.CmdLine()
cmd:text()
cmd:text('test the seq2seq with attention model')
cmd:text()
-- data
cmd:option('-data', 'data/nmt/data', 'path to the testing data')
cmd:option('-src_dict', 'src.dict.t7', 'the filename of source dictionary')
cmd:option('-tgt_dict', 'tgt.dict.t7', 'the filename of target dictionary')
cmd:option('-thresh', 0, 'the minimum length of shard')
cmd:option('-reverse', true, 'reverse the source sequence')
-- model
cmd:option('-model', '', 'which model to use')
-- search strategy
cmd:option('-search', 'beam', 'the search strategy: beam | greedy')
-- greedy search
cmd:option('-batch_size', 1, 'the batch size for greedy search')
-- beam search
cmd:option('-beam_size', 5, 'the beam size')
cmd:option('-src_seq_len', 200, 'maximum sequence length to be generated')
cmd:option('-tgt_seq_len', 200, 'maximum sequence length to be generated')
cmd:option('-strategy', true, 'whether or not select the best sequence with eos')
cmd:option('-nbest', false, 'whether or not output the n best list')
-- bookkeeping
cmd:option('-save', 'backup/nmt', 'path to save the model')
cmd:option('-output', 'output/nmt', 'path to save the output')
cmd:option('-name', 'vanilla', 'the optional identifier of output')
cmd:option('-seed', 123, 'torch manual random number generator seed')
-- GPU
cmd:option('-cuda', true, 'whether or not use cuda')
cmd:option('-gpu', 0, 'which gpu to use. -1 = use cpu')
-- misc
cmd:option('-nprint', 10, 'the frequency of print')
cmd:text()

-- parse the input params
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

if opt.cuda and opt.gpu >= 0 then
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

opt.src_pad = src_dict.w2i[src_dict.PAD]
opt.src_eos = src_dict.w2i[src_dict.EOS] 
opt.tgt_pad = tgt_dict.w2i[tgt_dict.PAD]
opt.tgt_eos = tgt_dict.w2i[tgt_dict.EOS]
-- load datasets
local test = DataLoader({
	batch_size = opt.batch_size,
	src_seq_len = opt.src_seq_len,
	tgt_seq_len = opt.tgt_seq_len,
	src_pad = opt.src_pad,
	src_eos = opt.src_eos,
	tgt_pad = opt.tgt_pad,
	tgt_eos = opt.tgt_eos,
	thresh = opt.thresh,
	path = opt.data,
	label = 'test',
})

-- load the seq2seq model
local model = assert(Seq2seq.load(paths.concat(opt.save, opt.model)))

-- create tester
local tester = Tester(model, src_dict, tgt_dict)
tester:run(test, opt)
