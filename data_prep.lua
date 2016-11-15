require 'paths'

local maker = require 'data_maker'

cmd = torch.CmdLine()
cmd:text()
cmd:text('preprocess the corpus')
cmd:text()
cmd:text('Options')
cmd:option('-src_path', 'data/nmt/prep', 'path to pre-processed data')
cmd:option('-dst_path', 'data/nmt/data', 'path to where dictionaries and datasets should be written')
cmd:option('-src_train', 'train.de-en.de', 'the name of source training data')
cmd:option('-src_valid', 'valid.de-en.de', 'the name of source valid data')
cmd:option('-src_test', 'test.de-en.de', 'the name of source testing data')
cmd:option('-tgt_train', 'train.de-en.en', 'the name of target training data')
cmd:option('-tgt_valid', 'valid.de-en.en', 'the name of target valid data')
cmd:option('-tgt_test', 'test.de-en.en', 'the name of target test data')
cmd:option('-min_freq', 3, 'remove words appearing less than min_freq')
cmd:option('-seed', 123, 'torch manual random number generator seed')
cmd:text()

local opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

if not paths.dirp(opt.dst_path) then
	os.execute('mkdir -p ' .. opt.dst_path)
end


local src_train = paths.concat(opt.src_path, opt.src_train)
local src_valid = paths.concat(opt.src_path, opt.src_valid)
local src_test = paths.concat(opt.src_path, opt.src_test)
local tgt_train = paths.concat(opt.src_path, opt.tgt_train)
local tgt_valid = paths.concat(opt.src_path, opt.tgt_valid)
local tgt_test = paths.concat(opt.src_path, opt.tgt_test)

local sdict_path = paths.concat(opt.dst_path, 'src.dict.t7')
local tdict_path = paths.concat(opt.dst_path, 'tgt.dict.t7')

print('building source dictionary ...')
local sdict = maker.dict(src_train, opt.min_freq)
torch.save(sdict_path, sdict)

print('building target dictionary ...')
local tdict = maker.dict(tgt_train, opt.min_freq)
torch.save(tdict_path, tdict)

local train_path = paths.concat(opt.dst_path, 'train.t7')
local valid_path = paths.concat(opt.dst_path, 'valid.t7')
local test_path = paths.concat(opt.dst_path, 'test.t7')

print('coverting training data ...') 
local train = maker.convert(src_train, tgt_train, sdict, tdict)
torch.save(train_path, train)

print('converting valid data ...')
local valid = maker.convert(src_valid, tgt_valid, sdict, tdict)
torch.save(valid_path, valid)

print('converting testing data ..')
local test = maker.convert(src_test, tgt_test, sdict, tdict)
torch.save(test_path, test)
