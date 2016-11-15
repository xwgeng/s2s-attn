
local models = require 'models_maker'

local policy = torch.class('PolicyGradient')

function policy:__init(opt)
	self.opt = opt

end

function policy:create_networks(opt)
	local actor = (opt.attn == 1) and models.decoder_rnn_attn1(opt)
					or models.decoder.rnn_attn2(opt)
	local critic = 
end
