require 'gnuplot'

local input = {}

local f = assert(io.open('out', 'r'))
while true do
	local raw = f:read()
	if raw == nil then break end
	input[#input + 1] = tonumber(raw)
end
f:close()

gnuplot.pdffigure('BLEU.pdf')
gnuplot.plot(torch.DoubleTensor(input), '~')
gnuplot.xlabel('epoch')
gnuplot.ylabel('BLEU')

