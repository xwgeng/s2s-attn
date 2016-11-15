## Sequece-to-Sequence Model with attention

Sequence-to-sequence model with attention implemented by [Torch](http://torch.ch).
The encoder can be bidirectional recurrent neural network(LSTM | GRU | RNN). Additionally, the convolutional attentive encoder([Rush et al.](https://www.aclweb.org/anthology/D/D15/D15-1044.pdf)) is also provided, inspired by [Bahdanau et al.](https://arxiv.org/pdf/1409.0473v7.pdf). 

### Dependencies

#### Lua
* Moses
* Penlight

#### Torch
The model is implemented by [torch](http://torch.ch). It requires the following packages:
* torch7
* nn
* nngraph
* cutorch
* cunn
* paths

### Quikstart

#### preprocess

* First, I pre-process the training data using the tokenizer of the Moses toolkit with the script `nmt-prep.sh` in `script/` folder.
```
sh nmt-prep.sh
```

* And then, prepare the data with `data_prep.lua` which transform the data into tensor.
```
th data_prep.lua -src_path 'data/nmt/prep' -dst_path 'data/nmt/data' -src_train 'train.de-en.de' -src_valid 'valid.de-en.de' -src_test 'test.de-en.de' -tgt_train 'train.de-en.en' -tgt_valid 'valid.de-en.en' tgt_test 'test.de-en.en' -min_freq 3 -seed 123
```
