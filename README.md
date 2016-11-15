## Sequece-to-Sequence Model with attention

Sequence-to-sequence model with attention implemented by [Torch](http://torch.ch).
The encoder can be bidirectional recurrent neural network(LSTM | GRU | RNN). Additionally, the convolutional attentive encoder([Rush et al.](https://www.aclweb.org/anthology/D/D15/D15-1044.pdf)) is also provided, inspired by [Bahdanau et al.](https://arxiv.org/pdf/1409.0473v7.pdf) 

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

* And then, prepare the data with `data_prep.lua` which transforms the data into tensor.
```
th data_prep.lua 
```

#### train
Now, start to train the model
```
th main.lua -learningRate 0.001 -optim 'adam' -dropout 0.2 -rnn 'gru'
```
This will run the model, which uses convolutional attentive encoder and a 1-layer GRU decoder with 256 hidden units.

#### evaluate

* Given a trained model, use beam search to obtain the output. Additionally, greedy search is also provided for efficiency. To do this, just run as follows:
```
th evluate.lua -search 'greedy' -batch_size 32
```

* I use the `muti-bleu.perl` script from Moses to compute the BLEU. Given the *identifier* of model, just run as follows:
```
sh nmt-eval.sh '*identifier*'
```


