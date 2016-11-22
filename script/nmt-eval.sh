#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=${2:-0}

evaluate=evaluate.lua
search=greedy
batch_size=32
identifier=$1

echo "the configuration of evaluating is as follows:"
echo -e "\tsearch = ${search}"
echo -e "\tbatch_size = ${batch_size}"
echo -e "\tidentifier = ${identifier}"
echo "--------"

cd ..

models=backup/nmt

echo 'starting to evaluate the models ...'

for m in `ls -rt $models/model*.t7 |\
		  grep $identifier`;
do
	fname=${m##*/}
	echo "evaluating the model: ${fname} "
	th $evaluate -model $fname -search $search \
		-batch_size $batch_size -name $identifier
	echo "--------"
done

output=output/nmt
bleu=script/multi-bleu.perl
gold=data/nmt/prep/test.de-en.en

echo 'starting to compute the BLEU ...'

for out in `ls -rt $output/* |\
			grep $identifier`;
do
	fname=${out##*/}
	echo "compuating the BLEU of file: ${fname}"
	tmp=`mktemp`
	cat $out | sort -k1n |\
	   	awk '{print substr($0, length($1)+2)}' > $tmp
	perl $bleu $gold < $tmp | tee -a $output/$identifier.BLEU
	echo "--------"
done
