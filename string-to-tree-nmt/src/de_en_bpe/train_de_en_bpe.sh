#!/bin/sh

base_path=/home/nlp/aharonr6

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$base_path/git/nematus

# train model with nematus
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu0,lib.cnmem=0.8 $nematus/nematus/nmt.py \
--datasets $base_path/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.de.bpe.copy \
$base_path/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.en.bpe \
--dictionaries $base_path/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.de.bpe.json \
$base_path/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.en.parsed2.final.true.bped.final.json \
--model $base_path/git/research/nmt/models/de_en_bpe/de_en_bpe_model.npz \
--saveFreq 30000 \
--validFreq 10000 \
--dim_word 500 \
--dim 1024 \
--valid_datasets $base_path/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-src.tok.true.de.bpe \
$base_path/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.tok.true.en.bpe \
--external_validation_script $base_path/git/research/nmt/src/de_en_bpe/validate.sh \
--optimizer adadelta \
--alpha_c 0.0 \
--clip_c 1.0 \
--maxlen 50 \
--batch_size 80 \
--valid_batch_size 10 \
--reload
