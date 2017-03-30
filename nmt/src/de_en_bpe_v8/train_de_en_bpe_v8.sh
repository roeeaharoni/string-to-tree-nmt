#!/bin/sh

base_path=/home/nlp/aharonr6

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$base_path/git/nematus

# train model with nematus
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu1,lib.cnmem=0.8 $nematus/nematus/nmt.py \
--datasets $base_path/git/research/nmt/data/news-de-en/train/news-commentary-v8.de-en.tok.penntrg.clean.true.bpe.de \
$base_path/git/research/nmt/data/news-de-en/train/news-commentary-v8.de-en.tok.penntrg.clean.true.bpe.en \
--dictionaries $base_path/git/research/nmt/data/news-de-en/train/news-commentary-v8.de-en.tok.penntrg.clean.true.bpe.de.json \
$base_path/git/research/nmt/data/news-de-en/train/news-commentary-v8.de-en.tok.penntrg.clean.true.bpe.en.json \
--model $base_path/git/research/nmt/models/de_en_bpe_v8/de_en_bpe_v8_model.npz \
--saveFreq 30000 \
--validFreq 10000 \
--dim_word 500 \
--dim 1024 \
--valid_datasets $base_path/git/research/nmt/data/news-de-en/dev/newstest2015-deen.tok.penntrg.clean.true.bpe.de \
$base_path/git/research/nmt/data/news-de-en/dev/newstest2015-deen.tok.penntrg.clean.true.bpe.en \
--external_validation_script $base_path/git/research/nmt/src/de_en_bpe_v8/validate.sh \
--optimizer adadelta \
--lrate 0.0001 \
--alpha_c 0.0 \
--clip_c 1.0 \
--dropout_embedding 0 \
--dropout_hidden 0 \
--dropout_source 0 \
--dropout_target 0 \
--maxlen 50 \
--batch_size 40 \
--valid_batch_size 80 \
--reload
