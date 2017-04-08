#!/bin/sh

base_path=/home/nlp/aharonr6
export PYTHONPATH=$base_path/git/research/nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$base_path/git/nematus
model_name=de_en_stt_v8_small_vocab

# train model with nematus
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu0,lib.cnmem=0.8 $nematus/nematus/nmt.py \
--datasets $base_path/git/research/nmt/data/news-de-en/train/small_vocab/news-commentary-v8.de-en.tok.penntrg.clean.true.bpe.de.copy \
           $base_path/git/research/nmt/data/news-de-en/train/small_vocab/news-commentary-v8.de-en.tok.penntrg.clean.true.desc.parsed.linear.bpe.en \
--dictionaries $base_path/git/research/nmt/data/news-de-en/train/small_vocab/news-commentary-v8.de-en.tok.penntrg.clean.true.bpe.de.json \
               $base_path/git/research/nmt/data/news-de-en/train/small_vocab/news-commentary-v8.de-en.tok.penntrg.clean.true.desc.parsed.linear.bpe.en.json \
--model $base_path/git/research/nmt/models/$model_name/${model_name}_model.npz \
--saveFreq 10000 \
--validFreq 5000 \
--dim_word 256 \
--dim 256 \
--valid_datasets $base_path/git/research/nmt/data/news-de-en/dev/small_vocab/newstest2015-deen.tok.penntrg.clean.true.bpe.de \
                 $base_path/git/research/nmt/data/news-de-en/dev/small_vocab/newstest2015-deen.tok.penntrg.clean.true.desc.parsed.linear.bpe.en \
--external_validation_script $base_path/git/research/nmt/src/$model_name/validate_trees.sh \
--optimizer adadelta \
--lrate 0.0001 \
--alpha_c 0.0 \
--clip_c 1.0 \
--dropout_embedding 0 \
--dropout_hidden 0 \
--dropout_source 0 \
--dropout_target 0 \
--maxlen 150 \
--batch_size 40 \
--valid_batch_size 10 \
--reload
