#!/bin/sh

base_path=/home/nlp/aharonr6

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$base_path/git/nematus

model_name="cs_en_bpe_v8"

# train model with nematus
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu2,lib.cnmem=0.8 $nematus/nematus/nmt.py \
--datasets $base_path/git/research/nmt/data/news-cs-en/train/news-commentary-v8.cs-en.tok.penntrg.clean.true.bpe.cs \
           $base_path/git/research/nmt/data/news-cs-en/train/news-commentary-v8.cs-en.tok.penntrg.clean.true.bpe.en \
--dictionaries $base_path/git/research/nmt/data/news-cs-en/train/news-commentary-v8.cs-en.tok.penntrg.clean.true.bpe.cs.json \
               $base_path/git/research/nmt/data/news-cs-en/train/news-commentary-v8.cs-en.tok.penntrg.clean.true.bpe.en.json \
--model $base_path/git/research/nmt/models/$model_name/${model_name}_model.npz \
--saveFreq 10000 \
--validFreq 5000 \
--dim_word 256 \
--dim 256 \
--valid_datasets $base_path/git/research/nmt/data/news-cs-en/dev/newstest2015-csen.tok.penntrg.clean.true.bpe.cs \
                 $base_path/git/research/nmt/data/news-cs-en/dev/newstest2015-csen.tok.penntrg.clean.true.bpe.en \
--external_validation_script $base_path/git/research/nmt/src/$model_name/validate.sh \
--optimizer adadelta \
--lrate 0.0001 \
--clip_c 1.0 \
--dropout_embedding 0.2 \
--dropout_hidden 0.2 \
--dropout_source 0.1 \
--dropout_target 0.1 \
--maxlen 400 \
--batch_size 40 \
--valid_batch_size 80 \
--reload

