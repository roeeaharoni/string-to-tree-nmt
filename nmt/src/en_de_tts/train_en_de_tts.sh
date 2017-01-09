#!/bin/sh

base_path=/home/nlp/aharonr6

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$base_path/git/nematus

# train model with nematus
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu3,lib.cnmem=0.8 $nematus/nematus/nmt.py \
--datasets $base_path/git/research/nmt/data/WMT16/en-de/train/corpus.parallel.tok.en.parsed2.final.true.bped.final \
$base_path/git/research/nmt/data/WMT16/en-de/train/corpus.parallel.tok.true.de.bpe \
--dictionaries $base_path/git/research/nmt/data/WMT16/en-de/train/corpus.parallel.tok.en.parsed2.final.true.bped.final.json \
$base_path/git/research/nmt/data/WMT16/en-de/train/corpus.parallel.tok.true.de.bpe.json \
--model $base_path/git/research/nmt/models/en_de_tts/en_de_tts_model.npz \
--saveFreq 30000 \
--validFreq 10000 \
--dim_word 500 \
--dim 1024 \
--valid_datasets $base_path/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.tok.true.parsed.en.bped \
$base_path/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-src.tok.true.de.bpe \
--external_validation_script $base_path/git/research/nmt/src/en_de_tts/validate.sh \
--optimizer adadelta \
--alpha_c 0.0 \
--clip_c 1.0 \
--maxlen 150 \
--batch_size 40 \
--valid_batch_size 10 \
--reload
