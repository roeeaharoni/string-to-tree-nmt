#!/bin/sh

# this sample script translates a test set, including
# preprocessing (tokenization, truecasing, and subword segmentation),
# and postprocessing (merging subword units, detruecasing, detokenization).

# instructions: set paths to mosesdecoder, subword_nmt, and nematus,
# then run "./translate.sh < input_file > output_file"

# suffix of source language
SRC=de

# suffix of target language
TRG=en

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/home/nlp/aharonr6/git/mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/home/nlp/aharonr6/git/subword-nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/nlp/aharonr6/git/nematus

# theano device
device=gpu3

# preprocess
$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
$mosesdecoder/scripts/tokenizer/tokenizer.perl -l $SRC | \
$mosesdecoder/scripts/recaser/truecase.perl -model truecase-model.$SRC | \
$subword_nmt/apply_bpe.py -c $SRC$TRG.bpe | \
# translate
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn,lib.cnmem=0.09 python $nematus/nematus/translate.py \
     -m model.npz \
     -k 12 -n -p 1 --suppress-unk | \
# postprocess
sed 's/\@\@ //g' | \
$mosesdecoder/scripts/recaser/detruecase.perl | \
$mosesdecoder/scripts/tokenizer/detokenizer.perl -l $TRG -penn
