#!/bin/sh

base_path=/home/nlp/aharonr6

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$base_path/git/nematus

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$base_path/git/mosesdecoder

# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu

# model prefix
model_prefix=$base_path/git/research/nmt/models/de_en_stt_model.npz

# dev ref file
ref=$base_path/git/research/nmt/data/WMT16/en-de/dev/newstest2015-deen-ref.en.100

dev_target=$base_path/git/research/nmt/models/newstest2015-deen-src.tok.true.de.bpe.100.output.sents.dev.postprocessed

# get prev best BLEU
BEST=`cat ${model_prefix}_best_bleu` || echo 0
echo "got prev best BLEU: $BEST"

# write current BLEU
$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev_target >> ${model_prefix}_bleu_scores
echo "wrote current BLEU to ${model_prefix}_bleu_scores"

# extract current BLEU
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev_target | cut -f 3 -d ' ' | cut -f 1 -d ','`
echo 'extracted current BLEU: $BLEU'


echo 'check if better...'
if [ "$BLEU" -gt "$BEST" ]; then
    echo "new best; saving"
    echo $BLEU > ${model_prefix}_best_bleu
    cp ${model_prefix}.dev.npz ${model_prefix}.npz.best_bleu
else
    echo "current: $BLEU"
    echo "best: $BEST"
    echo "no improvement"
fi
