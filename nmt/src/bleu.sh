#!/bin/sh

base_path=/home/nlp/aharonr6

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$base_path/git/nematus

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$base_path/git/mosesdecoder

# model prefix
model_prefix=$base_path/git/research/nmt/models/de_en_stt/de_en_stt_model.npz

# dev ref file
ref=$base_path/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.en

dev_target=$base_path/git/research/nmt/models/de_en_stt/newstest2015-deen-src.tok.true.de.bpe.output.sents.dev.postprocessed

# get prev best BLEU
BEST=`cat ${model_prefix}_best_bleu || echo 0`
echo "prev best BLEU: $BEST"

# write current BLEU
$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev_target >> ${model_prefix}_bleu_scores
echo "wrote current BLEU to: ${model_prefix}_bleu_scores"

# extract current BLEU
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $dev_target | cut -f 3 -d ' ' | cut -f 1 -d ','`
echo "extracted current BLEU: $BLEU"

# check if to save new model as best model
echo 'check if better...'
echo "current: $BLEU"
echo "best: $BEST"

if (( $(echo "$BLEU > $BEST" |bc -l) )); then
    echo "new best; saving"
    echo $BLEU > ${model_prefix}_best_bleu
    cp ${model_prefix}.dev.npz ${model_prefix}.npz.best_bleu
    echo "improved! saved best in: ${model_prefix}.npz.best_bleu"
else
    echo "no improvement"
fi
