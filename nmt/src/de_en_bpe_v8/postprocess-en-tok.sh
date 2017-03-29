#!/bin/sh

base_path=/home/nlp/aharonr6
#base_path=~/

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$base_path/git/mosesdecoder

# fix BPE split words
sed 's/\@\@ //g' | \


# detruecase
$mosesdecoder/scripts/recaser/detruecase.perl
