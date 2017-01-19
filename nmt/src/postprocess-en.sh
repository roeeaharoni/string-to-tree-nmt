#!/bin/sh

base_path=/home/nlp/aharonr6
#base_path=~/

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=$base_path/git/mosesdecoder

# suffix of target language files
lng=en

# fix BPE split words
sed 's/\@\@ //g' | \


# detruecase
$mosesdecoder/scripts/recaser/detruecase.perl | \

# detokenize
$mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng | \

# fix quotation marks
sed 's/``/"/g' | \
sed 's/ '\'''\''/" /g' | \

# fix braces
sed 's/-LRB- /(/g' | \
sed 's/ -RRB-/)/g' | \

# fix @/@
sed 's/ @\/@ /\//g'
