taken from http://data.statmt.org/rsennrich/wmt16_factors/de-en/

The German corpus was tokenized with Moses scripts, and parsed as follows:

mosesdir = /home/rsennrich/tools/mosesdecoder
parzu-path = /home/rsennrich/ParZu # https://github.com/rsennrich/ParZu

$mosesdir/scripts/tokenizer/normalize-punctuation.perl de \
  | $mosesdir/scripts/tokenizer/tokenizer.perl -l de \
  | $mosesdir/scripts/tokenizer/deescape-special-chars.perl \
  | $parzu-path/parzu -i tokenized_lines --projective


The English side of the training corpus is provided for consistency in sentence alignment.
It has been tokenized as follows:

$mosesdir/scripts/tokenizer/normalize-punctuation.perl en \
  | $mosesdir/scripts/tokenizer/tokenizer.perl -l en -penn


The synthetic training corpus was produced via back-translating monolingual target-side data. See also http://data.statmt.org/rsennrich/wmt16_backtranslations/
Rico Sennrich, Barry Haddow, Alexandra Birch (2016): Improving Neural Machine Translation Models with Monolingual Data. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.


The parse trees for the .en part were parsed using the bllip parser with WSJ+GIGAWORD option by xing shi

looked for 4200000 trees from training corpora, found 3263515 so far
looked for 4215807 trees from training corpora, found 3548089 total
looked for 4215807 trees from training corpora, found 3603547 total
looked for 4215807 trees from training corpora, found 3762724 total
looked for 4215807 trees from training corpora, found 3771838 total
looked for 4215807 trees from training corpora, found 3772210 total
looked for 4215807 trees from training corpora, found 3851277 total
looked for 4215807 trees from training corpora, found 3986938 total
looked for 4215807 trees from training corpora, found 4026537 total
looked for 4215807 trees from training corpora, found 4085997 total
looked for 4215807 trees from training corpora, found 4171977 total
looked for 4215807 trees from training corpora, found 4180027 total
looked for 4215807 trees from training corpora, found 4181486 total



