import codecs
from collections import defaultdict
import os


# TODO:
# write a pipeline that does all:
# tokenize with moses scripts - done
# count words per output sentence - done
# run BPE on both sides -
# extract TSS prefix - done
# write train versions (BPE2BPE): with output length, with output TSS, with both, without anything
# 4 models for 2 language pairs - totals in 8 models, 1 week per model - 8 weeks (one week if takeover)
# can be run in one week on all NLP GPUs / use cyber GPUs?
# start with fr-en TSS, fr-en len
# process is: tokenize -> clean --> truecase --> BPE --> add prefixes --> build dictionaries --> train --> evaluate


MOSES_HOME = '/Users/roeeaharoni/git/mosesdecoder'
BPE_HOME = '/Users/roeeaharoni/git/subword-nmt'
NEMATUS_HOME = '/Users/roeeaharoni/git/nematus'
BPE_OPERATIONS = 89500


def main ():

    return

    TSS = get_TSS_from_tree(
        'TOP (S (NP DT NN )NP (VP VBZ (NP (NP NN )NP (PP IN (NP DT NNP NNP NNP )NP )PP )NP )VP . )S )TOP')

    TSS = get_TSS_prefixes('/Users/roeeaharoni/git/research/nmt/data/shi/Eng_Fre_4M/xing.train.txt.tok.fr',
                         '/Users/roeeaharoni/git/research/nmt/data/shi/Eng_Fre_4M/xing.train.txt.tok.en',
                         '/Users/roeeaharoni/git/research/nmt/data/shi/Eng_Parse_3/9m.train.trainline',
                         '/Users/roeeaharoni/git/research/nmt/data/shi/Eng_Parse_3/9m.train.trainwords',
                         'bla.out')

    All_TSS = list(set([" ".join(s) for s in TSS]))

    return

    fr_en_TSS_exp()

    en_he_len_exp()


    # src_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Parse_3/8m.train.trainwords'
    # target_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Parse_3/8m.train.trainwords'
    # output_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Parse_3/8m.train.TSS.en'
    # syntax_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Parse_3/8m.train.trainline'

    # add_TSS_prefix(src_file, target_file, syntax_file, output_file)


def TODO():
    train_moses_truecase('/Users/roeeaharoni/git/research/nmt/data/WMT16/en-de/train/corpus.parallel.tok.en',
                         '/Users/roeeaharoni/git/research/nmt/models/en-de.en.truecase.model')

    train_moses_truecase('/Users/roeeaharoni/git/research/nmt/data/WMT16/en-de/train/corpus.parallel.tok.de',
                         '/Users/roeeaharoni/git/research/nmt/models/en-de.en.truecase.model')


def bllip_parse(input_file, output_file):
    from bllipparser import RerankingParser
    rrp = RerankingParser.fetch_and_load('WSJ+Gigaword-v2', verbose=True)
    parses = []
    with codecs.open(input_file, encoding='utf8') as sents:
        while True:
            sent = sents.readline()
            parses.append('<s> ' + rrp.simple_parse(sent) + ' </s>')
            if not sent: break  # EOF
    return parses



def fr_en_TSS_exp():

    # get file paths
    prefix = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/xing'

    train_src = prefix + '.train.txt.tok.fr'
    train_target = prefix + '.train.txt.tok.en'

    dev_src = prefix + '.dev.txt.tok.fr'
    dev_target = prefix + '.dev.txt.tok.en'

    test_src = prefix + '.test.txt.tok.fr'
    test_target = prefix + '.test.txt.tok.en'

    in_lang = 'fr'
    out_lang = 'en'

    all_files = [train_src, train_target, dev_src, dev_target, test_src, test_target]

    src_files = [train_src, dev_src, test_src]

    target_files = [train_target, dev_target, test_target]

    # tokenize
    for file in src_files:
        moses_tokenize(file, file + '.tok', in_lang)

    for file in target_files:
        moses_tokenize(file, file + '.tok', out_lang)

    # clean
    for f in ['train', 'dev', 'test']:
        input_corpus_prefix_path = prefix + '.{}.txt.tok'
        output_corpus_prefix_path = prefix + '.{}.txt.tok.clean'
        moses_clean(input_corpus_prefix_path.format(f), output_corpus_prefix_path.format(f), 'fr', 'en')

    train_src_clean = prefix + '.train.txt.tok.clean.fr'
    train_target_clean = prefix + '.train.txt.tok.clean.en'

    # train src truecase
    src_tc_model_path = prefix + '.train.txt.tok.clean.fr.tcmodel'
    src_tc_model_path = train_moses_truecase(train_src_clean, src_tc_model_path)

    # apply src truecase
    for f in ['train', 'dev', 'test']:
        apply_moses_truecase(prefix + '.{}.txt.tok.clean.fr'.format(f),
                             prefix + '.{}.txt.tok.clean.true.fr'.format(f),
                             src_tc_model_path)

    # train target truecase
    trg_tc_model_path = prefix + '.train.txt.tok.clean.en.tcmodel'
    trg_tc_model_path = train_moses_truecase(train_target_clean, trg_tc_model_path)

    # apply target truecase
    for f in ['train', 'dev', 'test']:
        apply_moses_truecase(
            prefix + '.{}.txt.tok.clean.en'.format(f),
            prefix + '.{}.txt.tok.clean.true.en'.format(f),
            trg_tc_model_path)

    # BPE

    # train bpe
    clean_src_train_file = prefix + '.train.txt.tok.clean.true.en'
    clean_target_train_file = prefix + '.train.txt.tok.clean.true.en'
    bpe_model_path = prefix + '.train.txt.tok.clean.true.fr_en.bpe'
    bpe_model_path = train_BPE(clean_src_train_file, clean_target_train_file, BPE_OPERATIONS, bpe_model_path)

    # apply bpe
    for l in ['fr', 'en']:
        for f in ['train', 'dev', 'test']:
            apply_BPE(
                prefix + '.{}.txt.tok.clean.true.{}'.format(f, l),
                prefix + '.{}.txt.tok.clean.true.bpe.{}'.format(f, l),
                bpe_model_path)

    # add len prefixes
    for f in ['train', 'dev', 'test']:
        src_file = prefix + '.{}.txt.tok.clean.true.bpe.fr'.format(f)
        with codecs.open(src_file, encoding='utf8') as src:
            src_lines = src.readlines()

        output_lengths = get_binned_lengths(
            prefix + '.{}.txt.tok.clean.true.en'.format(f))

        new_src_lines = []
        for i, length in enumerate(output_lengths):
            new_src_lines.append('TL{} '.format(length) + src_lines[i])

        output_file = prefix + '.{}.txt.tok.clean.true.bpe.len.fr'.format(f)
        with codecs.open(output_file, 'w', encoding='utf8') as predictions:
            for i, line in enumerate(new_src_lines):
                predictions.write(u'{}'.format(line))

    # add TSS prefixes
    for f in ['train', 'dev', 'test']:

        src_file = prefix + '.{}.txt.tok.clean.true.bpe.fr'.format(f)

        with codecs.open(src_file, encoding='utf8') as src:
            src_lines = src.readlines()

        output_TSS_prefixes = get_TSS_prefixes(
            prefix + '.{}.txt.tok.clean.true.en'.format(f))

        new_src_lines = []
        for i, length in enumerate(output_lengths):
            new_src_lines.append('TL{} '.format(length) + src_lines[i])

        output_file = prefix + '.{}.txt.tok.clean.true.bpe.len.fr'.format(
            f)
        with codecs.open(output_file, 'w', encoding='utf8') as predictions:
            for i, line in enumerate(new_src_lines):
                predictions.write(u'{}'.format(line))

    # build dictionaries

    # train

    # evaluate


def en_he_len_exp():
    src_file = '/Users/roeeaharoni/git/research/nmt/en-he/IWSLT14.TED.dev2010.he-en.en.tok'
    target_file = '/Users/roeeaharoni/git/research/nmt/en-he/IWSLT14.TED.dev2010.he-en.he.tok'
    output_file = '/Users/roeeaharoni/git/research/nmt/en-he/IWSLT14.TED.dev2010.he-en.en.tok.len'
    add_length_prefix(src_file, target_file, output_file)
    return src_file, target_file


def moses_tokenize(input_file_path, output_file_path, lang):
    commandString = '{}/scripts/tokenizer/tokenizer.perl -l {} < {} > {}'.format(MOSES_HOME,
                                                                                 lang,
                                                                                 input_file_path,
                                                                                 output_file_path)
    os.system(commandString)
    return output_file_path


def moses_clean(input_corpus_prefix_path, output_corpus_prefix_path, src_lang, trg_lang):
    command_string = '{}/scripts/training/clean-corpus-n.perl {} {} {} {} 1 80'.format(
        MOSES_HOME,
        input_corpus_prefix_path,
        src_lang,
        trg_lang,
        output_corpus_prefix_path)
    os.system(command_string)


# input is cleaned files
def train_moses_truecase(src_file_path, tc_model_path):
    # train on clean src
    command_string = '{}/scripts/recaser/train-truecaser.perl -corpus {} -model {}'.format(MOSES_HOME,
                                                                                           src_file_path,
                                                                                           tc_model_path)
    os.system(command_string)
    return tc_model_path


def apply_moses_truecase(input_file_path, output_file_path, tc_model_path):

    # apply on clean source
    command_string = '{}/scripts/recaser/truecase.perl -model {} < {} > {}'.format(MOSES_HOME,
                                                                                   tc_model_path,
                                                                                   input_file_path,
                                                                                   output_file_path)
    os.system(command_string)
    return output_file_path

def train_BPE(src_file_path, trg_file_path, bpe_ops, bpe_model_path):

    # apply on clean source
    command_string = 'cat {} {} | {}/learn_bpe.py -s {} > {}'.format(src_file_path,
                                                                     trg_file_path,
                                                                     BPE_HOME,
                                                                     bpe_ops,
                                                                     bpe_model_path)
    os.system(command_string)
    return bpe_model_path


def apply_BPE(input_file_path, output_file_path, bpe_model_path):
    command_string = '{}/apply_bpe.py -c {} < {} > {}'.format(BPE_HOME,
                                                              bpe_model_path,
                                                              input_file_path,
                                                              output_file_path)
    os.system(command_string)
    return output_file_path


def build_nematus_dictionary(train_src_bpe_file_path, train_target_bpe_file_path):

    # build network dictionary
    command_string = '{}/data/build_dictionary.py {} {}'.format(NEMATUS_HOME,
                                                                train_src_bpe_file_path,
                                                                train_target_bpe_file_path)
    os.system(command_string)


def get_TSS_from_tree(tree):

    tokens = tree.split()
    depth = 0
    depth2labels = {}
    for token in tokens:
        if depth not in depth2labels:
            depth2labels[depth] = []

        if '(' in token:
            depth2labels[depth].append(token.replace('(',''))
            depth += 1

        if ')' in token:
            depth -=1

    return depth2labels[2]



def get_TSS_prefixes(src_file, target_file, trees_file, sentences_file, output_file):

    # Top-level Syntactic Sequence:

    #####NP#### #################VP##################

    # POS tags:

    #  DT   NN  VBZ NN  IN  DT   NNP     NNP    NNP

    # sentence:

    # This site is part of The Imaging Source Network .

    # constituency tree:

    # TOP (S (NP DT NN )NP (VP VBZ (NP (NP NN )NP (PP IN (NP DT NNP NNP NNP )NP )PP )NP )VP . )S )TOP

    # to visualize: (S (NP DT NN ) (VP VBZ (NP (NP NN ) (PP IN (NP DT NNP NNP NNP ) ) ) ) . )

    # should return:

    # ['NP', 'VP']

    new_src_lines = []
    text2tree = {}

    # read syntax trees and matching sentences
    with codecs.open(sentences_file, encoding='utf8') as sents:
        with codecs.open(trees_file, encoding='utf8') as trees:
            i = 0
            print 'reading parsed sentences and matching trees...'
            while True and i < 1000:
                i += 1
                sent = sents.readline()
                tree = trees.readline()
                text2tree[sent] = tree
                if not sent: break  # EOF
                # sent_lines = sent.readlines()
                # sent_lines = [next(sent) for x in xrange(600000)]
            # tree_lines = trees.readlines()
            # tree_lines = [next(trees) for x in xrange(600000)]
            # for i, treeline in enumerate(tree_lines):
            #     text2tree[sent_lines[i]] = treeline

    TSS_strings = []
    TSS_with_lex_trees = []
    TSS_with_clean_trees = []
    text2lextree = {}
    for text in text2tree:

        # get the raw tree
        TSS = get_TSS_from_tree(text2tree[text])

        # clean the tree for visualization
        clean_tree = []
        for s in text2tree[text].split():
            if ')' in s:
                clean_tree.append(')')
            else:
                clean_tree.append(s)

        # clean the tree and add words for visualization
        lex_tree = []
        words = text.split()
        i = 0
        for s in text2tree[text].split():
            if ')' in s:
                lex_tree.append(')')
            else:
                if '(' in s:
                    lex_tree.append(s)
                else:
                    # pos, replace with word
                    lex_tree.append('(' + s)
                    lex_tree.append(words[i])
                    lex_tree.append(')')
                    i += 1

        text2lextree[text] = ' '.join(lex_tree)

        TSS_with_lex_trees.append(' '.join(TSS) + '_____' + ' '.join(lex_tree))
        TSS_strings.append(' '.join(TSS))
        TSS_with_clean_trees.append(' '.join(clean_tree))

    from collections import Counter

    TSS_counter = Counter(TSS_strings)
    x = TSS_counter.most_common()
    TSS_distinct = list(set(TSS_strings))

    # read parallel data
    with codecs.open(src_file, encoding='utf8') as src:
        with codecs.open(target_file, encoding='utf8') as trgt:
            print 'reading source sentences...'
            # src_lines = src.readlines()
            src_lines = [next(src) for x in xrange(100000)]

            print 'reading target sentences...'
            # trgt_lines = trgt.readlines()
            trgt_lines = [next(trgt) for x in xrange(100000)]

    print 'looking for trees...'
    found = 0
    total = 0
    TSS_strings = []
    for i, line in enumerate(trgt_lines):
        total += 1
        if line in text2tree:
            # found the tree for the sentence
            found += 1
            print text2tree[line], '\n', line, '\n\n'

            tree = text2tree[line]
            TSS = get_TSS_from_tree(tree)
            TSS_strings.append(' '.join(TSS))

    print 'found trees for {} out of {} sentences'.format(found, total)
    return TSS_strings


def add_length_prefix(src_file, target_file, output_file):
    len_histogram = defaultdict(int)
    binned_histogram = defaultdict(int)
    BIN_SIZE = 5
    binned_lens = []
    new_src_lines = []
    with codecs.open(src_file, encoding='utf8') as src:
        with codecs.open(target_file, encoding='utf8') as trgt:
            print 'tokenizing...'
            src_lines = src.readlines()
            trgt_lines = trgt.readlines()
            for i, line in enumerate(trgt_lines):
                tokens = line.split()
                length = len(tokens)
                len_histogram[length] = len_histogram[length] + 1

                # compute bin
                if length % BIN_SIZE == 0:
                    binned_length = length
                else:
                    # if not in bin go to the next bin
                    binned_length = length - length % BIN_SIZE + BIN_SIZE
                binned_histogram[binned_length] += 1
                new_src_lines.append('TL{} '.format(binned_length) + src_lines[i])
                binned_lens.append(binned_length)


    for i in xrange(100):
        print u'{}\n{}'.format(new_src_lines[i], trgt_lines[i])
    with codecs.open(output_file, 'w', encoding='utf8') as predictions:
        for i, line in enumerate(new_src_lines):
            predictions.write(u'{}'.format(line))
    for key in len_histogram:
        print key, '\t', len_histogram[key]
    for key in binned_histogram:
        print key, '\t', binned_histogram[key]


def get_binned_lengths(target_file, bin_size = 5):
    binned_lens = []

    with codecs.open(target_file, encoding='utf8') as trgt:
        trgt_lines = trgt.readlines()
        for i, line in enumerate(trgt_lines):
            tokens = line.split()
            length = len(tokens)

            # compute bin
            if length % bin_size == 0:
                binned_length = length
            else:
                # if not in bin go to the next bin
                binned_length = length - length % bin_size + bin_size
            binned_lens.append(binned_length)

    for i in xrange(10):
        print 'sanity check:'
        print 'len is {} for: {}'.format(binned_lens[i], trgt_lines[i])
    return binned_lens




if __name__ == '__main__':
    main()