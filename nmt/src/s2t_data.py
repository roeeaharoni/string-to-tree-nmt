# -*- coding: utf-8 -*-

"""code for processing corpora for string-to-tree nmt"""

import codecs
from collections import defaultdict
import os
import re
import time

import yoav_trees
import docopt as do
import apply_bpe as apply_bpe
import traceback
from moses_tools import *

from multiprocessing import Pool


# TTS TODO: use amunmt decoder? nice to have
# TTS TODO: check how to manipulate decoder in nematus


# BASE_PATH = '/Users/roeeaharoni'
BASE_PATH = '/home/nlp/aharonr6'
# BASE_PATH = '~'
MOSES_HOME = BASE_PATH + '/git/mosesdecoder'
BPE_HOME = BASE_PATH + '/git/subword-nmt'
NEMATUS_HOME = BASE_PATH + '/git/nematus'
BPE_OPERATIONS = 89500


def main():
    train_prefix = BASE_PATH + '/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train'
    dev_prefix = BASE_PATH + '/git/research/nmt/data/WMT16/de-en-raw/dev/newstest-2013-2014-deen'
    test2015_prefix = BASE_PATH + '/git/research/nmt/data/WMT16/de-en-raw/test/newstest2015-deen'
    test2016_prefix = BASE_PATH + '/git/research/nmt/data/WMT16/de-en-raw/test/newstest-2016-deen'
    preprocess_bllip(test2015_prefix, 'de', 'en', train_prefix)
    preprocess_bllip(test2016_prefix, 'de', 'en', train_prefix)
    return

    # train penntrg bpe model
    # train_bpe('/home/nlp/aharonr6/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train.tok.penntrg.clean.true.de',
    #           '/home/nlp/aharonr6/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train.tok.penntrg.clean.true.en',
    #           BPE_OPERATIONS,
    #           '/home/nlp/aharonr6/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train.tok.penntrg.clean.true.bpemodel.deen')
    # return

    # parse large file
    input_path = BASE_PATH + '/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train.tok.penntrg.clean.true.desc.en'
    # input_path = '/home/nlp/aharonr6/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train.tok.penntrg.clean.true.desc.en.sample'
    output_path = input_path + '.parsed'
    bpe_model_path = BASE_PATH + '/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train.tok.penntrg.clean.true.bpemodel.deen'
    # parallel_bllip_parse_large_file(input_path, output_path, 1000)
    # return


    # convert bllip trees to linearized
    # bllip_to_linearized_parallel(output_path, output_path + '.lin', bpe_model_path)
    # os.system('wc -l {}'.format(output_path + '.lin'))

    # compare_tree_yield_to_bpe_test(BASE_PATH + '/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train.tok.penntrg.clean.true.desc.en.parsed.lin',
    #                            BASE_PATH + '/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train.tok.penntrg.clean.true.desc.bpe.en')

    # return

    # preprocess de_en_raw wmt16 for bllip
    prefix = BASE_PATH + '/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train'
    src = 'de'
    trg = 'en'

    # build stt raw dictionaries
    build_nematus_dictionary('{}.tok.penntrg.clean.true.bpe.{}'.format(prefix, src),
                             '{}.tok.penntrg.clean.true.desc.en.parsed.lin'.format(prefix))

    # preprocess_bllip(prefix, src, trg)
    return

    # preprocess de_en_raw wmt16
    # preprocess_de_en_wmt16_from_scratch()

    # run sanity check on trees
    # bped_trees = base_path + '/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.en.parsed2.final.true.bped.final'
    # bped_sentences = base_path + '/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.en.bpe'
    # trees_sanity(bped_sentences, bped_trees)
    # return

    # train bpe model
    # train_bpe('/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.de',
    #           '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.en',
    #           BPE_OPERATIONS, '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en/train/de-en-true-bpe.model')
    # return

    # create lexicalized trees based on truecased data for train
    # true_bpe_model = base_path + '/git/research/nmt/data/WMT16/de-en/train/de-en-true-bpe.model'
    # text_path = '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.en'
    # trees_path = '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.en.parsed2.final'
    # divide_file(text_path)
    # divide_file(trees_path)
    # pool = Pool(processes=5)
    #
    # # do in parallel as slow
    # for i in xrange(5):
    #     pool.apply_async(apply_bpe_on_trees,
    #                      (true_bpe_model,
    #                       text_path + '._{}'.format(i),
    #                       trees_path + '._{}'.format(i),
    #                       trees_path + '._{}.bped'.format(i)))
    #
    # pool.close()
    # pool.join()
    # merge_files([trees_path + '._{}.bped'.format(i) for i in xrange(5)],
    #             trees_path + '.true.bped')
    # return

    # parse with bllip
    # dev
    # dev_true_en_file = base_path + '/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.tok.true.en'
    # dev_true_en_parsed_file = base_path + '/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.tok.true.parsed.en'
    # complete_missing_parse_tress_with_bllip(dev_true_en_file, dev_true_en_parsed_file)
    # apply_bpe_on_trees(true_bpe_model, dev_true_en_file, dev_true_en_parsed_file + '.fixed', dev_true_en_parsed_file + '.bped')
    # return

    # test
    # test_true_en_file = base_path + '/git/research/nmt/data/WMT16/de-en/test/newstest2016-deen-ref.tok.true.en'
    # test_true_en_parsed_file = base_path + '/git/research/nmt/data/WMT16/de-en/test/newstest2016-deen-ref.tok.true.parsed.en'
    # complete_missing_parse_tress_with_bllip(test_true_en_file, test_true_en_parsed_file)
    # apply_bpe_on_trees(true_bpe_model, test_true_en_file, test_true_en_parsed_file + '.fixed', test_true_en_parsed_file + '.bped')
    #
    # return

    # re run bllip for missing trees on final.true.bped
    # sents = base_path + '/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.en'
    # trees = base_path + '/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.en.parsed2.final.true.bped'
    #
    # # try to parse the missing trees
    # complete_missing_parse_tress_with_bllip(sents, trees)
    #
    # # lexicalize and bpe the missing trees
    # apply_bpe_on_trees(true_bpe_model, sents + '.fixed', trees +'.fixed', trees +'.fixed.bped')
    #
    # # add the missing trees to the main file
    # fill_missing_trees(sents, trees, sents + '.fixed', trees + '.fixed.bped')
    # return

    return


def compare_tree_yield_to_bpe_test(trees_path, bped_path):
    count = 0
    diff = 0
    with codecs.open(trees_path, 'r', 'utf-8') as trees:
        with codecs.open(bped_path, 'r', 'utf-8') as bpe:
            while True:
                if count % 100000 == 0:
                    print count
                # compare yields to bped, see where and how much diff, if ok train
                tree = trees.readline().strip()
                sent = bpe.readline().strip()
                if not tree:
                    break
                count += 1
                tokens = tree.split()
                stripped_tree = ' '.join([t for t in tokens if '(' not in t and ')' not in t])
                # stripped_tree = stripped_tree.replace(' @-@ ', '-')
                # stripped_tree = stripped_tree.replace(' @/@ ', '/')
                if sent != stripped_tree:
                    # if not '-' in sent and not '/' in sent:
                    diff +=1
                    print u'{} are different out of {}\n'.format(diff, count)
                    print u'tree:\n{}\nsentence:\n{}\n'.format(stripped_tree, sent)
    print u'{} are different out of {}'.format(diff, count)


def preprocess_de_en_wmt16_from_scratch():
    # create unified WMT de-en train corpus
    # de_file_paths = [
    #     '/Users/roeeaharoni/git/research/nmt/data/WMT16/all/train/training-europarl-v7/europarl-v7.de-en.de',
    #     '/Users/roeeaharoni/git/research/nmt/data/WMT16/all/train/training-parallel-commoncrawl/commoncrawl.de-en.de',
    #     '/Users/roeeaharoni/git/research/nmt/data/WMT16/all/train/training-parallel-nc-v11/news-commentary-v11.de-en.de']
    #
    # en_file_paths = [
    #     '/Users/roeeaharoni/git/research/nmt/data/WMT16/all/train/training-europarl-v7/europarl-v7.de-en.en',
    #     '/Users/roeeaharoni/git/research/nmt/data/WMT16/all/train/training-parallel-commoncrawl/commoncrawl.de-en.en',
    #     '/Users/roeeaharoni/git/research/nmt/data/WMT16/all/train/training-parallel-nc-v11/news-commentary-v11.de-en.en']
    #
    de_target_path = '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en-raw/wmt16.train.de'
    en_target_path = '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en-raw/wmt16.train.en'
    #
    # command = 'cat {} {} {} > {}'.format(de_file_paths[0], de_file_paths[1], de_file_paths[2], de_target_path)
    # os.system(command)
    # command = 'cat {} {} {} > {}'.format(en_file_paths[0], en_file_paths[1], en_file_paths[2], en_target_path)
    # os.system(command)

    # "conventional" preprocessing
    prefix = '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train'
    src = 'de'
    trg = 'en'

    # full_preprocess(prefix, src, trg)

    # check train data size diff (by cleaning with 50 thresh) - maybe 4.5m and not 4.2m due to len 50 limit?
    # seems like it as shrinks to 4153515 sentences - DONE

    # preprocess new dev files - DONE
    dev_prefix = '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en-raw/dev/newstest-2013-2014-deen'
    # full_preprocess(dev_prefix, src, trg)

    # preprocess new test files - DONE
    test2015_prefix = '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en-raw/test/newstest2015-deen'
    # full_preprocess(test2015_prefix, src, trg)
    test2016_prefix = '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en-raw/test/newstest2016-deen'
    # full_preprocess(test2016_prefix, src, trg)

    # train model on raw data - DONE

    # create the parsed version of the new corpus: moses ptb tok -> detok symbols -> BLLIP parse (on server, parallel) -> bpe trees
    preprocess_bllip(BASE_PATH + '/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train', 'de', 'en')

    return

# standard preprocess wmt16 from scratch
def full_preprocess(prefix, src, trg, prev_train_prefix=None):
    # train_prefix = '/Users/roeeaharoni/git/research/nmt/data/WMT16/de-en-raw/train/wmt16.train'
    if prev_train_prefix is not None:
        on_train = False
        train_prefix = prev_train_prefix
    else:
        on_train = True
        train_prefix = prefix

    # normalize punctuation (mainly spaces near punctuation. also has -penn option, irrelevant as penn comes later)
    # tokenize (-a is aggressive hyphen splitting)
    # $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $SRC > data/$prefix.tok.$SRC
    normalize_tokenize_command_format = 'cat {}.{} | {}/scripts/tokenizer/normalize-punctuation.perl -l {} | \
    {}/scripts/tokenizer/tokenizer.perl -a -l {} > {}.tok.{}'

    # tokenize src
    os.system(normalize_tokenize_command_format.format(prefix, src, MOSES_HOME, src, MOSES_HOME, src, prefix, src))

    # tokenize target
    os.system(normalize_tokenize_command_format.format(prefix, trg, MOSES_HOME, trg, MOSES_HOME, trg, prefix, trg))
    print 'finished tokenization'

    # clean
    # $mosesdecoder/scripts/training/clean-corpus-n.perl data/corpus.tok $SRC $TRG data/corpus.tok.clean 1 80
    clean_command = '{}/scripts/training/clean-corpus-n.perl {}.tok {} {} {}.tok.clean 1 80'.format(MOSES_HOME,
                                                                                                    prefix,
                                                                                                    src, trg, prefix)
    os.system(clean_command)
    print 'finished cleaning'

    # truecase - train
    # $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus data/corpus.tok.clean.$SRC -model model/truecase-model.$SRC
    if on_train:
        train_moses_truecase(prefix + '.tok.clean.' + src, prefix + '.tok.clean.' + src + '.tcmodel')
        train_moses_truecase(prefix + '.tok.clean.' + trg, prefix + '.tok.clean.' + trg + '.tcmodel')
        print 'trained truecasing'

    # truecase - apply
    # $mosesdecoder/scripts/recaser/truecase.perl -model model/truecase-model.$SRC < data/$prefix.tok.clean.$SRC > data/$prefix.tc.$SRC
    # truecase source
    apply_moses_truecase(prefix + '.tok.clean.' + src,
                         prefix + '.tok.clean.true.' + src,
                         train_prefix + '.tok.clean.' + src + '.tcmodel')
    # truecase target
    apply_moses_truecase(prefix + '.tok.clean.' + trg,
                         prefix + '.tok.clean.true.' + trg,
                         train_prefix + '.tok.clean.' + trg + '.tcmodel')

    print 'finished truecasing'

    # BPE - train (on both sides together)
    # cat data/corpus.tc.$SRC data/corpus.tc.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRC$TRG.bpe
    if on_train:
        train_bpe(prefix + '.tok.clean.true.' + src,
                  prefix + '.tok.clean.true.' + trg,
                  BPE_OPERATIONS,
                  prefix + '.tok.clean.true.bpemodel.' + src + trg)
        print 'trained BPE'

    # BPE - apply
    # $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < data/$prefix.tc.$SRC > data/$prefix.bpe.$SRC
    apply_BPE(prefix + '.tok.clean.true.' + src,
              prefix + '.tok.clean.true.bpe.' + src,
              train_prefix + '.tok.clean.true.bpemodel.' + src + trg)

    apply_BPE(prefix + '.tok.clean.true.' + trg,
              prefix + '.tok.clean.true.bpe.' + trg,
              train_prefix + '.tok.clean.true.bpemodel.' + src + trg)
    print 'applied BPE'

    if on_train:
        # build dict $nematus/data/build_dictionary.py data/corpus.bpe.$SRC data/corpus.bpe.$TRG
        build_nematus_dictionary(prefix + '.tok.clean.true.bpe.' + src, prefix + '.tok.clean.true.bpe.' + trg)
        print 'built dictionaries'

    print 'finished preprocessing. files to use:'

    # train files
    print prefix + '.tok.clean.true.bpe.' + src
    print prefix + '.tok.clean.true.bpe.' + trg


# convert the output of the bllip trees to the format we use for training the nmt system
def bllip_to_linearized(parse_trees_path, linearized_path, bpe_model_path):
    count = 0
    bpe_model = apply_bpe.BPE(bpe_model_path)
    with codecs.open(parse_trees_path, 'r', 'utf-8') as bllip_trees:
        with codecs.open(linearized_path, 'w', 'utf-8') as linearized_trees:
            while True:
                count += 1
                if count % 1000 == 0:
                    print 'went through {} trees'.format(count)
                line = bllip_trees.readline()
                if not line:
                    break
                if line == u'MISSING\n':
                    linearized_trees.write(u'MISSING\n')
                    continue

                tree = yoav_trees.Tree('TOP').from_sexpr(line.encode('utf-8'))

                # remove POS tags
                no_pos_tree = remove_pos_tags_from_tree(tree)

                # split hyphens/slashes in words
                # split_tree = split_hyphens_slashes(no_pos_tree)

                # TODO: re-escape special chars? check this

                # bpe the words
                bped = bpe_leaves(no_pos_tree, bpe_model)

                # create linearized string
                bped_string = bped.nonter_closing()

                # print bped_string
                linearized_trees.write(u'{}\n'.format(bped_string))
    return


# run bllip_to_linearized in parallel
def bllip_to_linearized_parallel(parse_trees_path, linearized_path, bpe_model_path, lines_per_file=500000):
    input_paths = divide_file(parse_trees_path, lines_per_file)
    output_paths = [path + '.lin' for path in input_paths]
    pool = Pool(processes=len(input_paths))
    for i, path in enumerate(input_paths):
        pool.apply_async(bllip_to_linearized,(path, output_paths[i], bpe_model_path))

    pool.close()
    pool.join()

    merge_files(output_paths, linearized_path)
    delete_files(input_paths)
    delete_files(output_paths)

    return


# remove the pos tags level from the tree
def remove_pos_tags_from_tree(tree):
    new_children = []
    for child in tree.children:
        if child.isleaf():
            return yoav_trees.Tree(child.label)
        else:
            new_children.append(remove_pos_tags_from_tree(child))

    return yoav_trees.Tree(tree.label, new_children)


# ended up not using this
def split_hyphens_slashes(tree):
    tree_str = str(tree)

    # split hyphens inside words
    hyphen_split = re.sub(r'([A-Za-z0-9]+)-([A-Za-z0-9]+)', r'\1 @-@ \2', tree_str)

    # split slashes inside words
    slash_split = re.sub(r'([A-Za-z0-9]+)/([A-Za-z0-9]+)', r'\1 @/@ \2', hyphen_split)

    return yoav_trees.Tree('TOP').from_sexpr(slash_split)


# preprocessing + parse with bllip. english is ptb tokenized, german is "normally" tokenized
def preprocess_bllip(prefix, src, trg, train_prefix = None):
    is_train = False
    if train_prefix == None:
        is_train = True
        train_prefix = prefix

    # normalize punctuation (mainly spaces near punctuation.)
    # tokenize (-a is aggressive hyphen splitting)
    normalize_tokenize_command_format = 'cat {}.{} | {}/scripts/tokenizer/normalize-punctuation.perl -l {} | \
    {}/scripts/tokenizer/tokenizer.perl {} -a -l {} > {}.tok.penntrg.{}'

    # tokenize src
    os.system(normalize_tokenize_command_format.format(prefix, src, MOSES_HOME, src, MOSES_HOME, '', src, prefix, src))

    # tokenize target (ptb style)
    os.system(normalize_tokenize_command_format.format(prefix, trg, MOSES_HOME, trg, MOSES_HOME, '-penn', trg, prefix,
                                                       trg))
    print 'finished tokenization'

    # clean
    # $mosesdecoder/scripts/training/clean-corpus-n.perl data/corpus.tok $SRC $TRG data/corpus.tok.clean 1 80

    if is_train:
        length_threshold = 80
    else:
        length_threshold = 999
    clean_command = '{}/scripts/training/clean-corpus-n.perl {}.tok.penntrg {} {} {}.tok.penntrg.clean 1 {}'.format(
                                                                                                    MOSES_HOME,
                                                                                                    prefix,
                                                                                                    src, trg, prefix,
                                                                                                    length_threshold)
    os.system(clean_command)
    print 'finished cleaning'

    # truecase - apply
    # $mosesdecoder/scripts/recaser/truecase.perl -model model/truecase-model.$SRC < data/$prefix.tok.clean.$SRC > data/$prefix.tc.$SRC
    # truecase source
    apply_moses_truecase(prefix + '.tok.penntrg.clean.' + src,
                         prefix + '.tok.penntrg.clean.true.' + src,
                         train_prefix + '.tok.clean.' + src + '.tcmodel')
    # truecase target
    apply_moses_truecase(prefix + '.tok.penntrg.clean.' + trg,
                         prefix + '.tok.penntrg.clean.true.' + trg,
                         train_prefix + '.tok.clean.' + trg + '.tcmodel')

    print 'finished truecasing'

    # BPE - apply
    # $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < data/$prefix.tc.$SRC > data/$prefix.bpe.$SRC
    apply_BPE(prefix + '.tok.penntrg.clean.true.' + src,
              prefix + '.tok.penntrg.clean.true.bpe.' + src,
              train_prefix + '.tok.penntrg.clean.true.bpemodel.' + src + trg)

    print 'applied BPE on source side'

    # prepare for BLLIP parsing:

    # de-escape special chars in target
    deescape_command = '{}/scripts/tokenizer/deescape-special-chars-PTB.perl < {} > {}'.format(MOSES_HOME,
                                                                        prefix + '.tok.penntrg.clean.true.' + trg,
                                                                        prefix + '.tok.penntrg.clean.true.desc.' + trg)
    os.system(deescape_command)

    # Unsplit hyphens
    command = 'sed -i \"s/ \@-\@ /-/g\" {} '.format(prefix + '.tok.penntrg.clean.true.desc.' + trg)
    os.system(command)

    # Unsplit slashes
    command = 'sed -i \"s/ \@\/\@ /\//g\" {}'.format(prefix + '.tok.penntrg.clean.true.desc.' + trg)
    os.system(command)

    # create bped target file for test purposes
    apply_BPE(prefix + '.tok.penntrg.clean.true.desc.' + trg,
              prefix + '.tok.penntrg.clean.true.desc.bpe.' + trg,
              train_prefix + '.tok.penntrg.clean.true.bpemodel.' + src + trg)

    print 'applied BPE on target side (for test purposes)'

    # so far we should have:
    # bped version of src(non-penn) tokenized, clean, truecased
    # target(penn) tokenized, clean, truecased
    # non bped vesion of target, with unsplit hyphens and slashes, ready for bllip parsing

    # parse with python bllip version
    input_path = '{}.tok.penntrg.clean.true.desc.{}'.format(prefix, trg)
    parse_trees_path = input_path + '.parsed'
    print 'parsing target with bllip...'
    parallel_bllip_parse_large_file(input_path, parse_trees_path, lines_per_sub_file=1000)
    print 'done parsing.'

    # create linearized trees (remove POS tags, bpe the words)
    linearized_path = '{}.tok.penntrg.clean.true.desc.parsed.linear.bpe.{}'.format(prefix, trg)
    bpe_model_path = train_prefix + '.tok.penntrg.clean.true.bpemodel.' + src + trg

    # bllip_to_linearized(parse_trees_path, linearized_path, bpe_model_path)
    bllip_to_linearized_parallel(parse_trees_path, linearized_path, bpe_model_path)

    # check if yields are identical to bped ptb version
    compare_tree_yield_to_bpe_test(linearized_path, prefix + '.tok.penntrg.clean.true.desc.bpe.' + trg)

    # build dictionaries
    if is_train:
        build_nematus_dictionary('{}.tok.penntrg.clean.true.bpe.{}'.format(prefix, src),
                                 '{}.tok.penntrg.clean.true.desc.parsed.linear.bpe.{}'.format(prefix, trg))
        print 'built dictionaries'

    # print 'finished preprocessing. files to use:'
    #
    # # train files
    # print prefix + '.tok.clean.true.bpe.' + src
    # print prefix + '.tok.clean.true.bpe.' + trg

    # TODO: re-train bpe2tree


def delete_files(paths):
    for path in paths:
        os.system('rm {}'.format(path))


def parallel_bllip_parse_large_file(input_path, output_path, lines_per_sub_file=200000):
    start = time.time()
    paths = divide_file(input_path, lines_per_sub_file)
    parsed_paths = [path + '.parsed' for path in paths]
    # print paths[0:5]
    # print parsed_paths[0:5]

    # run with GNU parallel on all the files, limit cpu amount, add eta option
    commands_file = input_path + '.commands'
    with codecs.open(commands_file, 'w', 'utf-8') as commands:
        for i,path in enumerate(paths):
            commands.write('python {}/git/research/nmt/src/bllip_parse.py {} {}\n'.format(BASE_PATH, path, parsed_paths[i]))
    # parallel_command = 'parallel --bar -j 30 --xapply \'python {}/git/research/nmt/src/bllip_parse.py\' ::: {} ::: {}'\
    #     .format(BASE_PATH, ' '.join(paths[0:100]), ' '.join(parsed_paths[0:100]))
    # print parallel_command
    parallel_command = 'parallel --bar -j 30 < {}'.format(commands_file)
    os.system(parallel_command)

    # pool = Pool(processes=len(paths))
    # for path in paths:
    #     pool.apply_async(bllip_parse, (path, path + '.parsed'))
    #
    # pool.close()
    # pool.join()

    merge_files(parsed_paths, output_path)
    delete_files(paths)
    delete_files(parsed_paths)
    end = time.time()
    print 'parsing took {} seconds'.format(end - start)
    print 'parsed sentences are in: {}'.format(output_path)
    return


def bllip_parse(input_file, output_file):
    from bllipparser import RerankingParser
    rrp = RerankingParser.fetch_and_load('WSJ+Gigaword-v2', verbose=True)
    print rrp.set_parser_options(nbest=50)
    # parses = []
    count = 0
    with codecs.open(input_file, 'r', encoding='utf8') as input:
        with codecs.open(output_file, 'w', encoding='utf-8') as output:
            while True:
                count += 1
                if count % 10000 == 0:
                    print 'parsed {} sentences from {}'.format(count, input_file)
                sent = input.readline()
                if not sent:
                    break  # EOF
                # print sent
                try:
                    tokens = [s.encode('utf-8') for s in sent.strip().split(' ')]
                    parse = rrp.simple_parse(tokens)
                    output.write(format(parse.decode('utf-8')) + '\n')
                except Exception as e:
                    print u'exception when parsing: {}'.format(sent)
                    print e
                    output.write('MISSING\n')
                    # parse = 'MISSING'
                # print parse
                # print '\n\n'
                # parses.append(parse)

    # return parses
    return

# count how many missing, how many overPOS/underPOS (uneven), how many invalid trees, avg. sent/tree length
def trees_sanity(bped_sentences, bped_trees):
    missing = 0
    uneven = 0
    failed = 0
    total = 0
    tree_len_sum = 0
    tok_len_sum = 0
    tree_len_hist = defaultdict(int)
    sent_len_hist = defaultdict(int)
    with codecs.open(bped_sentences, encoding='utf8') as sents:
        with codecs.open(bped_trees, encoding='utf8') as trees:
            while True:
                if total % 100000 == 0 and total != 0:
                    print 'avg. sent len:{}\navg. tree len:{}\nmissing:{}\nfailed:{}\nuneven:{}\ntotal:{}'.format(
                        tok_len_sum / total,
                        tree_len_sum / total,
                        missing,
                        failed,
                        uneven,
                        total)

                total += 1
                sent = sents.readline()
                tree = trees.readline()

                tokens = sent.split()
                tok_amount = len(tokens)
                tok_len_sum += tok_amount
                sent_len_hist[tok_amount] += 1

                tree_len = len(tree.split())
                tree_len_sum += tree_len
                tree_len_hist[tree_len] += 1

                if 'MISSING' in tree:
                    missing += 1
                else:
                    try:
                        cleaned = ' '.join([')' if ')' in t else t for t in tree.split()])
                        parsed = yoav_trees.Tree('TOP').from_sexpr(cleaned)
                        if len(parsed.leaves()) != tok_amount:
                            uneven += 1
                    except Exception as e:
                        failed += 1
                if not sent:
                    break  # EOF
    print '#########Tree len hist#########'
    for key in sorted(tree_len_hist.keys()):
        print key, tree_len_hist[key]

    print '#########Sent len hist#########'
    for key in sorted(sent_len_hist.keys()):
        print key, sent_len_hist[key]

    print 'avg. sent len:{}\navg. tree len:{}\nmissing:{}\nfailed:{}\nuneven:{}\ntotal:{}' \
        .format(tok_len_sum / total, tree_len_sum / total, missing, failed, uneven, total)


# truecase de-en train dev test
def truecase_de_en():
    base_path = '/Users/roeeaharoni'

    en_tc_model_path = base_path + '/git/research/nmt/models/de-en.en.truecase.model'
    en_train_tok_path = base_path + '/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.en'
    en_train_true_path = base_path + '/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.en'
    en_dev_tok_path = base_path + '/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.en.tok'
    en_dev_true_path = base_path + '/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.tok.true.en'
    en_test_tok_path = base_path + '/git/research/nmt/data/WMT16/de-en/test/newstest2016-deen-ref.en.tok'
    en_test_true_path = base_path + '/git/research/nmt/data/WMT16/de-en/test/newstest2016-deen-ref.tok.true.en'

    train_moses_truecase(en_train_tok_path, en_tc_model_path)
    apply_moses_truecase(en_train_tok_path, en_train_true_path, en_tc_model_path)
    apply_moses_truecase(en_dev_tok_path, en_dev_true_path, en_tc_model_path)
    apply_moses_truecase(en_test_tok_path, en_test_true_path, en_tc_model_path)

    de_tc_model_path = base_path + '/git/research/nmt/models/de-en.de.truecase.model'
    de_train_tok_path = base_path + '/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.de'
    de_train_true_path = base_path + '/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.de'
    de_dev_tok_path = base_path + '/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-src.de.tok'
    de_dev_true_path = base_path + '/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-src.tok.true.de'
    de_test_tok_path = base_path + '/git/research/nmt/data/WMT16/de-en/test/newstest2016-deen-src.de.tok'
    de_test_true_path = base_path + '/git/research/nmt/data/WMT16/de-en/test/newstest2016-deen-src.tok.true.de'

    train_moses_truecase(de_train_tok_path, de_tc_model_path)
    apply_moses_truecase(de_train_tok_path, de_train_true_path, de_tc_model_path)
    apply_moses_truecase(de_dev_tok_path, de_dev_true_path, de_tc_model_path)
    apply_moses_truecase(de_test_tok_path, de_test_true_path, de_tc_model_path)
    return


def divide_file(path, lines_per_file = 1000000):
    i = 0
    j = 0
    sub_paths = []
    output = False
    dir_path = os.path.dirname(os.path.abspath(path))
    file_name = os.path.basename(path)
    if not os.path.exists(dir_path + '/tmp'):
        os.makedirs(dir_path + '/tmp')
    with codecs.open(path, encoding='utf8') as lines:
        while True:
            sent = lines.readline()
            # if i % 100000 == 0:
            #     print i
            if i % lines_per_file == 0:
                if i > 0:
                    output.close()
                sub_file_path = '{}/tmp/{}._{}'.format(dir_path, file_name, j)
                sub_paths.append(sub_file_path)
                output = codecs.open(sub_file_path, 'w', encoding='utf8')
                j += 1
            output.write(sent)
            i += 1
            if not sent:
                break  # EOF
    return sub_paths


def merge_files(file_paths, output_path):
    with codecs.open(output_path, 'w', encoding='utf8') as output:
        for file in file_paths:
            with codecs.open(file, encoding='utf8') as lines:
                while True:
                    sent = lines.readline()
                    if not sent:
                        break  # EOF
                    output.write(sent)


# fill the first 2 with the missing lines found in the second 2
def fill_missing_trees(words_file_path, trees_file_path, missing_words_file_path,
                       missing_trees_file_path):
    text2tree = {}
    with codecs.open(missing_words_file_path, encoding='utf8') as fixed_sents:
        with codecs.open(missing_trees_file_path, encoding='utf8') as fixed_trees:
            while True:
                sent = fixed_sents.readline()
                tree = fixed_trees.readline()
                text2tree[sent] = tree
                if not sent:
                    break  # EOF

    i = 0
    j = 0
    with codecs.open(words_file_path, encoding='utf8') as sents:
        with codecs.open(trees_file_path, encoding='utf8') as trees:
            with codecs.open(trees_file_path + '.final', 'w', encoding='utf8') as final:
                while True:
                    if j % 100000 == 0:
                        print 'went through {} lines'.format(j)
                    j += 1
                    sent = sents.readline()
                    tree = trees.readline()
                    if 'MISSING' in tree:
                        final.write(text2tree[sent])
                        print 'fixed'
                        i += 1
                    else:
                        final.write(tree)
                    if not sent:
                        break  # EOF
    print 'fixed {} trees'.format(i)
    return


# trees file contains POS tags as terminals
def apply_bpe_on_trees(bpe_model_path, words_file_path, trees_file_path, bped_trees_file_path):
    # load bpe model
    bpe = apply_bpe.BPE(bpe_model_path)

    i = 0
    too_many_pos = 0
    perfect_cover = 0
    failed = 0
    with codecs.open(words_file_path, encoding='utf8') as sents:
        with codecs.open(trees_file_path, encoding='utf8') as trees:
            with codecs.open(bped_trees_file_path, 'w', encoding='utf8') as output:
                while True:
                    if i % 100000 == 0:
                        print 'went through {} lines.\nperfect cover:{}\ntoo many POS:{}\nfailed: {}\n'.format(
                            i,
                            perfect_cover,
                            too_many_pos,
                            failed)
                    i += 1
                    sent = sents.readline()
                    tree = trees.readline()
                    words = sent.split()
                    tree_toks = tree.split()
                    word_count = 0
                    lex_tree = []
                    # print sent
                    # print tree
                    sent_len = len(words)
                    for t in tree_toks:
                        if '(' in t:
                            lex_tree.append(t)
                        else:
                            if ')' in t:
                                lex_tree.append(')')
                            else:
                                if word_count < sent_len:
                                    lex_tree.append(words[word_count])
                                else:
                                    # print 'too many POS'
                                    too_many_pos += 1
                                word_count += 1
                    if word_count == sent_len:
                        perfect_cover += 1
                    lex_tree = ' '.join(lex_tree)

                    # bpe the words
                    try:
                        if lex_tree != '':
                            parsed = yoav_trees.Tree('TOP').from_sexpr(lex_tree)
                            bped = bpe_leaves(parsed, bpe)
                            bped_str = bped.nonter_closing() + '\n'
                            output.write(bped_str)
                    except Exception as e:
                        failed += 1
                        output.write('MISSING\n')
                        print str(e)
                        traceback.print_exc()

                    if not sent:
                        break  # EOF

    print 'Done!\nwent through {} lines.\nperfect cover:{}\ntoo many POS:{}\nfailed: {}\n'.format(
        i,
        perfect_cover,
        too_many_pos,
        failed)


# applied only on root or non terminals
def bpe_leaves(tree, bpe):
    bped_children = []
    for child in tree.children:
        if child.isleaf():
            segs = bpe.segment(child.label.decode('utf-8')).strip().split()
            for seg in segs:
                bped_children.append(yoav_trees.Tree(seg, None))
        else:
            bped_children.append(bpe_leaves(child, bpe))

    return yoav_trees.Tree(tree.label, bped_children)


def get_shi_parse_tree_for_tokenized_wmt(sentences_file='../data/shi/Eng_Parse_3/8m.train.trainwords',
                                         trees_file='../data/shi/Eng_Parse_3/8m.train.trainline',
                                         wmt_file='/Users/roeeaharoni/Google Drive/de-en-wmt16/corpus.parallel.tok.en'):
    text2tree = {}

    # read syntax trees and matching sentences
    with codecs.open(sentences_file, encoding='utf8') as sents:
        with codecs.open(trees_file, encoding='utf8') as trees:
            i = 0
            print 'reading parsed sentences and matching trees...'
            while True:

                i += 1
                sent = sents.readline()
                tree = trees.readline()
                # text2tree[sent.lower().replace(' ','').replace(u'-','').replace(u'–','')] = tree
                text2tree[clean_sent(sent)] = tree
                if 'which has been running for four years , is being continued and extended' in sent:
                    print sent

                if 'When you repeatedly come up against the limits of the domain you are operating in' in sent:
                    print sent

                if 'Gradual Software and callas software today' in sent:
                    print sent

                if 'Puma Flipper Valentines Day Wmns' in sent:
                    print sent

                if 'Keep all partitions and use existing free space' in sent:
                    print sent

                if i % 10000 == 0:
                    print 'read {} trees'.format(i)
                if not sent:
                    break  # EOF
    # return
    i = 0
    found = 0
    with codecs.open(wmt_file, encoding='utf8') as sents:
        with codecs.open(wmt_file + '.parsed', mode='w', encoding='utf8') as output:
            while True:
                i += 1
                if i % 100000 == 0:
                    print '#' * 100
                    print 'looked for {} trees from training corpora, found {} so far'.format(i, found)
                    print '#' * 100
                sent = sents.readline()

                cleaned_sent = clean_sent(sent)

                if cleaned_sent in text2tree:
                    found += 1
                    output.write(text2tree[cleaned_sent])
                else:
                    print u'{} missing tree for: {}\n'.format(i, sent)
                    output.write('MISSING\n')

                if not sent:  # EOF
                    print 'looked for {} trees from training corpora, found {} total'.format(i, found)
                    print '(missing {} tress)'.format(i - found)
                    break


def clean_sent(sent):
    return sent.replace('`', '') \
        .replace('...', '') \
        .replace(u'…', '') \
        .replace(u'„', '') \
        .replace(u'´', '') \
        .replace(u'»', '') \
        .replace(u'«', '') \
        .replace('{', '') \
        .replace('}', '') \
        .replace('-LCB-', '') \
        .replace('-RCB-', '') \
        .replace('_', '') \
        .replace('&quot;', '') \
        .replace('&apos;', '') \
        .replace('@/@', '') \
        .replace('/', '') \
        .replace('-LRB-', '') \
        .replace('-RRB-', '') \
        .replace('-LSB-', '') \
        .replace('-RSB-', '') \
        .replace('(', '') \
        .replace(')', '') \
        .replace('&#91;', '') \
        .replace('&#93;', '') \
        .replace(u'-', '') \
        .replace(u'–', '') \
        .replace(' ', '') \
        .lower()


def complete_missing_parse_tress_with_bllip(sentences_file, trees_file):
    # initialize bllip
    from bllipparser import RerankingParser
    rrp = RerankingParser.fetch_and_load('WSJ+Gigaword-v2', verbose=True)

    fixed = 0
    failed = 0

    # read syntax trees and matching sentences
    with codecs.open(sentences_file, encoding='utf8') as sents:
        with codecs.open(trees_file, encoding='utf8') as trees:
            with codecs.open(trees_file + '.fixed', mode='w', encoding='utf8') as trees_output:
                with codecs.open(sentences_file + '.fixed', mode='w', encoding='utf8') as sents_output:
                    i = 0
                    print 'reading parsed sentences and matching trees...'
                    while True:
                        i += 1
                        if i % 100000 == 0:
                            print '*' * 100 + 'went through {} lines'.format(i) + '*' * 100
                        sent = sents.readline()
                        tree = trees.readline()
                        if 'MISSING' in tree:
                            sents_output.write(sent)
                            try:
                                parsed = rrp.simple_parse(sent.encode('utf-8'))
                                trees_output.write(convert_bllip_tree_to_xing_tree(parsed) + '\n')
                                fixed += 1
                                print 'parsed missing tree'
                            except Exception as e:
                                trees_output.write('MISSING\n')
                                print u'failed to parse missing tree for: {}'.format(sent)
                                print str(e)
                                traceback.print_exc()
                                failed += 1
                        # else:
                        #     trees_output.write(tree)

                        if not sent:
                            break  # EOF
                    print 'parsed {} missing trees, failed to parse {} missing trees'.format(fixed, failed)
    return


def convert_bllip_tree_to_xing_tree(bllip_tree):
    # converts bllip tree (lexicalized, no labels on closing brackets) to xing tree (unlexicalized, labels on closing
    # brac.)
    # example:

    # bllip:
    # '(S1 (S (S (NP (PRP I)) (VP (VBP declare) (VP (VBN resumed) (NP (NP (DT the) (NN session)) (PP (IN of) (NP
    # (DT the) (NNP European) (NNP Parliament))) (VP (VBN adjourned) (PP (IN on) (NP (NNP Friday) (CD 17))) (NP
    # (NNP December) (CD 1999))))))) (, ,) (CC and) (S (NP (PRP I)) (VP (MD would) (VP (VB like) (ADVP (RB once)
    # (RB again)) (S (VP (TO to) (VP (VB wish) (NP (PRP you)) (NP (NP (DT a) (JJ happy) (JJ new) (NN year)) (PP (IN in)
    # (NP (DT the) (NN hope) (SBAR (IN that) (S (NP (PRP you)) (VP (VBD enjoyed) (NP (DT a) (JJ pleasant) (JJ festive)
    # (NN period)))))))))))))) (. .)))'

    # xing shi:
    # '(TOP (S (S (NP PRP )NP (VP VBP (VP VBN (NP (NP DT NN )NP (PP IN (NP DT NNP NNP )NP )PP (VP VBN (PP IN (NP NNP CD
    # )NP )PP (NP NNP CD )NP )VP )NP )VP )VP )S , CC (S (NP PRP )NP (VP MD (VP VB (ADVP RB RB )ADVP (S (VP TO (VP VB
    # (NP PRP )NP (NP (NP DT JJ JJ NN )NP (PP IN (NP DT NN (SBAR IN (S (NP PRP )NP (VP VBD (NP DT JJ JJ NN )NP )VP )S
    # )SBAR )NP )PP )NP )VP )VP )S )VP )VP )S . )S )TOP'

    root = yoav_trees.Tree.from_sexpr(bllip_tree)

    # remove words and turn into POS terminals (=remove leaves):
    removed_leaves = remove_leaves(root)
    removed_leaves.label = 'TOP'
    return removed_leaves.nonter_closing()


def remove_leaves(tree):
    non_leaves = []
    for child in tree.children:
        if not child.isleaf():
            non_leaves.append(remove_leaves(child))
    if len(non_leaves) == 0:
        non_leaves = None
    return yoav_trees.Tree(tree.label, non_leaves)


def en_he_len_exp():
    src_file = '/Users/roeeaharoni/git/research/nmt/en-he/IWSLT14.TED.dev2010.he-en.en.tok'
    target_file = '/Users/roeeaharoni/git/research/nmt/en-he/IWSLT14.TED.dev2010.he-en.he.tok'
    output_file = '/Users/roeeaharoni/git/research/nmt/en-he/IWSLT14.TED.dev2010.he-en.en.tok.len'
    add_length_prefix(src_file, target_file, output_file)
    return src_file, target_file


def get_tss_from_tree(tree):
    tokens = tree.split()
    depth = 0
    depth2labels = {}
    for token in tokens:
        if depth not in depth2labels:
            depth2labels[depth] = []

        if '(' in token:
            depth2labels[depth].append(token.replace('(', ''))
            depth += 1

        if ')' in token:
            depth -= 1

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
                if not sent:
                    break  # EOF
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
        TSS = get_tss_from_tree(text2tree[text])

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
            TSS = get_tss_from_tree(tree)
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
                len_histogram[length] += 1

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


def get_binned_lengths(target_file, bin_size=5):
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
