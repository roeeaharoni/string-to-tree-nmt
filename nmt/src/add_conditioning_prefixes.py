import codecs
from collections import defaultdict
import os

# TODO:
# write a pipeline that does all:
# tokenize with moses scripts - done
# count words per output sentence - done
# run BPE on both sides -
# extract TSS prefix
# write train versions (BPE2BPE): with output length, with output TSS, with both, without anything
# 4 models for 2 language pairs - totals in 8 models, 1 week per model - 8 weeks
# can be run in one week on all NLP GPUs / use cyber GPUs?
# start with fr-en TSS, fr-en len
# process is: tokenize -> clean --> truecase --> BPE --> add prefixes --> build dictionaries --> train --> evaluate

MOSES_HOME = '/Users/roeeaharoni/git/mosesdecoder'
BPE_HOME = '/Users/roeeaharoni/git/subword-nmt'
NEMATUS_HOME = '/Users/roeeaharoni/git/nematus'
BPE_OPERATIONS = 89500


def fr_en_TSS_exp():

    # get file paths
    train_src = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/xing.train.txt.tok.fr'
    train_target = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/xing.train.txt.tok.en'

    dev_src = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/xing.dev.txt.tok.fr'
    dev_target = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/xing.dev.txt.tok.en'

    test_src = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/xing.test.txt.tok.fr'
    test_target = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/xing.test.txt.tok.en'

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
        input_corpus_prefix_path = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/xing.{}.txt.tok'
        output_corpus_prefix_path = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean'
        moses_clean(input_corpus_prefix_path.format(f), output_corpus_prefix_path.format(f), 'fr', 'en')

    train_src_clean = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.train.txt.tok.clean.fr'
    train_target_clean = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.train.txt.tok.clean.en'

    # train src truecase
    src_tc_model_path = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.train.txt.tok.clean.fr.tcmodel'
    src_tc_model_path = train_moses_truecase(train_src_clean, src_tc_model_path)

    # apply src truecase
    for f in ['train', 'dev', 'test']:
        apply_moses_truecase('/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean.fr'.format(f),
                             '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean.true.fr'.format(f),
                             src_tc_model_path)

    # train target truecase
    trg_tc_model_path = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.train.txt.tok.clean.en.tcmodel'
    trg_tc_model_path = train_moses_truecase(train_target_clean, trg_tc_model_path)

    # apply target truecase
    for f in ['train', 'dev', 'test']:
        apply_moses_truecase(
            '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean.en'.format(f),
            '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean.true.en'.format(f),
            trg_tc_model_path)

    # BPE

    # train bpe
    clean_src_train_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.train.txt.tok.clean.true.en'
    clean_target_train_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.train.txt.tok.clean.true.en'
    bpe_model_path = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.train.txt.tok.clean.true.fr_en.bpe'
    bpe_model_path = train_BPE(clean_src_train_file, clean_target_train_file, BPE_OPERATIONS, bpe_model_path)

    # apply bpe
    for l in ['fr', 'en']:
        for f in ['train', 'dev', 'test']:
            apply_BPE(
                '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean.true.{}'.format(f, l),
                '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean.true.bpe.{}'.format(f, l),
                bpe_model_path)

    # add len prefixes
    for f in ['train', 'dev', 'test']:
        src_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean.true.bpe.fr'.format(f)
        with codecs.open(src_file, encoding='utf8') as src:
            src_lines = src.readlines()

        output_lengths = get_binned_lengths(
            '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean.true.en'.format(f))

        new_src_lines = []
        for i, length in enumerate(output_lengths):
            new_src_lines.append('TL{} '.format(length) + src_lines[i])

        output_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean.true.bpe.len.fr'.format(f)
        with codecs.open(output_file, 'w', encoding='utf8') as predictions:
            for i, line in enumerate(new_src_lines):
                predictions.write(u'{}'.format(line))

    # add TSS prefixes
    for f in ['train', 'dev', 'test']:
        src_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean.true.bpe.fr'.format(
            f)
        with codecs.open(src_file, encoding='utf8') as src:
            src_lines = src.readlines()

        output_lengths = get_binned_lengths(
            '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean.true.en'.format(
                f))

        new_src_lines = []
        for i, length in enumerate(output_lengths):
            new_src_lines.append('TL{} '.format(length) + src_lines[i])

        output_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Fre_4M/model/xing.{}.txt.tok.clean.true.bpe.len.fr'.format(
            f)
        with codecs.open(output_file, 'w', encoding='utf8') as predictions:
            for i, line in enumerate(new_src_lines):
                predictions.write(u'{}'.format(line))





    # build dictionaries

    # train

    # evaluate


def main ():

    fr_en_TSS_exp()

    en_he_len_exp()


    # src_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Parse_3/8m.train.trainwords'
    # target_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Parse_3/8m.train.trainwords'
    # output_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Parse_3/8m.train.TSS.en'
    # syntax_file = '/Users/roeeaharoni/git/research/nmt/data/Eng_Parse_3/8m.train.trainline'

    # add_TSS_prefix(src_file, target_file, syntax_file, output_file)


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


def add_TSS_prefix(src_file, target_file, syntax_file, output_file):
    # TOP (S (NP DT NN ) NP (VP VBZ (NP (NP NN )NP (PP IN (NP DT NNP NNP NNP )NP )PP )NP )VP. )S )TOP
    # This site is part of The Imaging Source Network.

    new_src_lines = []

    with codecs.open(syntax_file, encoding='utf8') as src:
        tree_lines = src.readlines()

    with codecs.open(src_file, encoding='utf8') as src:
        with codecs.open(target_file, encoding='utf8') as trgt:
            src_lines = src.readlines()
            trgt_lines = trgt.readlines()
            for i, line in enumerate(trgt_lines):
                # extract TSS
                # write TSS + src
    return

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