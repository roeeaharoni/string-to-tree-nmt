import os

# BASE_PATH = '/Users/roeeaharoni'
BASE_PATH = '/home/nlp/aharonr6'
MOSES_HOME = BASE_PATH + '/git/mosesdecoder'
BPE_HOME = BASE_PATH + '/git/subword-nmt'
NEMATUS_HOME = BASE_PATH + '/git/nematus'

def moses_tokenize(input_file_path, output_file_path, lang):
    command_string = '{}/scripts/tokenizer/tokenizer.perl -l {} < {} > {}'.format(MOSES_HOME,
                                                                                  lang,
                                                                                  input_file_path,
                                                                                  output_file_path)
    os.system(command_string)
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


def train_bpe(src_file_path, trg_file_path, bpe_ops, bpe_model_path):
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