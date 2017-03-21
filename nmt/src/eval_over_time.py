import os
import codecs
import yoav_trees
import moses_tools


def postprocess(dev_target_sents):
    # postprocess stripped trees (remove bpe, de-truecase)
    postprocess_command = './postprocess-en.sh < {} > {}.postprocessed'.format(dev_target_sents, dev_target_sents)
    os.system(postprocess_command)
    print 'postprocessed (de-bped, de-truecase) {} into {}.postprocessed'.format(dev_target_sents, dev_target_sents)
    return dev_target_sents + '.postprocessed'


# non-ptb tokenization, no detok
def postprocess_normal_tok(dev_target_sents):
    # postprocess stripped trees (remove bpe, de-truecase)
    postprocess_command = './de_en_bpe_raw/postprocess-en-tok.sh < {} > {}.postprocessed.tok'.format(dev_target_sents, dev_target_sents)
    os.system(postprocess_command)
    print 'postprocessed (de-bped, de-truecase) {} into {}.postprocessed.tok'.format(dev_target_sents, dev_target_sents)
    return dev_target_sents + '.postprocessed.tok'


# non-ptb tokenization, no detok
def postprocess_normal(dev_target_sents):
    # postprocess stripped trees (remove bpe, de-truecase)
    postprocess_command = './de_en_bpe_raw/postprocess-en.sh < {} > {}.postprocessed'.format(dev_target_sents, dev_target_sents)
    os.system(postprocess_command)
    print 'postprocessed (de-bped, de-truecase) {} into {}.postprocessed'.format(dev_target_sents, dev_target_sents)
    return dev_target_sents + '.postprocessed'


def postprocess_stt_raw(sents):
    # postprocess stripped trees (remove bpe, de-truecase)
    postprocess_command = './de_en_stt_raw/postprocess-en.sh < {} > {}.postprocessed'.format(sents,
                                                                                             sents)
    os.system(postprocess_command)
    print 'postprocessed (de-bped, de-truecase) {} into {}.postprocessed'.format(sents, sents)
    return sents + '.postprocessed'


def validate_and_strip_trees(dev_target_sents, valid_trees_log, dev_target):
    # validate and strip trees
    valid_trees = 0
    total = 0
    with codecs.open(dev_target, encoding='utf-8') as trees:
        with codecs.open(dev_target_sents, 'w', encoding='utf-8') as sents:
            with codecs.open(valid_trees_log, 'a', encoding='utf-8') as log:
                while True:
                    tree = trees.readline()
                    if not tree:
                        break  # EOF
                    total += 1
                    try:
                        parsed = yoav_trees.Tree('Top').from_sexpr(tree)
                        valid_trees += 1
                        sent = ' '.join(parsed.leaves())
                    except Exception as e:
                        sent = ' '.join([t for t in tree.split() if '(' not in t and ')' not in t])
                    sents.write(sent + '\n')
                log.write(str(valid_trees) + '\n')


def translate(alignments_path, dev_src, dev_target, model_path, nematus):
    # translate dev set using model (validate)
    print 'translating...'
    # decode: k - beam size, n - normalize scores by length, p - processes
    decode_command = 'THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu0,lib.cnmem=0.09,on_unused_input=warn python {}/nematus/translate.py \
             -m {} \
             -i {} \
             -o {} \
             -a {} \
             -k 12 -n -p 8 -v'.format(nematus, model_path, dev_src, dev_target, alignments_path)
    os.system(decode_command)
    print 'finished translating {}'.format(dev_src)


def translate_with_ensemble(alignments_path, dev_src, dev_target, model_paths, nematus):
    # translate dev set using model (validate)
    print 'translating...'
    # decode: k - beam size, n - normalize scores by length, p - processes
    decode_command = 'THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu0,lib.cnmem=0.09,on_unused_input=warn python {}/nematus/translate.py \
             -m {} \
             -i {} \
             -o {} \
             -a {} \
             -k 12 -n -p 1 -v'.format(nematus, ' '.join(model_paths), dev_src, dev_target, alignments_path)
    os.system(decode_command)
    print 'finished translating {}'.format(dev_src)


def evaluate_best_stt_raw():
    base_path = '/home/nlp/aharonr6'
    nematus_path = base_path + '/git/nematus'
    model_path = base_path + '/git/research/nmt/models/de_en_stt_raw/de_en_stt_raw_model.npz.npz.best_bleu'
    config_path = base_path + '/git/research/nmt/models/de_en_stt_raw/de_en_stt_raw_model.npz.json'
    moses_path = base_path + '/git/mosesdecoder'
    ensemble_model_paths = \
        [base_path + '/git/research/nmt/models/de_en_stt_raw/' + 'de_en_stt_raw_model.iter1110000.npz',
                    base_path + '/git/research/nmt/models/de_en_stt_raw/' + 'de_en_stt_raw_model.iter1140000.npz',
                    base_path + '/git/research/nmt/models/de_en_stt_raw/' + 'de_en_stt_raw_model.iter1170000.npz',
                    base_path + '/git/research/nmt/models/de_en_stt_raw/' + 'de_en_stt_raw_model.iter1200000.npz',
                    base_path + '/git/research/nmt/models/de_en_stt_raw/' + 'de_en_stt_raw_model.iter1230000.npz']

    os.system('cp {} {}'.format(config_path, model_path + '.json'))
    for ensemble_model_path in ensemble_model_paths:
        os.system('cp {} {}'.format(config_path, ensemble_model_path + '.json'))

    # sgm files newstest2016
    src_sgm_2016 = base_path + '/git/research/nmt/data/WMT16/all/test/newstest2016-deen-src.de.sgm'
    ref_sgm_2016 = base_path + '/git/research/nmt/data/WMT16/all/test/newstest2016-deen-ref.en.sgm'

    # sgm files newstest2015
    src_sgm_2015 = base_path + '/git/research/nmt/data/WMT16/all/dev/newstest2015-deen-src.de.sgm'
    ref_sgm_2015 = base_path + '/git/research/nmt/data/WMT16/all/dev/newstest2015-deen-ref.en.sgm'

    # src/references 2015
    src_2015 = base_path + '/git/research/nmt/data/WMT16/de-en-raw/test/newstest2015-deen.tok.clean.true.bpe.de'

    # src/references 2016
    src_2016 = base_path + '/git/research/nmt/data/WMT16/de-en-raw/test/newstest2016-deen.tok.clean.true.bpe.de'

    trg_2015_trees = base_path + '/git/research/nmt/models/de_en_stt_raw/newstest2015-deen.tok.clean.true.bpe.de.output.trees.en'
    trg_2015_sents = base_path + '/git/research/nmt/models/de_en_stt_raw/newstest2015-deen.tok.clean.true.bpe.de.output.sents.en'
    align_2015 = base_path + '/git/research/nmt/models/de_en_stt_raw/newstest2015-deen.tok.clean.true.bpe.de.alignments.txt'

    trg_2016_trees = base_path + '/git/research/nmt/models/de_en_stt_raw/newstest2016-deen.tok.clean.true.bpe.de.output.trees.en'
    trg_2016_sents = base_path + '/git/research/nmt/models/de_en_stt_raw/newstest2016-deen.tok.clean.true.bpe.de.output.sents.en'
    align_2016 = base_path + '/git/research/nmt/models/de_en_stt_raw/newstest2016-deen.tok.clean.true.bpe.de.alignments.txt'

    valid_trees_log_2015 = trg_2015_trees + '_validtrees'
    # translate(align_2015, src_2015, trg_2015_trees, model_path, nematus_path)
    translate_with_ensemble(align_2015, src_2015, trg_2015_trees, ensemble_model_paths, nematus_path)
    validate_and_strip_trees(trg_2015_sents, valid_trees_log_2015, trg_2015_trees)
    post_2015 = postprocess_stt_raw(trg_2015_sents)

    valid_trees_log_2016 = trg_2016_trees + '_validtrees'
    # translate(align_2016, src_2016, trg_2016_trees, model_path, nematus_path)
    translate_with_ensemble(align_2016, src_2016, trg_2016_trees, ensemble_model_paths, nematus_path)
    validate_and_strip_trees(trg_2016_sents, valid_trees_log_2016, trg_2016_trees)
    post_2016 = postprocess_stt_raw(trg_2016_sents)

    nist2015 = moses_tools.nist_bleu(moses_path, src_sgm_2015, ref_sgm_2015, post_2015, 'en')
    nist2016 = moses_tools.nist_bleu(moses_path, src_sgm_2016, ref_sgm_2016, post_2016, 'en')

    # nist bleu: 27.33
    print 'nist bleu 2015: {}'.format(nist2015)

    # nist bleu 2016: 31.19
    print 'nist bleu 2016: {}'.format(nist2016)


def main():
    evaluate_best_stt_raw()
    return

    base_path = '/home/nlp/aharonr6'
    # base_path = '~'
    # base_path = '/Users/roeeaharoni'
    nematus_path = base_path + '/git/nematus'
    moses_path = base_path + '/git/mosesdecoder'

    # sgm files newstest2016
    src_sgm_2016 = base_path + '/git/research/nmt/data/WMT16/all/test/newstest2016-deen-src.de.sgm'
    ref_sgm_2016 = base_path + '/git/research/nmt/data/WMT16/all/test/newstest2016-deen-ref.en.sgm'

    # sgm files newstest2015
    src_sgm_2015 = base_path + '/git/research/nmt/data/WMT16/all/dev/newstest2015-deen-src.de.sgm'
    ref_sgm_2015 = base_path + '/git/research/nmt/data/WMT16/all/dev/newstest2015-deen-ref.en.sgm'


    # prediction_path_stt_2015 = base_path + '/git/research/nmt/models/de_en_stt/newstest2015-deen-src.tok.true.de.bpe.output.sents.dev.postprocessed.best'
    # prediction_path_stt_2016 = base_path + '/git/research/nmt/models/de_en_stt/newstest2016-deen-src.penn.tok.true.de.bpe.output.sents.postprocessed'
    # stt results
    # 0.2835
    # nist_bleu(moses_path, src_sgm_2016, ref_sgm_2016, prediction_path_stt_2016, 'en')
    # 0.2736
    # nist_bleu(moses_path, src_sgm_2015, ref_sgm_2015, prediction_path_stt_2015, 'en')

    # prediction_path_bpe_2015 = base_path + '/git/research/nmt/models/de_en_bpe/newstest2015-deen-src.tok.true.de.bpe.output.dev.postprocessed.best'
    # prediction_path_bpe_2016 = base_path + '/git/research/nmt/models/de_en_bpe/newstest2016-deen-src.tok.true.de.bpe.output.dev.postprocessed.best.postprocessed'
    # bpe results
    # 0.2820
    # moses_tools.nist_bleu(moses_path, src_sgm_2016, ref_sgm_2016, prediction_path_bpe_2016, 'en')
    # 0.2719
    # moses_tools.nist_bleu(moses_path, src_sgm_2015, ref_sgm_2015, prediction_path_bpe_2015, 'en')
    # return

    # translate and evaluate bleu with de_en_bpe_raw model on newstest2015, newstest2016
    model_path = base_path + '/git/research/nmt/models/de_en_bpe_raw/de_en_bpe_raw_model.npz.npz.best_bleu'
    config_path = base_path + '/git/research/nmt/models/de_en_bpe_raw/de_en_bpe_raw_model.npz.json'
    os.system('cp {} {}'.format(config_path, model_path + '.json'))

    src_2015 = base_path + '/git/research/nmt/data/WMT16/de-en-raw/test/newstest2015-deen.tok.clean.true.bpe.de'
    trg_2015 = base_path + '/git/research/nmt/models/de_en_bpe_raw/newstest2015-deen.tok.clean.true.bpe.de.output.en'
    align_2015 = base_path + '/git/research/nmt/models/de_en_bpe_raw/newstest2015-deen.tok.clean.true.bpe.de.alignments.txt'
    ref_2015 = base_path + '/git/research/nmt/data/WMT16/de-en-raw/test/newstest2015-deen.en'
    tok_ref_2015 = base_path + '/git/research/nmt/data/WMT16/de-en-raw/test/newstest2015-deen.tok.clean.true.en'

    src_2016 = base_path + '/git/research/nmt/data/WMT16/de-en-raw/test/newstest2016-deen.tok.clean.true.bpe.de'
    trg_2016 = base_path + '/git/research/nmt/models/de_en_bpe_raw/newstest2016-deen.tok.clean.true.bpe.de.output.en'
    align_2016 = base_path + '/git/research/nmt/models/de_en_bpe_raw/newstest2016-deen.tok.clean.true.bpe.de.alignments.txt'
    ref_2016 = base_path + '/git/research/nmt/data/WMT16/de-en-raw/test/newstest2016-deen.en'
    tok_ref_2016 = base_path + '/git/research/nmt/data/WMT16/de-en-raw/test/newstest2016-deen.tok.clean.true.en'

    translate(align_2015, src_2015, trg_2015, model_path, nematus_path)
    post_2015 = postprocess_normal(trg_2015)
    # post_2015_tok = postprocess_normal_tok(trg_2015)

    # detok_2015_score = bleu(moses_path, ref_2015, post_2015)
    # tok_2015_score = bleu(moses_path, tok_ref_2015, post_2015_tok)

    translate(align_2016, src_2016, trg_2016, model_path, nematus_path)
    post_2016 = postprocess_normal(trg_2016)
    # post_2016_tok = postprocess_normal_tok(trg_2016)

    # detok_2016_score = bleu(moses_path, ref_2016, post_2016)
    # tok_2016_score = bleu(moses_path, tok_ref_2016, post_2016_tok)

    # predictions_path_2016 = base_path + '/git/research/nmt/models/de_en_bpe_raw/newstest2016-deen.tok.clean.true.bpe.de.output.en.postprocessed'
    # predictions_path_2015 = base_path + '/git/research/nmt/models/de_en_bpe_raw/newstest2015-deen.tok.clean.true.bpe.de.output.en.postprocessed'
    nist2015 = moses_tools.nist_bleu(moses_path, src_sgm_2015, ref_sgm_2015, post_2015, 'en')
    nist2016 = moses_tools.nist_bleu(moses_path, src_sgm_2016, ref_sgm_2016, post_2016, 'en')

    # nist bleu: 27.33
    print 'nist bleu 2015: {}'.format(nist2015)

    # nist bleu 2016: 31.19
    print 'nist bleu 2016: {}'.format(nist2016)

    return

    dev_src = base_path + '/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-src.tok.true.de.bpe'
    ref_path = base_path + '/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-ref.en'

    test_alignments_path = base_path + '/git/research/nmt/models/de_en_bpe/test_alignments.txt.best'
    test_src = base_path + '/git/research/nmt/data/WMT16/de-en/test/newstest2016-deen-src.penn.tok.true.de.bpe'

    bpe_config_path = base_path + '/git/research/nmt/models/de_en_bpe/de_en_bpe_model.npz.json'
    test_target = base_path + '/git/research/nmt/models/de_en_bpe/newstest2016-deen-src.tok.true.de.bpe.output.dev.postprocessed.best'
    bpe_model_path = '/home/nlp/aharonr6/git/research/nmt/models/de_en_bpe/de_en_bpe_model.npz.npz.best_bleu'

    # compute test bleu on bpe2bpe
    test_ref_path = base_path + '/git/research/nmt/data/WMT16/de-en/test/newstest2016-deen-ref.en'
    os.system('cp {} {}'.format(bpe_config_path, bpe_model_path + '.json'))
    # translate(test_alignments_path, test_src, test_target, bpe_model_path, nematus_path)
    postprocessed_path = postprocess(test_target)
    score = bleu(moses_path, test_ref_path, postprocessed_path)
    print score
    return



    stt_models_files = [
        'de_en_stt_model.iter1320000.npz',
        'de_en_stt_model.iter1290000.npz',
        'de_en_stt_model.iter1260000.npz',
        'de_en_stt_model.iter1230000.npz',
        'de_en_stt_model.iter1200000.npz',
        'de_en_stt_model.iter1170000.npz',
        'de_en_stt_model.iter1140000.npz',
        'de_en_stt_model.iter1110000.npz',
        'de_en_stt_model.iter1080000.npz',
        'de_en_stt_model.iter1050000.npz',
        'de_en_stt_model.iter1020000.npz',
        'de_en_stt_model.iter990000.npz',
        'de_en_stt_model.iter960000.npz',
        'de_en_stt_model.iter930000.npz',
        'de_en_stt_model.iter900000.npz',
        'de_en_stt_model.iter870000.npz',
        'de_en_stt_model.iter840000.npz',
        'de_en_stt_model.iter810000.npz',
        'de_en_stt_model.iter780000.npz',
        'de_en_stt_model.iter750000.npz',
        'de_en_stt_model.iter720000.npz',
        'de_en_stt_model.iter690000.npz',
        'de_en_stt_model.iter660000.npz',
        'de_en_stt_model.iter630000.npz',
        'de_en_stt_model.iter600000.npz',
        'de_en_stt_model.iter570000.npz',
        'de_en_stt_model.iter540000.npz',
        'de_en_stt_model.iter510000.npz',
        'de_en_stt_model.iter480000.npz',
        'de_en_stt_model.iter450000.npz',
        'de_en_stt_model.iter420000.npz',
        'de_en_stt_model.iter390000.npz',
        'de_en_stt_model.iter360000.npz',
        'de_en_stt_model.iter330000.npz',
        'de_en_stt_model.iter300000.npz',
        'de_en_stt_model.iter270000.npz',
        'de_en_stt_model.iter240000.npz',
        'de_en_stt_model.iter210000.npz',
        'de_en_stt_model.iter180000.npz',
        'de_en_stt_model.iter150000.npz',
        'de_en_stt_model.iter120000.npz',
        'de_en_stt_model.iter90000.npz',
        'de_en_stt_model.iter60000.npz',
        'de_en_stt_model.iter30000.npz']
    stt_models_files.reverse()

    stt_bleu_path = base_path + '/git/research/nmt/models/de_en_stt/overtime/bleu.txt'
    stt_config_path = base_path + '/git/research/nmt/models/de_en_stt/de_en_stt_model.npz.json'

    # os.mkdir(base_path + '/git/research/nmt/models/de_en_stt/overtime/')
    stt = False
    if stt:

        # foreach model file
        for f in stt_models_files:
            stt_model_path = base_path + '/git/research/nmt/models/de_en_stt/' + f

            os.system('cp {} {}'.format(stt_config_path, stt_model_path + '.json'))

            dev_target = base_path + '/git/research/nmt/models/de_en_stt/overtime/{}_newstest2015-deen-src.tok.true.de.bpe.output.trees.dev'.format(
                f)
            dev_target_sents = base_path + '/git/research/nmt/models/de_en_stt/overtime/{}_newstest2015-deen-src.tok.true.de.bpe.output.sents.dev'.format(
                f)
            alignments_path = base_path + '/git/research/nmt/models/de_en_stt/overtime/{}_dev_alignments.txt'.format(f)
            valid_trees_log = base_path + '/git/research/nmt/models/de_en_stt/overtime/{}.valid_trees_log'.format(f)

            translate(alignments_path, dev_src, dev_target, stt_model_path, nematus_path)

            validate_and_strip_trees(dev_target_sents, valid_trees_log, dev_target)

            postprocessed_path = postprocess(dev_target_sents)

            score = bleu(moses_path, ref_path, postprocessed_path)
            with codecs.open(stt_bleu_path, 'a', 'utf-8') as bleu_file:
                bleu_file.write('{}\t{}\n'.format(f, score))

            # remove config file copy
            os.system('rm {}'.format(stt_model_path + '.json'))

        return

    bpe_model_files = [ 'de_en_bpe_model.iter660000.npz',
    'de_en_bpe_model.iter630000.npz',
    'de_en_bpe_model.iter600000.npz',
    'de_en_bpe_model.iter570000.npz',
    'de_en_bpe_model.iter540000.npz',
    'de_en_bpe_model.iter510000.npz',
    'de_en_bpe_model.iter480000.npz',
    'de_en_bpe_model.iter450000.npz',
    'de_en_bpe_model.iter420000.npz']
    # bpe_model_files = ['de_en_bpe_model.iter390000.npz',
    #                    'de_en_bpe_model.iter360000.npz',
    #                    'de_en_bpe_model.iter330000.npz',
    #                    'de_en_bpe_model.iter300000.npz',
    #                    'de_en_bpe_model.iter270000.npz',
    #                    'de_en_bpe_model.iter240000.npz',
    #                    'de_en_bpe_model.iter210000.npz',
    #                    'de_en_bpe_model.iter180000.npz',
    #                    'de_en_bpe_model.iter150000.npz',
    #                    'de_en_bpe_model.iter120000.npz',
    #                    'de_en_bpe_model.iter90000.npz',
    #                    'de_en_bpe_model.iter60000.npz',
    #                    'de_en_bpe_model.iter30000.npz']

    bpe_model_files.reverse()

    bpe_bleu_path = base_path + '/git/research/nmt/models/de_en_bpe/overtime/bleu.txt'
    bpe_config_path = base_path + '/git/research/nmt/models/de_en_bpe/de_en_bpe_model.npz.json'
    bpe = True
    if bpe:
        # foreach model file
        for f in bpe_model_files:

            bpe_model_path = base_path + '/git/research/nmt/models/de_en_bpe/' + f

            os.system('cp {} {}'.format(bpe_config_path, bpe_model_path + '.json'))

            dev_target = base_path + '/git/research/nmt/models/de_en_bpe/overtime/{}_newstest2015-deen-src.tok.true.de.bpe.output.dev'.format(
                f)

            alignments_path = base_path + '/git/research/nmt/models/de_en_bpe/overtime/{}_dev_alignments.txt'.format(f)

            translate(alignments_path, dev_src, dev_target, bpe_model_path, nematus_path)

            postprocessed_path = postprocess(dev_target)

            score = bleu(moses_path, ref_path, postprocessed_path)
            with codecs.open(bpe_bleu_path, 'a', 'utf-8') as bleu_file:
                bleu_file.write('{}\t{}\n'.format(f, score))

            # remove config file copy
            os.system('rm {}'.format(bpe_bleu_path + '.json'))

        return

if __name__ == '__main__':
    main()
