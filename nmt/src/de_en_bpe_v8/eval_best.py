from src import eval_over_time
from src import moses_tools
import os
def main():
    base_path = '/home/nlp/aharonr6'
    # base_path = '~'
    # base_path = '/Users/roeeaharoni'
    nematus_path = base_path + '/git/nematus'
    moses_path = base_path + '/git/mosesdecoder'
    model_name = 'de_en_bpe_v8'

    # sgm files newstest2016
    src_sgm_2016 = base_path + '/git/research/nmt/data/WMT16/all/test/newstest2016-deen-src.de.sgm'
    ref_sgm_2016 = base_path + '/git/research/nmt/data/WMT16/all/test/newstest2016-deen-ref.en.sgm'

    # sgm files newstest2015
    src_sgm_2015 = base_path + '/git/research/nmt/data/WMT16/all/dev/newstest2015-deen-src.de.sgm'
    ref_sgm_2015 = base_path + '/git/research/nmt/data/WMT16/all/dev/newstest2015-deen-ref.en.sgm'

    model_path = base_path + '/git/research/nmt/models/{}/{}_model.iter90000.npz'.format(model_name,model_name)

    ensemble_models_path = [
        base_path + '/git/research/nmt/models/{}/{}_model.iter90000.npz'.format(model_name, model_name),
        base_path + '/git/research/nmt/models/{}/{}_model.iter120000.npz'.format(model_name,model_name),
        base_path + '/git/research/nmt/models/{}/{}_model.iter150000.npz'.format(model_name,model_name),
        base_path + '/git/research/nmt/models/{}/{}_model.iter180000.npz'.format(model_name,model_name),
        base_path + '/git/research/nmt/models/{}/{}_model.iter210000.npz'.format(model_name,model_name),
        ]

    config_path = base_path + '/git/research/nmt/models/{}/{}_model.npz.json'.format(model_name, model_name)

    os.system('cp {} {}'.format(config_path, model_path + '.json'))

    for model in ensemble_models_path:
        os.system('cp {} {}'.format(config_path, model + '.json'))

    src_2015 = base_path + '/git/research/nmt/data/news-de-en/dev/newstest2015-deen.tok.penntrg.clean.true.bpe.de'
    trg_2015 = base_path + '/git/research/nmt/models/{}/newstest2015-deen.tok.clean.true.bpe.de.output.en'.format(model_name)
    align_2015 = base_path + '/git/research/nmt/models/{}/newstest2015-deen.tok.clean.true.bpe.de.alignments.txt'.format(model_name)

    src_2016 = base_path + '/git/research/nmt/data/news-de-en/test/newstest2016-deen.tok.penntrg.clean.true.bpe.de'
    trg_2016 = base_path + '/git/research/nmt/models/{}/newstest2016-deen.tok.clean.true.bpe.de.output.en'.format(model_name)
    align_2016 = base_path + '/git/research/nmt/models/{}/newstest2016-deen.tok.clean.true.bpe.de.alignments.txt'.format(model_name)

    # single model eval
    # eval_over_time.translate(align_2015, src_2015, trg_2015, model_path, nematus_path)
    post_2015 = eval_over_time.postprocess_normal(trg_2015)

    # eval_over_time.translate(align_2016, src_2016, trg_2016, model_path, nematus_path)
    post_2016 = eval_over_time.postprocess_normal(trg_2016)

    nist2015 = moses_tools.nist_bleu(moses_path, src_sgm_2015, ref_sgm_2015, post_2015, 'en')
    nist2016 = moses_tools.nist_bleu(moses_path, src_sgm_2016, ref_sgm_2016, post_2016, 'en')

    print 'nist bleu 2015: {}\n'.format(nist2015)

    print 'nist bleu 2016: {}\n'.format(nist2016)

    # ensemble eval
    eval_over_time.translate_with_ensemble(align_2015, src_2015, trg_2015, ensemble_models_path, nematus_path)
    post_2015 = eval_over_time.postprocess_normal(trg_2015)

    eval_over_time.translate_with_ensemble(align_2016, src_2016, trg_2016, ensemble_models_path, nematus_path)
    post_2016 = eval_over_time.postprocess_normal(trg_2016)

    nist2015ens = moses_tools.nist_bleu(moses_path, src_sgm_2015, ref_sgm_2015, post_2015, 'en')
    nist2016ens = moses_tools.nist_bleu(moses_path, src_sgm_2016, ref_sgm_2016, post_2016, 'en')

    print 'evaluation for {}:\n'.format(model_name)

    print 'nist bleu 2015: {}\n'.format(nist2015)

    print 'nist bleu 2016: {}\n'.format(nist2016)

    print 'ens nist bleu 2015: {}\n'.format(nist2015ens)

    print 'ens nist bleu 2016: {}\n'.format(nist2016ens)
    return

if __name__ == '__main__':
    main()