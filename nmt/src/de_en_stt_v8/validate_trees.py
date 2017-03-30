import os
import codecs
from src import yoav_trees
from src import moses_tools
import os.path
from shutil import copyfile

def main():
    print 'validating trees...'

    base_path = '/home/nlp/aharonr6'

    nematus = base_path + '/git/nematus'

    model_name = 'de_en_stt_v8'

    # moses_path = base_path + '/git/mosesdecoder'

    model_prefix = base_path + '/git/research/nmt/models/{}/{}_model.npz'.format(model_name, model_name)

    dev_src = base_path + '/git/research/nmt/data/news-de-en/dev/newstest2015-deen.tok.penntrg.clean.true.bpe.de'

    dev_target = base_path + '/git/research/nmt/models/{}/newstest2015-deen.tok.penntrg.clean.true.bpe.de.output.trees.dev'.format(model_name)

    dev_target_sents = base_path + '/git/research/nmt/models/{}/newstest2015-deen.tok.penntrg.clean.true.bpe.de.output.sents.dev'.format(model_name)

    valid_trees_log = model_prefix + '.valid_trees_log'

    alignments_path = base_path + '/git/research/nmt/models/{}/dev_alignments.txt'.format(model_name)

    # decode: k - beam size, n - normalize scores by length, p - processes
    decode_command = 'THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu1,lib.cnmem=0.09,on_unused_input=warn python {}/nematus/translate.py \
     -m {}.dev.npz \
     -i {} \
     -o {} \
     -a {} \
     -k 12 -n -p 5 -v'.format(nematus, model_prefix, dev_src, dev_target, alignments_path)
    os.system(decode_command)

    print 'finished translating {}'.format(dev_src)

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
                # write how many valid trees to log
                log.write(str(valid_trees) + '\n')


    # postprocess stripped trees (de-bpe, de-truecase)
    postprocess_command = '{}/git/research/nmt/src/{}/postprocess-en.sh < {} > {}.postprocessed'.format(
        base_path, model_name, dev_target_sents, dev_target_sents)
    os.system(postprocess_command)
    print 'postprocessed (de-bped, de-truecase) {} into {}.postprocessed'.format(dev_target_sents, dev_target_sents)


    # nist bleu score - log and save best model
    moses_path = base_path + '/git/mosesdecoder/'
    src_sgm_path = base_path + '/git/research/nmt/data/news-de-en/dev/newstest2015-deen-src.de.sgm'
    ref_sgm_path = base_path + '/git/research/nmt/data/news-de-en/dev/newstest2015-deen-ref.en.sgm'
    postprocessed_path = dev_target + '.postprocessed'
    best_nist_path = model_prefix + '_best_nist_bleu.txt'
    nist_log = model_prefix + '_nist_bleu.txt'
    nist_score = moses_tools.nist_bleu(moses_path, src_sgm_path, ref_sgm_path, postprocessed_path, 'en')
    codecs.open(nist_log, 'a', 'utf8').write('{}\n'.format(nist_score))

    if os.path.exists(best_nist_path):
        best_score = float(codecs.open(best_nist_path, mode='r', encoding='utf8').readline())
        if best_score < float(nist_score):
            print 'new best nist bleu! prev: {} now: {}'.format(best_score, nist_score)
            copyfile(model_prefix, model_prefix + '_best_nist_bleu.npz')
            print 'saved new model in: {}'.format(model_prefix + '_best_nist_bleu.npz')
        else:
            print 'no improvement. prev: {} now: {}'.format(best_score, nist_score)
    else:
        codecs.open(best_nist_path, mode='w', encoding='utf8').write(str(nist_score))
        copyfile(model_prefix, model_prefix + '_best_nist_bleu.npz')


    # get current multi-BLEU, compare to last best model, save as best if improved
    bleu_command = base_path + '/git/research/nmt/src/{}/bleu.sh'.format(model_name)
    os.system(bleu_command)


if __name__ == '__main__':
    main()
