import os


def main():
    print 'validating...'

    model_name = 'de_en_bpe_v8'

    base_path = '/home/nlp/aharonr6'

    nematus = base_path + '/git/nematus'

    model_prefix = base_path + '/git/research/nmt/models/{}/{}_model.npz'.format(model_name, model_name)

    dev_src = base_path + '/git/research/nmt/data/news-de-en/dev/newstest2015-deen.tok.clean.true.bpe.de'

    dev_target = base_path + '/git/research/nmt/models/{}/newstest2015-deen.tok.clean.true.bpe.de.output.dev'.format(model_name)

    alignments_path = base_path + '/git/research/nmt/models/{}/dev_alignments.txt'.format(model_name)

    # decode dev set: k - beam size, n - normalize scores by length, p - processes
    decode_command = 'THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu1,lib.cnmem=0.09,on_unused_input=warn python {}/nematus/translate.py \
     -m {}.dev.npz \
     -i {} \
     -o {} \
     -a {} \
     -k 12 -n -p 5 -v'.format(nematus, model_prefix, dev_src, dev_target, alignments_path)
    os.system(decode_command)

    print 'finished translating {}'.format(dev_src)

    # postprocess predictions (remove bpe, de-truecase)
    postprocess_command = './postprocess-en.sh < {} > {}.postprocessed'.format(dev_target, dev_target)
    os.system(postprocess_command)
    print 'postprocessed (de-bped, de-truecase) {} into {}.postprocessed'.format(dev_target, dev_target)

    # get current BLEU, compare to last best model, save as best if improved
    bleu_command = './bleu.sh'
    os.system(bleu_command)



if __name__ == '__main__':
    main()
