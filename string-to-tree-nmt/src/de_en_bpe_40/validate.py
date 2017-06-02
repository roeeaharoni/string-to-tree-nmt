import os
import codecs

def main():
    print 'validating...'

    base_path = '/home/nlp/aharonr6'

    nematus = base_path + '/git/nematus'

    model_prefix = base_path + '/git/research/nmt/models/de_en_bpe_40/de_en_bpe_40_model.npz'

    dev_src = base_path + '/git/research/nmt/data/WMT16/de-en/dev/newstest2015-deen-src.tok.true.de.bpe'

    dev_target = base_path + '/git/research/nmt/models/de_en_bpe_40/newstest2015-deen-src.tok.true.de.bpe.output.dev'

    alignments_path = base_path + '/git/research/nmt/models/de_en_bpe_40/dev_alignments.txt'

    # decode: k - beam size, n - normalize scores by length, p - processes
    decode_command = 'THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu0,lib.cnmem=0.09,on_unused_input=warn python {}/nematus/translate.py \
     -m {}.dev.npz \
     -i {} \
     -o {} \
     -a {} \
     -k 12 -n -p 5 -v'.format(nematus, model_prefix, dev_src, dev_target, alignments_path)
    os.system(decode_command)

    print 'finished translating {}'.format(dev_src)

    # postprocess predictions (remove bpe, de-truecase)
    postprocess_command = base_path +'/git/research/nmt/src/postprocess-en.sh < {} > {}.postprocessed'.format(dev_target, dev_target)
    os.system(postprocess_command)
    print 'postprocessed (de-bped, de-truecase) {} into {}.postprocessed'.format(dev_target, dev_target)

    # '/home/nlp/aharonr6/git/research/nmt/models/newstest2015-deen-src.tok.true.de.output.sents.dev.postprocessed.dev'

    # get current BLEU, compare to last best model, save as best if improved
    bleu_command = './bleu.sh'
    os.system(bleu_command)


if __name__ == '__main__':
    main()
