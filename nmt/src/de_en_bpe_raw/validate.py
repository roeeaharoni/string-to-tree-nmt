import os
import codecs


def main():
    print 'validating...'

    base_path = '/home/nlp/aharonr6'

    nematus = base_path + '/git/nematus'

    model_prefix = base_path + '/git/research/nmt/models/de_en_bpe_raw/de_en_bpe_raw_model.npz'

    dev_src = base_path + '/git/research/nmt/data/WMT16/de-en-raw/dev/newstest-2013-2014-deen.tok.clean.true.bpe.de'

    dev_target = base_path + '/git/research/nmt/models/de_en_bpe_raw/newstest-2013-2014-deen.tok.clean.true.bpe.de.output.dev'

    alignments_path = base_path + '/git/research/nmt/models/de_en_bpe_raw/dev_alignments.txt'

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
    postprocess(base_path, dev_target, dev_target + '.postprocessed')
    print 'postprocessed (de-bped, de-truecase) {} into {}.postprocessed'.format(dev_target, dev_target)

    # get current BLEU, compare to last best model, save as best if improved
    bleu_command = './bleu.sh'
    os.system(bleu_command)


def postprocess(base_path, input_path, output_path):
    moses_home = base_path + '/git/mosesdecoder'

    # fix BPE split words, detruecase, detokenize
    command = 'sed \'s/\@\@ //g\' | \
    {}/scripts/recaser/detruecase.perl | \
    {}/scripts/tokenizer/detokenizer.perl -l en < {} > {}'.format(moses_home, moses_home, input_path, output_path)

    os.system(command)


if __name__ == '__main__':
    main()
