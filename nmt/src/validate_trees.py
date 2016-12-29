import os
import codecs
import yoav_trees

def main():
    base_path = '/home/nlp/aharonr6'
    nematus = base_path + '/git/nematus'
    model_prefix = base_path + '/git/research/nmt/models/de_en_stt_model.npz'
    dev_src = base_path + '/git/research/nmt/data/WMT16/en-de/dev/newstest2015-deen-src.tok.true.de'
    dev_target = dev_src + '.output.trees.dev'
    dev_target_sents = dev_src + '.output.sents.dev'
    valid_trees_log = model_prefix + '.valid_trees_log'

    # decode: k - beam size, n - normalize scores by length, p - processes
    decode_command = 'THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu,on_unused_input=warn python {}/nematus/translate.py \
     -m {}.dev.npz \
     -i {} \
     -o {} \
     -k 12 -n -p 1'.format(nematus, model_prefix, dev_src, dev_target)
    os.system(decode_command)

    # validate and strip trees
    valid_trees = 0
    total = 0
    with codecs.open(dev_target, encoding='utf-8') as trees:
        with codecs.open(dev_target_sents, 'w', encoding='utf-8') as sents:
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
                    sent = [t for t in tree.split() if '(' not in t and ')' not in t]
                sents.write(sent + '\n')


    # postprocess stripped trees
    postprocess_command = './postprocess-dev.sh < {} > {}.postprocessed'.format(dev_target_sents, dev_target_sents)
    os.system(postprocess_command)

    # '/home/nlp/aharonr6/git/research/nmt/models/newstest2015-deen-src.tok.true.de.output.sents.dev.postprocessed.dev'

    # get current BLEU, compare to last best model, save as best if improved
    bleu_command = './bleu.sh'
    os.system(bleu_command)


if __name__ == '__main__':
    main()