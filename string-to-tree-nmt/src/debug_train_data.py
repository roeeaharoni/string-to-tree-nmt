import codecs
import yoav_trees

def main():

    # compare bpe english side and the tree yields to find inconsistencies
    de_bpe_file_path = '/home/nlp/aharonr6/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.de.bpe'
    de_stt_file_path = '/home/nlp/aharonr6/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.de.bpe.copy'
    en_bpe_file_path = '/home/nlp/aharonr6/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.true.en.bpe'
    en_stt_file_path = '/home/nlp/aharonr6/git/research/nmt/data/WMT16/de-en/train/corpus.parallel.tok.en.parsed2.final.true.bped.final'
    # trees_file =

    max = 1000

    # open source files (same copied file, but to be sure)
    with codecs.open(de_bpe_file_path, 'r', 'utf-8') as de_bpe_file:
        with codecs.open(de_stt_file_path, 'r', 'utf-8') as de_stt_file:
            src_diff_count = 0
            i = 0
            while i < max:
                i+=1
                de_bpe_line = de_bpe_file.readline()
                de_stt_line = de_stt_file.readline()
                if not de_bpe_line:
                    break

                if de_bpe_line != de_stt_line:
                    print 'diff in src files:\nbpe:{}\nstt:{}\n'.format(de_bpe_line, de_stt_line)
                    src_diff_count += 1


    # open target files: trees/not trees
    with codecs.open(en_bpe_file_path, 'r', 'utf-8') as en_bpe_file:
        with codecs.open(en_stt_file_path, 'r', 'utf-8') as en_stt_file:
            i = 0
            trg_diff_count = 0
            while i < max:
                i += 1
                en_bpe_line = en_bpe_file.readline()
                en_stt_line = en_stt_file.readline()
                if not en_bpe_line:
                    break

                # strip the tree
                sent = ' '.join([t for t in en_stt_line.split() if '(' not in t and ')' not in t]).strip()
                # parsed = yoav_trees.Tree('Top').from_sexpr(en_stt_line)
                # print en_stt_line
                # print parsed.leaves()
                # sent = ' '.join(parsed.leaves())

                # check if same
                if en_bpe_line.strip() != sent:
                    print u'diff in trg files:\nbpe:{}\nstt:{}\n\ntree:{}\n'.format(en_bpe_line, sent, en_stt_line)
                    trg_diff_count += 1

    print 'found {} diffs in src and {} diffs in target'.format(src_diff_count, trg_diff_count)

if __name__ == '__main__':
    main()