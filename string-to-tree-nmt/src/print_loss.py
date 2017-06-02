import numpy
import sys

def main(path):
    # path = '/home/nlp/aharonr6/git/research/nmt/models/en_de_tts/en_de_tts_model.npz.dev.npz'
    # path = '/home/nlp/aharonr6/git/research/nmt/models/de_en_stt/de_en_stt_model.npz.dev.npz'
    rmodel = numpy.load(path)
    history_errs = list(rmodel['history_errs'])
    for i,loss in enumerate(history_errs):
        print i, loss

if __name__ == '__main__':
    main(sys.argv[1])

