import numpy

#saveto = '/home/nlp/aharonr6/git/research/nmt/models/en_de_tts/en_de_tts_model.npz.dev.npz'
saveto = '/home/nlp/aharonr6/git/research/nmt/models/de_en_stt/de_en_stt_model.npz.dev.npz'
rmodel = numpy.load(saveto)
history_errs = list(rmodel['history_errs'])
for i,loss in enumerate(history_errs):
	print i, loss

