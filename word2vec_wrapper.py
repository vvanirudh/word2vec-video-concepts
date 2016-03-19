import gensim.models
import time
time1 = time.time()

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


modelbase = gensim.models.Word2Vec()
sentences2 = gensim.models.word2vec.Sentences("text8-queen")
modelbase.build_vocab(sentences2)
modelbase.train(sentences2)
modelbase.save_word2vec_format("wordvectors/model-text8-queen-only")
modelbase.accuracy("questions-words.txt")

model = gensim.models.Word2Vec()
sentences = gensim.models.word2vec.Sentences("text8-rest")
model.build_vocab(sentences)
model.train(sentences)
model.save_word2vec_format("model-text8-rest")
model.accuracy("questions-words.txt")

sentences2 = gensim.models.word2vec.Sentences("text8-queen")
model.update_vocab(sentences2)
model.train(sentences2)
model.save_word2vec_format("wordvectors/model-text8-queen")
model.accuracy("questions-words.txt")

model1 = gensim.models.Word2Vec()
sentences = gensim.models.word2vec.Sentences("text8-all")
model1.build_vocab(sentences)
model1.train(sentences)
model1.save_word2vec_format("wordvectors/model-text8-all")
model1.accuracy("questions-words.txt")
print ("total time: %s" % (time.time() - time1))
