import numpy as np
import timeit
import nltk
import codecs
import theano
import theano.tensor as T

from gensim.models import Word2Vec
from nltk import word_tokenize, sent_tokenize
from string import punctuation
from scipy.spatial.distance import cosine
from theano import function

'''

####################################################################
##############################  DATA  ##############################
####################################################################

VECTORSIZE  = 300
FACTNUM     = 10
TOPSENTENCE = 10

f = codecs.open('Concepts-CK12.txt', 'r', 'utf8')
raw = f.read()
f.close()

# TODO: TRY GloVe
# NOTE: word2vec exclude stopwords and punctuation
w2v = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# stopwords = nltk.corpus.stopwords.words('english')
# exclude = list(punctuation) + stopwords

# sentence to vector
# NOTE: scipy.spatial.distance.cosine => 1 - cosine(u, v) => range(0, 2)
#       the result smaller, the vectors more similar

def sent_preprocess(sent):
    sent = [word for word in word_tokenize(sent) if word in w2v.vocab]
    return sent

def preprocess(text):
    # arg:   text
    # return sents
    # exclude stopwords and punctuation
    # only in word2vec vocab
    result = []
    sents = nltk.sent_tokenize(text)
    length = len(sents)
    for i in range(length):
        sent = sent_preprocess(sents[i])
        if len(sent) > 0:
            result.append(sent)
    return result


sents = preprocess(raw)
# sents = nltk.sent_tokenize(raw)

def sent_to_vec(sent):
    # tokens = [token for token in word_tokenize(sent)
    #           if token in w2v.vocab]
    # sent = sent_preprocess(sent)
    vec = np.zeros(VECTORSIZE, dtype=np.float32)
    for word in sent:
        vec += w2v[word]
    vec /= len(sent)
    return vec

def select_topk_vec(target, vecs):
    # return topk vec, index

    return sorted(vecs, key=lambda x: cosine(target, x))[:TOPSENTENCE]
    #return vecs[:TOPSENTENCE]

def read_text(fname):
    Q_str = []
    A_str = []
    Y_str = []

    f = open(fname, 'r')
    f.readline()

    for line in f:
        _, q, y, a1, a2, a3, a4 = line.split('\t')
        a = [a1, a2, a3, a4.strip()]
        Q_str.append(q)
        A_str.append(a)
        Y_str.append(y)

    return Q_str, A_str, Y_str

query_str, answer_str, target_str = read_text('training_set.tsv')
# NOTE: does question need sentence segmentation?
# q = sent_preprocess(Q_str[0])
# q_vec = sent_to_vec(q)

# sent_vecs = [] # idx vec
# topk_sent = []

# length = len(sents)
# for i in range(length):
#     sent_vecs.append({'idx':i, 'vec':sent_to_vec(sents[i])})

# for sent in sents:
#     sent_vecs.append(sent_to_vec(sent))
#
# topk = select_topk_vec(q_vec, sent_vecs)
# idx = []
# for vec in topk:
#     length = len(sent_vecs)
#     for i in range(length):
#         if np.array_equal(vec, sent_vecs[i]):
#             idx.append(i)
#
# facts = [sents[ix] for ix in idx]

# sent_vecs_with_index = []
# length = len(sents)
# for i in range(length):
#     sent_vecs_with_index.append({'idx': i, 'vec' : sent_to_vec(sents[i])})
#
# def select_facts(target, sent_vecs_with_index):
#     # return indexs
#     target_vec = sent_to_vec(sent_preprocess(target))
#     # TODO: parallel
#     cosine_value = map(lambda x: cosine(target_vec, x['vec']), sent_vecs_with_index)
#     # topk = sorted(sent_vecs_with_index, key=lambda x: cosine(target_vec, x['vec']))[:TOPSENTENCE]
#     return [sent['idx'] for sent in topk]

# numpy.argsort
sent_vecs = []
for sent in sents:
    sent_vecs.append(sent_to_vec(sent))

sent_mat = np.matrix(sent_vecs, dtype=theano.config.floatX)

M = T.dmatrix()
v = T.dvector()
cosine_result, _ = theano.scan(lambda m: 1 - T.dot(m, v) / (m.norm(2) * v.norm(2)), sequences=[M])
cosine_similarity_lines = function(inputs=[M, v], outputs=[cosine_result])
# cosine_value = cosine_similarity_lines(sent_mat, target)

def select_facts(ques, sents, sent_mat, select_func):
    target = sent_to_vec(sent_preprocess(ques))
    select_value = select_func(sent_mat, target)[0]
    sort_index = np.argsort(select_value)
    return [sents[sort_index[i]] for i in range(FACTNUM)]

'''

#####################################################################
############################## NETWORK ##############################
#####################################################################

############################## MEMN2N ##############################



# INPUT:    train_Q (sample_num, question_maxlen, w2v_size)
# OUTPUT:   (sample_num, question_maxlen, embedding_size)

# INPUT:    train_A (sample_num, answer_maxlen*answer_num, w2v_size)
# OUTPUT:   (sample_num, answer_maxlen*answer, embedding_size)

# INPUT:    train_F (sample_num, fact_maxlen*fact_num, w2v_size)
# OUTPUT:   (sample_num, question_maxlen, embedding_size)

class EmbeddingLayer(object):
    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(
                value = np.random.uniform(
                    -0.1, 0.1, (n_in, n_out)
                    ).astype(theano.config.floatX)
                )

        self.b = theano.shared(
                value = np.random.uniform(
                    -0.1, 0.1, (n_out, )
                    ).astype(theano.config.floatX)
                )

        self.params = [self.W, self.b]
        self.input = input
    def output(self):
        return T.dot(self.input, self.W) + self.b


w2v_size = 30
sample_num = 2500
vocab_size = 10
embedding_size = 64
question_maxlen = 20
answer_maxlen = 15
fact_maxlen = 40
answer_num = 4
fact_num = 10
batch_size = 50

question_tensor = T.matrix()
embedding_B = EmbeddingLayer(question_tensor, w2v_size, embedding_size)
question_embedding = embedding_B.output()
# (sample_num, embedding_size)

facts_tensor = T.tensor3()
embedding_A = EmbeddingLayer(facts_tensor, w2v_size, embedding_size)
facts_embedding_m = embedding_A.output()
# (sample_num, fact_num, embedding_size)

facts_shape = facts_embedding_m.shape
facts_embedding_m = T.reshape(facts_embedding_m, (facts_shape[0] * facts_shape[1], facts_shape[2]))
# (sample_num * fact_num, embedding_size)

match_question = T.reshape(T.tile(question_embedding, [1, facts_shape[1]]),
                           (facts_shape[0] * facts_shape[1], facts_shape[2]))
# (sample_num * fact_num, embedding_size)

match_result, _ = theano.scan(lambda u, v: 1 + T.dot(u, v) / (u.norm(2) * v.norm(2)),
                              sequences=[facts_embedding_m, match_question])
# (sample_num * fact_num, 1)
match_result = T.reshape(match_result, (facts_shape[0], facts_shape[1]))
# (sample_num, fact_num)

sum_of_match_result = T.sum(match_result, 1)
# (sample_num, 1)
sum_of_match_result = T.tile(sum_of_match_result, [facts_shape[1], 1]).T
match_result = match_result / sum_of_match_result
# (sample_num, fact_num)


embedding_C = EmbeddingLayer(facts_tensor, w2v_size, embedding_size)
facts_embedding_c = embedding_C.output()
# (sample_num, fact_num, embedding_size)

response, _ = theano.scan(lambda m, v: T.dot(m.T, v),
                          sequences=[facts_embedding_c, match_result])
# (sample_num, embedding_size)


answers_tensor = T.tensor3()
embedding_D = EmbeddingLayer(answers_tensor, w2v_size, embedding_size)
answers_embedding = embedding_D.output()

answers_shape = answers_embedding.shape
answers_embedding = T.reshape(answers_embedding, [answers_shape[0] * answers_shape[1], answers_shape[2]])
# (sample_num, answer_num, embedding_size)

repeated_response = T.tile(response, [1, answers_shape[1]])
repeated_response = T.reshape(repeated_response, answers_shape)
repeated_response = T.reshape(repeated_response, [answers_shape[0] * answers_shape[1], answers_shape[2]])
# (sample_num, answer_num, embedding_size)

temp_result, _ = theano.scan(lambda u, v: 1 + T.dot(u, v) / (u.norm(2) * v.norm(2)),
                        sequences=[answers_embedding, repeated_response])
temp_result = T.reshape(temp_result, [answers_shape[0], answers_shape[1]])
# (sample_num, answer_num)

result = T.nnet.softmax(temp_result)
# (sample_num, answer_num)

def negative_log_likelihood(pred, target):
    return T.mean(- target * T.log(pred) - (1 - target) * T.log(1 - pred))

target_tensor = T.matrix()
cost = negative_log_likelihood(result, target_tensor)

'''
############################## TEST ##############################

train_Q = theano.shared(np.random.rand(sample_num, w2v_size))
train_F = theano.shared(np.random.rand(sample_num, fact_num, w2v_size))
train_A = theano.shared(np.random.rand(sample_num, answer_num, w2v_size))
train_T = theano.shared(np.random.randint(2, size=(sample_num, answer_num))
                        .astype(theano.config.floatX))

index = T.lscalar()
model = theano.function(
         inputs=[index],
         outputs=[cost, result, target_tensor],
         givens={
             question_tensor:   train_Q[index * batch_size: (index + 1) * batch_size],
             facts_tensor:      train_F[index * batch_size: (index + 1) * batch_size],
             answers_tensor:    train_A[index * batch_size: (index + 1) * batch_size],
             target_tensor:     train_T[index * batch_size: (index + 1) * batch_size]
             },
        on_unused_input='warn'
        )

output = model(0)
'''

############################## TRAIN ##############################


# TODO
train_Q = theano.shared(np.random.rand(sample_num, w2v_size))
train_F = theano.shared(np.random.rand(sample_num, fact_num, w2v_size))
train_A = theano.shared(np.random.rand(sample_num, answer_num, w2v_size))
train_T = theano.shared(np.random.randint(2, size=(sample_num, answer_num))
                        .astype(theano.config.floatX))

test_Q = theano.shared(np.random.rand(sample_num, w2v_size))
test_F = theano.shared(np.random.rand(sample_num, fact_num, w2v_size))
test_A = theano.shared(np.random.rand(sample_num, answer_num, w2v_size))
test_T = theano.shared(np.random.randint(2, size=(sample_num, answer_num))
                        .astype(theano.config.floatX))

valid_Q = theano.shared(np.random.rand(sample_num, w2v_size))
valid_F = theano.shared(np.random.rand(sample_num, fact_num, w2v_size))
valid_A = theano.shared(np.random.rand(sample_num, answer_num, w2v_size))
valid_T = theano.shared(np.random.randint(2, size=(sample_num, answer_num))
                        .astype(theano.config.floatX))


index = T.lscalar()
test_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
             question_tensor:   test_Q[index * batch_size: (index + 1) * batch_size],
             facts_tensor:      test_F[index * batch_size: (index + 1) * batch_size],
             answers_tensor:    test_A[index * batch_size: (index + 1) * batch_size],
             target_tensor:     test_T[index * batch_size: (index + 1) * batch_size]
            }
        )

validate_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
             question_tensor:   test_Q[index * batch_size: (index + 1) * batch_size],
             facts_tensor:      test_F[index * batch_size: (index + 1) * batch_size],
             answers_tensor:    test_A[index * batch_size: (index + 1) * batch_size],
             target_tensor:     test_T[index * batch_size: (index + 1) * batch_size]
            }
        )


# TODO
learning_rate = 0.1
g_W = T.grad(cost=cost, wrt=embedding_A.W)
updates = [(embedding_A.W, embedding_A.W - learning_rate * g_W)]

train_model = theano.function(
         inputs=[index],
         outputs=[cost, result],
         updates=updates,
         givens={
             question_tensor:   valid_Q[index * batch_size: (index + 1) * batch_size],
             facts_tensor:      valid_F[index * batch_size: (index + 1) * batch_size],
             answers_tensor:    valid_A[index * batch_size: (index + 1) * batch_size],
             target_tensor:     valid_T[index * batch_size: (index + 1) * batch_size]
             }
         )

n_train_batches = train_Q.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_Q.get_value(borrow=True).shape[0] / batch_size
n_test_batches  =  test_Q.get_value(borrow=True).shape[0] / batch_size

#TODO

n_epoch = 4
print '...training the model'

patience = 5000
patience_increase = 2
improvement_threshold = 0.995
validation_frequency = min(n_train_batches, patience / 2)


best_validation_loss = np.inf
test_score = 0.
start_time = timeit.default_timer()

done_looping = False
epoch = 0
while (epoch < n_epoch) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)

        iter = (epoch - 1) * n_train_batches + minibatch_index
        if (iter + 1) % validation_frequency == 0:
            validation_losses = [validate_model(i)
                                 for i in xrange(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
            print(
                'epoch %i, minibatch %i/%i, validation error %f' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss
                )
            )

            if this_validation_loss < best_validation_loss:
                if this_validation_loss < best_validation_loss * \
                    improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                test_losses = [test_model(i)
                               for i in xrange(n_test_batches)]
                test_score = np.mean(test_losses)

                print(
                    (
                    '   epoch %i, minibatch %i/%i, test error of'
                    ' best model %f'
                    ) %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_score
                    )
                )

                # TODO
                # with open('best_model.pkl', 'w') as f:
                #     cPickle.dump([params], f)

        if patience <= iter:
            done_looping = True
            break

end_time = timeit.default_timer()
print(
    (
        'Optimization complete with best validation score of %f,'
        'with test performance %f'
    )
    % (best_validation_loss, test_score)
)

print 'The code run for %d epochs, with %f epochs/sec' % (
    epoch, 1. * epoch / (end_time - start_time))






'''
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Merge, Dense



sample_num = 100
vocab_size = 10
embedding_size = 64
question_maxlen = 20
answer_maxlen = 15
fact_maxlen = 40
answer_num = 4
fact_num = 10

# INPUT:    train_Q (sample_num, question_maxlen)
# OUTPUT:   (sample_num, question_maxlen, embedding_size)
# NOTE: content word_index
train_Q = np.random.randint(vocab_size, size=(sample_num, question_maxlen))
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=embedding_size,
                               input_length=question_maxlen))

# INPUT:    train_A (sample_num, answer_maxlen*answer_num)
# OUTPUT:   (sample_num, answer_maxlen*answer_num, embedding_size)
# NOTE: content word_index
#       Question (*4) really work?

train_A = np.random.randint(vocab_size, size=(sample_num, answer_maxlen*answer_num))
answer_encoder = Sequential()
answer_encoder.add(Embedding(input_dim=vocab_size,
                             output_dim=embedding_size*answer_num,
                             input_length=answer_maxlen*answer_num))

# INPUT:    train_A (sample_num, fact_maxlen*fact_num)
# OUTPUT:   (sample_num, fact_maxlen*fact_num, embedding_size)
# NOTE: content word_index
#       Question (*4) really work?
train_F = np.random.randint(vocab_size, size=(sample_num, fact_maxlen*fact_num))
fact_encoder_m = Sequential()
fact_encoder_m.add(Embedding(input_dim=vocab_size,
                               output_dim=embedding_size*fact_num,
                               input_length=fact_maxlen*fact_num))


fact_encoder_c = Sequential()
fact_encoder_c.add(Embedding(input_dim=vocab_size,
                               output_dim=embedding_size*fact_num,
                               input_length=fact_maxlen*fact_num))


identity_m = Sequential()
identity_m.add(Dense(fact_maxlen, init='identity', input_shape=(fact_maxlen,)))
identitys_m = []
for i in range(4):
    identitys_m.append(identity_m)



embeddings_m = []
for i in range(4):
    embedding_m = Sequential()
    embedding_m.add(Embedding(input_dim=vocab_size,
                              output_dim=embedding_size,
                              input_length=fact_maxlen))
    embeddings_m.append(embedding_m)

model = Sequential()
model.add(Merge(embeddings_m, mode='sum'))

model.compile(optimizer='sgd', loss='mse')

x = []
for i in range(4):
    x.append(np.random.randint(vocab_size, size=(sample_num, fact_maxlen)))

res = model.predict(x)



# inputb = Sequential()
#
# model = Sequential()
# model.add(Merge([inputa, inputb], mode='dot', dot_axes=[(2,), (2,)]))
# model.compile(optimizer='sgd', loss='mse')
#
# a = np.random.randint(100, 80, 64)
# b = np.random.randint(100, 30, 64)
# res = model.predict([a, b])

#answer_encoder.compile(optimizer='sgd', loss='mse')
#res = answer_encoder.predict(train_A)

class RNN(object):
    def __init__(self, nh, nc, ne, de, cs):
        self.emb = theano.shared(
                value=0.2 * np.random.uniform(-1.0, 1.0, (ne+1, de),
                dtype=theano.config.floatX)
                )

        self.wx = theano.shared(
                value=0.2 * np.random.uniform(-1.0, 1.0, (de * ce, nh),
                dtype=theano.config.floatX)
                )

        self.wh = theano.shared(
                value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh),
                dtype=theano.config.floatX)
                )

        self.w = theano.shared(
                value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nc),
                dtype=theano.config.floatX)
                )

        self.bh = theano.shared(
                value=numpy.zeros(nh, dtype=theano.config.floatX)
                )

        self.b = theano.shared(
                value=numpy.zeros(nc, dtype=theano.config.floatX)
                )

        self.h0 = theano.shared(
                value=numpy.zeros(nh, dtype=theano.config.floatX)
                )

        self.params = [self.emb, self.wx, self.wh, self.w, self.bh, self.b, self.h0]



'''
