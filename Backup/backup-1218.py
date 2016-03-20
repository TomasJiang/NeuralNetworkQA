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


#NOTE: Pick Facts
#      Word2Vector science

####################################################################
##############################  DATA  ##############################
####################################################################

'''
VECTORSIZE  = 300
FACTNUM     = 10
TOPSENTENCE = 10


# TODO: TRY GloVe
# NOTE: word2vec exclude stopwords and punctuation
print('Load Word2Vec Binary File...')
w2v = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# stopwords = nltk.corpus.stopwords.words('english')
# exclude = list(punctuation) + stopwords

# sentence to vector
# NOTE: scipy.spatial.distance.cosine => 1 - cosine(u, v) => range(0, 2)
#       the result smaller, the vectors more similar

def sent_preprocess(sent):
    sent = [word for word in word_tokenize(sent) if word in w2v.vocab]
    return sent


# def preprocess(text):
#     # arg:   text
#     # return sents
#     # exclude stopwords and punctuation
#     # only in word2vec vocab
#     result = []
#     sents = nltk.sent_tokenize(text)
#     length = len(sents)
#     for i in range(length):
#         sent = sent_preprocess(sents[i])
#         if len(sent) > 0:
#             result.append(sent)
#     return result

# print('Preprocess Knowledge Base...')

# knowledge_base_sents = preprocess(knowledge_base_raw)
# sents = nltk.sent_tokenize(raw)

def sentence_to_vector(sent):
    # tokens = [token for token in word_tokenize(sent)
    #           if token in w2v.vocab]
    # sent = sent_preprocess(sent)
    vec = np.zeros(VECTORSIZE, dtype=theano.config.floatX)
    for word in sent:
        vec += w2v[word]
    if len(sent) > 0:
        vec /= len(sent)
    return vec

# def select_topk_vector(target, vecs):
#     # return topk vec, index
#
#     return sorted(vecs, key=lambda x: cosine(target, x))[:TOPSENTENCE]
#     #return vecs[:TOPSENTENCE]

def read_data(fname):
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

def text_to_matrix(text):
    length = len(text)
    res = np.zeros([length, VECTORSIZE], dtype=theano.config.floatX)
    for i in range(length):
        if isinstance(text[i], str):
            text[i] = text[i].decode('utf8')
        res[i] = sentence_to_vector(sent_preprocess(text[i]))
    return res

# train_Q (sample_num, w2v_size)
# train_F (sample_num, fact_num, w2v_size)
# train_A (sample_num, answer_num, w2v_size)
# train_T (sample_num, answer_num) {0, 1}


print('Open Knowledge Base...')
f = codecs.open('Concepts-CK12.txt', 'r', 'utf8')
knowledge_base_raw = f.read()
f.close()
knowledge_base = sent_tokenize(knowledge_base_raw)
knowledge_base_matrix = text_to_matrix(knowledge_base)

print('Reading Text...')
train_question, train_answer, train_target = read_data('training_set.tsv')
train_Q = text_to_matrix(train_question)

# numpy.argsort
# sent_vecs = []
# for sent in knowledge_base_sents:
#     sent_vecs.append(sentence_to_vector(sent))
# knowledge_base_mat = np.matrix(sent_vecs, dtype=theano.config.floatX)

# print('Build Sentence Compare Model...')
# M = T.dmatrix()
# v = T.dvector()
# cosine_result, _ = theano.scan(lambda m: 1 - T.dot(m, v) / (m.norm(2) * v.norm(2)), sequences=[M])
# cosine_similarity_lines = function(inputs=[M, v], outputs=[cosine_result])
# cosine_value = cosine_similarity_lines(sent_mat, target)

'''


#####################################################################
############################## NETWORK ##############################
#####################################################################

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

def Negative_Log_Likelihood(pred, target):
    return T.mean(- target * T.log(pred) - (1 - target) * T.log(1 - pred))

############################## FRONT END ##############################
# Pick Facts

theano.config.optimizer="fast_compile"
theano.config.exception_verbosity="high"

w2v_size            = 300
embedding_size      = 64
sample_num          = 2500
vocab_size          = 1000
question_maxlen     = 80
answer_maxlen       = 15
fact_maxlen         = 40
knowledge_base_size = 11752
answer_num          = 4
fact_num            = 10
batch_size          = 50


knowledge_base_matrix   = np.random.rand(knowledge_base_size, w2v_size).astype(theano.config.floatX)
train_question_matrix   = np.random.rand(sample_num, w2v_size).astype(theano.config.floatX)
train_answers_tensor     = np.random.rand(sample_num, answer_num, w2v_size).astype(theano.config.floatX)
train_target_matrix     = np.random.randint(2, size=(sample_num, answer_num)).astype(theano.config.floatX)

print('Build Facts Matrix...')

Question_Matrix = T.matrix()
# (sample_num, w2v_size)
Answers_Tensor = T.tensor3()
# (sample_num, answer_num, w2v_size)
Target_Matrix = T.matrix()
# (sample_num, answer_num)
Knowledge_Base_Matrix = T.matrix()
# (knowledge_base_size, w2v_size)

Dot_Result_Matrix = T.dot(Question_Matrix, Knowledge_Base_Matrix.T)
Result_Shape   = Dot_Result_Matrix.shape
# (sample_num, knowledge_base_size)

Knowledge_Base_Row_Norm   = Knowledge_Base_Matrix.norm(2, axis=1)
Knowledge_Base_Norm_Matrix = T.tile(Knowledge_Base_Row_Norm, [Result_Shape[0], 1])
# (sample_num, knowledge_base_size)
Question_Row_Norm = Question_Matrix.norm(2, axis=1)
Question_Norm_Matrix = T.tile(Question_Row_Norm, [Result_Shape[1], 1]).T
# (sample_num, knowledge_base_size)
Match_Result_Matrix = 1 - Dot_Result_Matrix / (Question_Norm_Matrix * Knowledge_Base_Norm_Matrix)
Match_Index  = T.argsort(Match_Result_Matrix, axis=1)[:, :fact_num]
# (sample_num, fact_num)
Facts_Tensor = Knowledge_Base_Matrix[Match_Index]
# (sample_num, fact_num, w2v_size)
# TODO: test

# Select_Facts_Model = function(
#         inputs=[Question_Matrix, Knowledge_Base_Matrix],
#         # outputs=[Dot_Result_Matrix]
#         outputs=[Facts_Tensor, Match_Index, Match_Result_Matrix]
#         )

############################## BACK END ##############################

Embedding_B = EmbeddingLayer(Question_Matrix, w2v_size, embedding_size)
Question_Embedding = Embedding_B.output()
# (sample_num, embedding_size)

Embedding_A = EmbeddingLayer(Facts_Tensor, w2v_size, embedding_size)
Facts_Embedding_M = Embedding_A.output()
# (sample_num, fact_num, embedding_size)

Facts_Shape = Facts_Embedding_M.shape
Facts_Embedding_M = T.reshape(Facts_Embedding_M, (Facts_Shape[0] * Facts_Shape[1], Facts_Shape[2]))
# (sample_num * fact_num, embedding_size)

Reshaped_Question_Matrix = T.reshape(T.tile(Question_Embedding, [1, Facts_Shape[1]]),
                           (Facts_Shape[0] * Facts_Shape[1], Facts_Shape[2]))
# (sample_num * fact_num, embedding_size)

Weight_Matrix, _ = theano.scan(lambda u, v: 1 + T.dot(u, v) / (u.norm(2) * v.norm(2)),
                               sequences=[Facts_Embedding_M, Reshaped_Question_Matrix])
# (sample_num * fact_num, 1)
Weight_Matrix = T.reshape(Weight_Matrix, (Facts_Shape[0], Facts_Shape[1]))
# (sample_num, fact_num)

Sum_Of_Weight = T.sum(Weight_Matrix, axis=1)
# (sample_num, 1)
Sum_Of_Weight = T.tile(Sum_Of_Weight, [Facts_Shape[1], 1]).T
Normalized_Weight_Matrix = Weight_Matrix / Sum_Of_Weight
# (sample_num, fact_num)

Embedding_C = EmbeddingLayer(Facts_Tensor, w2v_size, embedding_size)
Facts_Embedding_C = Embedding_C.output()
# (sample_num, fact_num, embedding_size)

Response, _ = theano.scan(lambda m, v: T.dot(m.T, v),
                          sequences=[Facts_Embedding_C, Normalized_Weight_Matrix])
# (sample_num, embedding_size)

Embedding_D = EmbeddingLayer(Answers_Tensor, w2v_size, embedding_size)
Answers_Embedding = Embedding_D.output()
# (sample_num, answer_num, embedding_size)

Answers_Shape = Answers_Embedding.shape
Reshaped_Answers_Embedding = T.reshape(Answers_Embedding, [Answers_Shape[0] * Answers_Shape[1], Answers_Shape[2]])
# (sample_num * answer_num, embedding_size)

Repeated_Response = T.tile(Response, [1, Answers_Shape[1]])
# Repeated_Response = T.reshape(repeated_response, answers_shape)
Repeated_Response = T.reshape(Repeated_Response, [Answers_Shape[0] * Answers_Shape[1], Answers_Shape[2]])
# (sample_num * answer_num, embedding_size)

Temp_Result, _ = theano.scan(lambda u, v: 1 + T.dot(u, v) / (u.norm(2) * v.norm(2)),
                        sequences=[Reshaped_Answers_Embedding, Repeated_Response])
Temp_Result = T.reshape(Temp_Result, [Answers_Shape[0], Answers_Shape[1]])
# (sample_num, answer_num)

Result = T.nnet.softmax(Temp_Result)
# (sample_num, answer_num)

Cost = Negative_Log_Likelihood(Result, Target_Matrix)

print 'Model Build Completed'
# Compute_Match = function(
#         inputs=[Question_Matrix, Answers_Tensor, Target_Matrix, Knowledge_Base_Matrix],
#         # outputs=[Facts_Embedding_M, Reshaped_Question_Matrix]
#         outputs=[Cost, Result, Temp_Result, Repeated_Response, Reshaped_Answers_Embedding]
#         )
#
# res = Compute_Match(train_question_matrix, train_answers_tensor, train_target_matrix, knowledge_base_matrix)
# (sample_num, facts_num, w2v_size)


# a = np.arange(12).reshape(3, 4).astype(theano.config.floatX)
# b = np.arange(20).reshape(5, 4).astype(theano.config.floatX)
# res = Compute_Match(a, b)
# ssert res.shape = (3, 5)

# train_Q = theano.shared(np.random.rand(sample_num, w2v_size))
# train_F = theano.shared(np.random.rand(sample_num, fact_num, w2v_size))
# train_A = theano.shared(np.random.rand(sample_num, answer_num, w2v_size))
# train_T = theano.shared(np.random.randint(2, size=(sample_num, answer_num))
#                          .astype(theano.config.floatX))

############################## MEMN2N ##############################


'''
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

# facts_embedding_m = T.reshape(T.tile(Question_Embedding, [1, Facts_Shape[1]]),
#                            (Facts_Shape[0] * Facts_Shape[1], Facts_Shape[2]))
# (sample_num * fact_num, embedding_size)

# match_result, _ = theano.scan(lambda u, v: 1 + T.dot(u, v) / (u.norm(2) * v.norm(2)),
#                               sequences=[facts_embedding_m, match_question])
# # (sample_num * fact_num, 1)
# match_result = T.reshape(match_result, (facts_shape[0], facts_shape[1]))
# # (sample_num, fact_num)

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
