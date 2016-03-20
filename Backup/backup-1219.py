import numpy as np
import timeit
import nltk
import codecs
import theano
import theano.tensor as T
import cPickle as pickle
import os

from gensim.models import Word2Vec
from nltk import word_tokenize, sent_tokenize
from string import punctuation
from scipy.spatial.distance import cosine
from theano import function


####################################################################
##############################  DATA  ##############################
####################################################################

w2v_size            = 300
embedding_size      = 64
answer_num          = 4
fact_num            = 10

# Parameters
batch_size          = 50
learning_rate       = 0.4
n_epoch             = 25
patience            = 5000
patience_increase   = 2
improvement_threshold = 0.995
data_split_propotion  = np.array([40, 5, 5])

reload_data = True

# for generate test data
sample_num          = 2500
knowledge_base_size = 11752



# TODO: TRY GloVe
# NOTE: word2vec exclude stopwords and punctuation
def sent_preprocess(sent):
    sent = [word for word in word_tokenize(sent) if word in w2v.vocab]
    return sent

def sentence_to_vector(sent):
    # tokens = [token for token in word_tokenize(sent)
    #           if token in w2v.vocab]
    # sent = sent_preprocess(sent)
    vec = np.zeros(w2v_size, dtype=theano.config.floatX)
    for word in sent:
        vec += w2v[word]
    if len(sent) > 0:
        vec /= len(sent)
    return vec

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
    res = np.zeros([length, w2v_size], dtype=theano.config.floatX)
    for i in range(length):
        if isinstance(text[i], str):
            text[i] = text[i].decode('utf8')
        res[i] = sentence_to_vector(sent_preprocess(text[i]))
    return res

def answers_to_tensor(answers):
    answer_num = 4
    length = len(answers)
    res = np.zeros([length, answer_num, w2v_size], dtype=theano.config.floatX)
    for i in range(length):
        for j in range(answer_num):
            if isinstance(answers[i][j], str):
                answers[i][j] = answers[i][j].decode('utf8')
            res[i][j] = sentence_to_vector(sent_preprocess(answers[i][j]))
    return res

def target_to_matrix(target):
    answer_num = 4
    length = len(target)
    res = np.zeros([length, answer_num], dtype=theano.config.floatX)
    for i in range(length):
        res[i][ord(target[i]) - ord('A')] = 1.0
    return res

def split_data(numpy_tensor, proportion):
    num = numpy_tensor.shape[0]
    proportion = num * (proportion.astype(float) / np.sum(proportion))
    split_index = np.cumsum(proportion).astype(int)
    return np.split(numpy_tensor, split_index[:-1])


knoledge_base = None
split_question = []
split_answers  = []
split_target   = []

data_file      = 'training_set.tsv'
data_save_path = 'data.save'

# # For Adjust Parameters
# data_file      = 'test_dataset.tsv'
# data_save_path = 'test_data.save'

if os.path.exists(data_save_path) and not reload_data:
    print 'Load Data From', data_save_path
    f = file(data_save_path, 'rb')
    split_question = pickle.load(f)
    split_answers  = pickle.load(f)
    split_target   = pickle.load(f)
    knowledge_base = pickle.load(f)
    f.close()
else:
    print('Load Word2Vec Binary File...')
    w2v = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    print('Load Knowledge Base...')
    f = codecs.open('Concepts-CK12.txt', 'r', 'utf8')
    knowledge_base_raw = f.read()
    f.close()
    knowledge_base_text = sent_tokenize(knowledge_base_raw)
    knowledge_base = text_to_matrix(knowledge_base_text)

    print('Load Data...')
    question_text, answer_text, target_text = read_data(data_file)

    split_question  = split_data(text_to_matrix(question_text), np.array(data_split_propotion))
    split_answers   = split_data(answers_to_tensor(answer_text), np.array(data_split_propotion))
    split_target    = split_data(target_to_matrix(target_text), np.array(data_split_propotion))

    print 'Save Data To', data_save_path
    f = file(data_save_path, 'wb')
    pickle.dump(split_question, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(split_answers, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(split_target, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(knowledge_base, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

knowledge_base_matrix   = theano.shared(knowledge_base.astype(theano.config.floatX))


train_question      = split_question[0]
valid_question      = split_question[1]
test_question       = split_question[2]

train_answers       = split_answers[0]
valid_answers       = split_answers[1]
test_answers        = split_answers[2]

train_target        = split_target[0]
valid_target        = split_target[1]
test_target         = split_target[2]

train_question_matrix   = theano.shared(train_question.astype(theano.config.floatX))
train_answers_tensor    = theano.shared(train_answers.astype(theano.config.floatX))
train_target_matrix     = theano.shared(train_target.astype(theano.config.floatX))

valid_question_matrix   = theano.shared(valid_question.astype(theano.config.floatX))
valid_answers_tensor    = theano.shared(valid_answers.astype(theano.config.floatX))
valid_target_matrix     = theano.shared(valid_target.astype(theano.config.floatX))

test_question_matrix   = theano.shared(test_question.astype(theano.config.floatX))
test_answers_tensor    = theano.shared(test_answers.astype(theano.config.floatX))
test_target_matrix     = theano.shared(test_target.astype(theano.config.floatX))

print('Load Data Completed')
print
print '  train_question: ', train_question.shape
print '  train_answers:  ', train_answers.shape
print '  train_target:   ', train_target.shape
print '  valid_question: ', valid_question.shape
print '  valid_answers:  ', valid_answers.shape
print '  valid_target:   ', valid_target.shape
print '  test_question:  ', test_question.shape
print '  test_answers:   ', test_answers.shape
print '  test_target:    ', test_target.shape
print


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

print('Build Model...')

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

############################## MEMN2N ##############################

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

Predict_Answer = T.argmax(Result, axis=1)
# (sample_num)

Right_Num = T.sum(T.eq(Predict_Answer, T.argmax(Target_Matrix)))
Right_Percentage = T.cast(Right_Num, theano.config.floatX) / Target_Matrix.shape[0]

'''
############################## DEBUG ##############################

debug_knowledge_base_matrix   = np.random.rand(knowledge_base_size, w2v_size).astype(theano.config.floatX)
debug_question_matrix   = np.random.rand(sample_num, w2v_size).astype(theano.config.floatX)
debug_answers_tensor    = np.random.rand(sample_num, answer_num, w2v_size).astype(theano.config.floatX)
debug_target_matrix     = np.random.randint(2, size=(sample_num, answer_num)).astype(theano.config.floatX)

# Compute_Match = function(
#         inputs=[Question_Matrix, Answers_Tensor, Target_Matrix, Knowledge_Base_Matrix],
#         # outputs=[Facts_Embedding_M, Reshaped_Question_Matrix]
#         outputs=[Cost, Result, Temp_Result, Repeated_Response, Reshaped_Answers_Embedding]
#         )
#
# res = Compute_Match(train_question_matrix, train_answers_tensor, train_target_matrix, knowledge_base_matrix)
# (sample_num, facts_num, w2v_size)

debug_Q = theano.shared(debug_question_matrix)
debug_A = theano.shared(debug_answers_tensor)
debug_T = theano.shared(debug_target_matrix)
debug_K = theano.shared(debug_knowledge_base_matrix)

index = T.lscalar()
debug_model = theano.function(
         inputs=[index],
         outputs=[Cost, Right_Num, Right_Percentage],
         givens={
             Question_Matrix:        debug_Q[index * batch_size: (index + 1) * batch_size],
             Answers_Tensor:         debug_A[index * batch_size: (index + 1) * batch_size],
             Target_Matrix:          debug_T[index * batch_size: (index + 1) * batch_size],
             Knowledge_Base_Matrix:  debug_K[index * batch_size: (index + 1) * batch_size]
             },
        on_unused_input='warn'
        )

output = debug_model(0)


'''

##################################################################

# TODO
# knowledge_base  = theano.shared(np.random.rand(knowledge_base_size, w2v_size)
#                         .astype(theano.config.floatX))
#
# train_question  = theano.shared(np.random.rand(sample_num, w2v_size)
#                         .astype(theano.config.floatX))
# train_answers   = theano.shared(np.random.rand(sample_num, answer_num, w2v_size)
#                         .astype(theano.config.floatX))
# train_target    = theano.shared(np.random.randint(2, size=(sample_num, answer_num))
#                         .astype(theano.config.floatX))
#
# test_question   = theano.shared(np.random.rand(sample_num, w2v_size)
#                         .astype(theano.config.floatX))
# test_answers    = theano.shared(np.random.rand(sample_num, answer_num, w2v_size)
#                         .astype(theano.config.floatX))
# test_target     = theano.shared(np.random.randint(2, size=(sample_num, answer_num))
#                         .astype(theano.config.floatX))
#
# valid_question  = theano.shared(np.random.rand(sample_num, w2v_size)
#                         .astype(theano.config.floatX))
# valid_answers   = theano.shared(np.random.rand(sample_num, answer_num, w2v_size)
#                         .astype(theano.config.floatX))
# valid_target    = theano.shared(np.random.randint(2, size=(sample_num, answer_num))
#                         .astype(theano.config.floatX))

Index = T.lscalar()

test_model = theano.function(
        inputs=[Index],
        outputs=[Cost, Right_Percentage],
        givens={
             Question_Matrix:       test_question_matrix[Index * batch_size: (Index + 1) * batch_size],
             Answers_Tensor:        test_answers_tensor[Index * batch_size: (Index + 1) * batch_size],
             Target_Matrix:         test_target_matrix[Index * batch_size: (Index + 1) * batch_size],
             Knowledge_Base_Matrix: knowledge_base_matrix
            }
        )

valid_model = theano.function(
        inputs=[Index],
        outputs=Cost,
        givens={
             Question_Matrix:       valid_question_matrix[Index * batch_size: (Index + 1) * batch_size],
             Answers_Tensor:        valid_answers_tensor[Index * batch_size: (Index + 1) * batch_size],
             Target_Matrix:         valid_target_matrix[Index * batch_size: (Index + 1) * batch_size],
             Knowledge_Base_Matrix: knowledge_base_matrix
            }
        )



G_EmbeddingA_W = T.grad(cost=Cost, wrt=Embedding_A.W)
G_EmbeddingA_b = T.grad(cost=Cost, wrt=Embedding_A.b)
G_EmbeddingB_W = T.grad(cost=Cost, wrt=Embedding_B.W)
G_EmbeddingB_b = T.grad(cost=Cost, wrt=Embedding_B.b)
G_EmbeddingC_W = T.grad(cost=Cost, wrt=Embedding_C.W)
G_EmbeddingC_b = T.grad(cost=Cost, wrt=Embedding_C.b)
G_EmbeddingD_W = T.grad(cost=Cost, wrt=Embedding_D.W)
G_EmbeddingD_b = T.grad(cost=Cost, wrt=Embedding_D.b)

updates = [(Embedding_A.W, Embedding_A.W - learning_rate * G_EmbeddingA_W),
           (Embedding_A.b, Embedding_A.b - learning_rate * G_EmbeddingA_b),
           (Embedding_B.W, Embedding_B.W - learning_rate * G_EmbeddingB_W),
           (Embedding_B.b, Embedding_B.b - learning_rate * G_EmbeddingB_b),
           (Embedding_C.W, Embedding_C.W - learning_rate * G_EmbeddingC_W),
           (Embedding_C.b, Embedding_C.b - learning_rate * G_EmbeddingC_b),
           (Embedding_D.W, Embedding_D.W - learning_rate * G_EmbeddingD_W),
           (Embedding_D.b, Embedding_D.b - learning_rate * G_EmbeddingD_b)
    ]

train_model = theano.function(
        inputs=[Index],
        outputs=Cost,
        updates=updates,
        givens={
             Question_Matrix:       train_question_matrix[Index * batch_size: (Index + 1) * batch_size],
             Answers_Tensor:        train_answers_tensor[Index * batch_size: (Index + 1) * batch_size],
             Target_Matrix:         train_target_matrix[Index * batch_size: (Index + 1) * batch_size],
             Knowledge_Base_Matrix: knowledge_base_matrix
            }
        )

n_train_batches = train_question_matrix.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_question_matrix.get_value(borrow=True).shape[0] / batch_size
n_test_batches  =  test_question_matrix.get_value(borrow=True).shape[0] / batch_size

print 'Build Model Completed'

############################## TRAIN ##############################

print
print 'Training The Model...'

best_validation_loss = np.inf
validation_frequency = min(n_train_batches, patience / 2)
mean_test_loss = 0.
start_time = timeit.default_timer()
done_looping = False
epoch = 0

while (epoch < n_epoch) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)

        iter = (epoch - 1) * n_train_batches + minibatch_index
        if (iter + 1) % validation_frequency == 0:
            validation_losses = [valid_model(i)
                                 for i in xrange(n_valid_batches)]
            mean_validation_loss = np.mean(validation_losses)
            print(
                'epoch %i, minibatch %i/%i, validation loss %f' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    mean_validation_loss
                )
            )

            if mean_validation_loss < best_validation_loss:
                if mean_validation_loss < best_validation_loss * \
                    improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = mean_validation_loss
                test_result     = [test_model(i)
                                   for i in xrange(n_test_batches)]
                test_losses     = [test_result[i][0]
                                   for i in xrange(n_test_batches)]
                right_percent   = [test_result[i][1]
                                   for i in xrange(n_test_batches)]
                mean_test_loss = np.mean(test_losses)
                mean_right_percent = np.mean(right_percent)

                print(
                    (
                    '   epoch %i, minibatch %i/%i, test loss of'
                    ' best model %f, right percentage %f'
                    ) %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        mean_test_loss,
                        mean_right_percent
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
        'with test loss %f'
    )
    % (best_validation_loss, mean_test_loss)
)

print 'The code run for %d epochs, with %f epochs/sec' % (
    epoch, 1. * epoch / (end_time - start_time))

