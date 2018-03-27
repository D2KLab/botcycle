import json
import collections
import plac
import sklearn_crfsuite
from sklearn.externals import joblib
from sklearn.metrics import f1_score

# only for word vector features
import spacy

def load_data(test_type='test'):
    path = '../data/kvret/preprocessed/'

    with open(path + 'fold_train.json') as json_file:
        train = json.load(json_file)
    
    if test_type == 'validate':
        test_file = 'final_test.json'
    else:
        test_file = 'fold_test.json'

    with open(path + test_file) as json_file:
        test = json.load(json_file)

    return train, test

def prepare_data(dataset, features='words', nlp=None):
    # get the single sequence of words and intent (duplicating by words in the sentence)
    # words will store all the [words] concatenated
    # intents is the list of (duplicated by the number of words for each sentence) intents
    # slots is the list of annotate IOB
    # bounds is a list of integer containing the indexes for the original sentence split
    sessions = dataset['data']
    result = {'words': [], 'intents':[], 'slots': [], 'bounds':[]}
    last_bound = 0
    for s in sessions:
        for m in s:
            if m['turn'] == 'b':
                pass # agent turn are not considered
            if m['turn'] == 'u' and m['length']:
                if features == 'words':
                    # only consider the lowercase word as feature
                    word_features = [{'w': w.lower()} for w in m['words']]
                elif features == 'word_vectors':
                    # consider the word vector as features
                    # rebuild the original sentence
                    sentence = ' '.join(m['words']).replace(' \'', '\'')
                    doc = nlp(sentence)
                    if len(doc) != m['length']:
                        # this should never happen, tokenization problem
                        print('different length: {} doc {} length. Sentence: {}'.format(len(doc), m['length'], sentence))
                        m['length'] = len(doc)
                    word_features = []
                    for w in doc:
                        word_vector = w.vector.tolist()
                        # consider each vector component
                        word_features.append({'v{}'.format(key) : value for (key, value) in enumerate(word_vector)})
                result['words'] += word_features
                # duplicate the intent value for each word in the sentence
                result['intents'] += [m['intent']] * m['length']
                result['slots'] += m['slots']
                # compute the slice indexes on the resulting array for this sentence
                new_bound = last_bound + m['length']
                result['bounds'] += [(last_bound, new_bound)]
                last_bound = new_bound
    
    print('{} sentences {} words {} intents'.format(len(result['bounds']), len(result['words']), len(result['intents'])))
    # CRF suite expects a list of list of dicts of features: a list of features for each document
    # here we are considering a single document composed of all the sentences concatenated
    result['words'] = [result['words']]
    result['intents'] = [result['intents']]

    return result

def intents_collapse_from_word_level(intents, bounds, mode='all'):
    """Collapses the intent values putting together the output labels at word level,
    emitting a single value for each sentence according to its bounds.
    
    'mode' can be 'all' meaning that all the values must be unanimous from the words,
    or can be 'majority' to perform a majority vote """
    # flat the list of list wanted by CRF suite
    flattened_intents = [item for sublist in intents for item in sublist]
    # split the intents in groups corresponding to the sentences
    intent_groups = [flattened_intents[start:end] for start, end in bounds]
    # result is one intent for each group
    results = []
    for intent_group in intent_groups:
        if mode == 'all':
            # all should be the same
            if all(intent_group[0] == item for item in intent_group):
                # agreeing
                results.append(intent_group[0])
            else:
                # conflicting values
                results.append('<UNK>')
        elif mode == 'majority':
            # count the votes
            counter = collections.Counter(intent_group)
            # pick the winner
            winner, _ = counter.most_common(1)[0]
            results.append(winner)
        else:
            raise ValueError('unknown mode ' + mode)
    return results


def train_crf(inputs, outputs):
    """Train the CRF, inputs and outputs must be already preprocessed"""
    x_train = inputs
    y_train = outputs

    crf = sklearn_crfsuite.CRF(
        algorithm='ap',
        max_iterations=100,
        all_possible_transitions=True,
        verbose=True
    )

    crf.fit(x_train, y_train)
    print('trained')
    return crf

def test_crf(model, inputs):
    """Obtain the predicted values on the CRF, inputs must be preprocessed"""
    outputs = model.predict(inputs)
    #print(outputs)
    return outputs

def train_and_eval(features='words', test_type='test'):
    if features == 'word_vectors':
        nlp = spacy.load('en_vectors_web_lg')
    else:
        nlp = None

    train_raw, test_raw = load_data(test_type)
    train = prepare_data(train_raw, features, nlp)
    test = prepare_data(test_raw, features, nlp)

    x_train = train['words']
    y_train = train['intents']

    crf = train_crf(x_train, y_train)
    
    x_test = test['words']
    y_test = test['intents']
    bounds_test = test['bounds']
    y_test_collapsed = intents_collapse_from_word_level(y_test, bounds_test)

    y_predict = test_crf(crf, x_test)

    f1_word_level = f1_score([item for sublist in y_test for item in sublist], [item for sublist in y_predict for item in sublist], average='micro')
    print(f1_word_level)

    y_predict_collapsed_all = intents_collapse_from_word_level(y_predict, bounds_test)
    y_predict_collapsed_majority = intents_collapse_from_word_level(y_predict, bounds_test, 'majority')
    f1_intent_all = f1_score(y_test_collapsed, y_predict_collapsed_all, average='micro')
    f1_intent_majority = f1_score(y_test_collapsed, y_predict_collapsed_majority, average='micro')
    print(f1_intent_all, f1_intent_majority)

    with open('results/' + test_type + '_crf_' + features + '.json', 'w') as out_file:
        json.dump({'f1_intent_all_agree': f1_intent_all, 'f1_intent_majority': f1_intent_majority}, out_file)
    
    return f1_word_level, f1_intent_all, f1_intent_majority

def main(test_type='test'):
    d,e,f = train_and_eval('word_vectors', test_type)
    a,b,c = train_and_eval('words', test_type)
    print('Basic CRF results:')
    print('considering words, f1 on sequence', a, 'f1 on all_agree intent', b, 'f1 on majority_vote intent', c)
    print('considering word vectors, f1 on sequence', d, 'f1 on all_agree intent', e, 'f1 on majority_vote intent', f)

if __name__ == '__main__':
    plac.call(main)
    