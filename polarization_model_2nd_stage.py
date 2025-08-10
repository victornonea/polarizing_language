import csv
import spacy
import numpy as np
import json

nlp = spacy.load('en_core_web_sm')

from polarization_model_1st_stage import load_objects, sentence_predict
from data_viewer import load_article_set
from demo_polarization_examples import example1, example2, example3
import util as ut


agg_patterns = {
    'Heavy_Language': 0,
    'Loaded_Language': 1,
    'Emotional_Language': 1,
    'Amplifier/Minimizer': 2,
    'Hyperbole/Oversimplification': 3,
    'Provocative_Unsubstatiated_Claim': 3,
    'Inappropriately_Informal_Tone/Irony': 3,
}

considered_questions = {'Q1', 'Q2', 'Q3', 'Q4', 'Q7', 'Q10', 'Q14', 'Q15'}
dependent_questions = {'Q2': 'Q1', 'Q3': 'Q1'}

low_signal_threshold = 0.15
max_confidence = 0.6
cutoff_length = -1

class Arg:
    def eval(self, realization):
        return NotImplementedError()
    
    def __invert__(self):
        return NotExp(self)
    
    def __and__(self, other):
        return AndExp(self, other)
    
    def __or__(self, other):
        return OrExp(self, other)

class Sym(Arg):
    def __init__(self, symbol):
        self.symbol = symbol
    
    def eval(self, realization):
        return realization[self.symbol]

class ArgVal(Arg):
    def __init__(self, value):
        self.value = value
    
    def eval(self, realization):
        return self.value

class NotExp(Arg):
    def __init__(self, arg):
        self.arg = arg
    
    def eval(self, realization):
        return 1 - self.arg.eval(realization)

class AndExp(Arg):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2
    
    def eval(self, realization):
        return self.arg1.eval(realization) * self.arg2.eval(realization)

class OrExp(Arg):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2
    
    def eval(self, realization):
        return 1 - (1 - self.arg1.eval(realization)) * (1 - self.arg2.eval(realization))

intentional_polarization_exp = Sym('Loaded_Language') | Sym('Emotional_Language') | Sym('Provocative_Unsubstatiated_Claim') | Sym('Hyperbole/Oversimplification') | Sym('Inappropriately_Informal_Tone/Irony')

rule_map = {
    'Q1': Sym('Heavy_Language') | Sym('Loaded_Language') | Sym('Emotional_Language'),
    'Q2': Sym('Loaded_Language') | Sym('Emotional_Language'),
    'Q3': Sym('Heavy_Language') & ~(Sym('Loaded_Language') | Sym('Emotional_Language')),
    'Q4': Sym('Provocative_Unsubstatiated_Claim'),
    'Q7': Sym('Hyperbole/Oversimplification'),
    'Q10': Sym('Inappropriately_Informal_Tone/Irony'),
    'Q14': intentional_polarization_exp,
    'Q15': (Sym('Heavy_Language') | Sym('Amplifier/Minimizer')) & ~intentional_polarization_exp,
}


class Table(dict):
    def __len__(self):
        return len(next(value for value in self.values()))
        
    def __bool__(self):
        return len(self) != 0
    
    def rows(self):
        for i in range(len(self)):
            yield {key: self[key][i] for key in self}
    
    def __getitem__(self, key):
        if not isinstance(key, int):
            return super().__getitem__(key)
        return self.get_by_index(key)
    
    def get_by_index(self, i):
        return {key: self[key][i] for key in self}

def read_csv(file_name):
    with open(file_name, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        res = Table({key: [] for key in header})
        for row in csv_reader:
            for i, e in enumerate(row):
                res[header[i]].append(e)
    return res

def sent_split(s):
    return [sent.text for sent in nlp(s).sents]

def signal_to_prob(x):
    return 0. if x < low_signal_threshold else max_confidence * (x - low_signal_threshold) / (1 - low_signal_threshold)

def any_over_probs(ps):
    ps = np.array(ps)
    neg_ps = 1 - ps
    neg_p = np.prod(neg_ps)
    p = 1 - neg_p
    return p

def reverse_any(p, n):
    # calculate the probability of n equal probability independent events, if their agg. any probability is p
    none_prob = 1 - p
    single_neg_prob = none_prob ** (1 / n)
    return 1 - single_neg_prob

def predict(text):
    indv_ps = {key: [] for key in agg_patterns.values()}
    for sent in sent_split(text):
        res = sentence_predict(sent)
        for key, val in res.items():
            indv_ps[key].append(signal_to_prob(val))
    
    agg_probs = {key: any_over_probs(sorted(ps, reverse=True)[:cutoff_length]) for key, ps in indv_ps.items()}
    
    agg_count = {agg_id: len([key for key in agg_patterns if agg_patterns[key] == agg_id]) for agg_id in set(agg_patterns.values())}
    
    realization = {key: reverse_any(agg_probs[agg_id], agg_count[agg_id]) for key, agg_id in agg_patterns.items()}
    
    return {q: rule_map[q].eval(realization) for q in rule_map}

def evaluate_train_set():
    tr = load_article_set('data/train', 'data/train_polarization_token_labels')
    tr = {art.id: art for art in tr}
    
    article_wise_answers = read_csv('data/polarization_article_labels/FormAnswers.csv')

    answers = {}

    for answer in article_wise_answers.rows():
        if answer['article_id'] not in tr:
            continue
        if answer['article_id'] in tr and answer['question_id'] in considered_questions:
            key = answer['article_id'], answer['question_id']
            if key in answers:
                print(f'Warning, question {key[1]} for article {key[0]} answered multiple times.')
            answers[key] = int(answer['answer'])

    for art_id in tr:
        for q_id in considered_questions:
            if (art_id, q_id) in answers:
                continue
            if q_id in dependent_questions and ((art_id, dependent_questions[q_id]) not in answers or answers[art_id, dependent_questions[q_id]] < 3):
                continue
            print(f'Warning, question {q_id} for article {art_id} missing expected answer.')


    q_traces_pred = {q_id: [] for q_id in considered_questions}
    q_traces_label = {q_id: [] for q_id in considered_questions}
    
    q_label_counts = {q_id: {i: 0 for i in range(1, 6)} for q_id in considered_questions}
    
    for i, art_id in enumerate(tr):
        print(i)
        art = tr[art_id]
        pred = predict(art.text)
        for q_id in considered_questions:
            if (art_id, q_id) in answers:
                q_traces_pred[q_id].append(pred[q_id])
                q_traces_label[q_id].append(answers[art_id, q_id])
                
                q_label_counts[q_id][answers[art_id, q_id]] += 1
    
    coef_payload = lambda q_id: np.array([q_traces_pred[q_id], q_traces_label[q_id]]).transpose()
    corrcoef_wrapper = lambda ab: ut.corrcoef(ab[:, 0], ab[:, 1])
    
    return {q_id: (ut.corrcoef(q_traces_pred[q_id], q_traces_label[q_id]), len(q_traces_pred[q_id]), ut.bootstrap_confidence(corrcoef_wrapper, coef_payload(q_id))) for q_id in considered_questions}, q_label_counts

def evaluate_test_set():
    text_set = load_article_set('data/dev') + load_article_set('data/test')
    text_set = {art.id: art for art in text_set}
    
    row_answers = read_csv('data/polarization_article_labels/Eval-FormAnswers.csv')
    
    eval_arts = set(row_answers['article_id'])
    
    answer_map = {}
    for answer in row_answers.rows():
        key = answer['article_id'], answer['question_id']
        if key not in answer_map:
            answer_map[key] = []
        answer_map[key].append(int(answer['answer']))


    for key in answer_map:
        answer_map[key] = np.mean(answer_map[key])

    q_traces_pred = {q_id: [] for q_id in considered_questions}
    q_traces_label = {q_id: [] for q_id in considered_questions}
    
    for i, art_id in enumerate(eval_arts):
        print(i)
        art = text_set[art_id]
        pred = predict(art.text)
        for q_id in considered_questions:
            if (art_id, q_id) in answer_map:
                q_traces_pred[q_id].append(pred[q_id])
                q_traces_label[q_id].append(answer_map[art_id, q_id])
    
    coef_payload = lambda q_id: np.array([q_traces_pred[q_id], q_traces_label[q_id]]).transpose()
    corrcoef_wrapper = lambda ab: ut.corrcoef(ab[:, 0], ab[:, 1])
    
    return {q_id: (ut.corrcoef(q_traces_pred[q_id], q_traces_label[q_id]), len(q_traces_pred[q_id]), ut.bootstrap_confidence(corrcoef_wrapper, coef_payload(q_id))) for q_id in considered_questions}


if __name__ == '__main__':
    load_objects(exclude_data=True)

    print(predict(example1))
