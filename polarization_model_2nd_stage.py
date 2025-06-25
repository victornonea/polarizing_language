import csv
import spacy
import numpy as np

nlp = spacy.load('en_core_web_sm')

from polarization_model_1st_stage import load_objects, sentence_predict
from data_viewer import load_article_set
from demo_polarization_first_stage import example1, example2, example3


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

tr = load_article_set('data/train', '../doccano/labels')
tr = {art.id: art for art in tr}

article_wise_answers = read_csv('article_answers-v1.csv')
answers = {}

for answer in article_wise_answers.rows():
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
        print(f'Warning, question {key[1]} for article {key[0]} missing expected answer.')

def sent_split(s):
    return [sent.text for sent in nlp(s).sents]

load_objects()

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
    
    return {q_id: np.corrcoef(np.array([q_traces_pred[q_id], q_traces_label[q_id]]))[0, 1] for q_id in considered_questions}, q_label_counts

res, counts = evaluate_train_set()

# print(predict(example1))
print(res)
print()
print(counts)

# {'Q2': 0.46321652660216023, 'Q15': -0.03334622815069757, 'Q10': 0.2033902365363096, 'Q14': 0.514186668677716, 'Q1': 0.4843448546445513, 'Q4': 0.18538972357395567, 'Q3': -0.18290211761180264, 'Q7': 0.44496790621705273}

# {'Q2': {1: 14, 2: 3, 3: 8, 4: 8, 5: 14}, 'Q15': {1: 41, 2: 8, 3: 7, 4: 2, 5: 1}, 'Q10': {1: 42, 2: 5, 3: 3, 4: 4, 5: 5}, 'Q14': {1: 17, 2: 5, 3: 10, 4: 5, 5: 22}, 'Q1': {1: 6, 2: 6, 3: 13, 4: 16, 5: 18}, 'Q4': {1: 27, 2: 10, 3: 6, 4: 6, 5: 10}, 'Q3': {1: 34, 2: 7, 3: 2, 4: 1, 5: 3}, 'Q7': {1: 23, 2: 13, 3: 7, 4: 12, 5: 4}}
