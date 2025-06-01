# Victor Nonea 2024

from datasets import ClassLabel
from dataclasses import dataclass
import os
import regex as re
import csv
from collections import Counter
import random as rn
import numpy as np

from util import list_split

patterns = ['Appeal_to_Authority',
    'Appeal_to_fear-prejudice',
    'Bandwagon',
    'Black-and-White_Fallacy',
    'Causal_Oversimplification',
    'Doubt',
    'Exaggeration,Minimisation',
    'Flag-Waving',
    'Loaded_Language',
    'Name_Calling,Labeling',
    'Obfuscation,Intentional_Vagueness,Confusion',
    'Red_Herring',
    'Reductio_ad_hitlerum',
    'Repetition',
    'Slogans',
    'Straw_Men',
    'Thought-terminating_Cliches',
    'Whataboutism']

pattern_classmap = ClassLabel(num_classes=len(patterns) + 1, names=['Any_Pattern'] + patterns)

keyword_classmap = ClassLabel(num_classes=5, names=['None', 'Violence', 'Commotion', 'Disparage', 'Legal'])

@dataclass
class Label:
    def __init__(self, start, end, value):
        self.start = int(start)
        self.end = int(end)
        self.value = value
    
    def __iter__(self):
        return iter((self.start, self.end, self.value))
    
    def __repr__(self): 
        return str((self.start, self.end, self.value))

@dataclass
class Article:
    id: str
    text: str
    labels: list[Label]

    def __repr__(self):
        labels = list(self.labels)
        labels.sort(key=lambda label: label.start)
        curr = 0
        accum = []
        for label_start, label_end, label_value in labels:
            if curr > label_end:
                continue
            
            if curr > label_start:
                label_start = curr
            
            accum.append(self.text[curr:label_start])
            accum.append('**')
            accum.append(self.text[label_start:label_end])
            accum.append('**<-[')
            accum.append(label_value)
            accum.append('] ')
            
            curr = label_end
        
        accum.append(self.text[curr:-1])
        return ''.join(accum)

    def get_all_line_ranges(self, start=0, end=None):
        if end is None:
            end = len(self.text)
        last = start
        res = []
        for i in range(start, end):
            if self.text[i] == '\n':
                if re.search(r'\w', self.text[last:i]):
                    res.append((last, i))
                last = i + 1
        if i > last:
            res.append((last, i))
        return res

    def get_line(self, label):
        start, end, _ = label
        while start > 0 and self.text[start - 1] != '\n':
            start -= 1
        while end < len(self.text) and self.text[end] != '\n':
            end += 1
        
        return start, end

    def get_line_highlight(self, label, include_art_id=False):
        label_start, label_end, label_value = label
        
        start, end = self.get_line(label)
        
        res = self.text[start:label_start] + '**' + \
            self.text[label_start:label_end] + '**<-[' + \
            label_value + ']' + self.text[label_end:end]
        
        if include_art_id:
            res = f'Art. {self.id}: ' + res
        
        return res
    
    def split_label_across_lines(self, label):
        ranges = self.get_all_line_ranges(label.start, label.end)
        for s, e in ranges:
            yield Label(s, e, label.value)
    
    def as_lightweight_trainable(self):
        # construct a trainable with sentence-wise samples, word-wise tokens and token-wise labels
        labels = list(self.labels)
        labels.sort(key=lambda label: label.start)
        curr = 0
        whitespaces = r'[ \t\n\r]'
        newlines = r'[\n\r]'
        label_mask = ['O' if not re.match(whitespaces, c) else c for c in self.text]
        for label_start, label_end, label_value in labels:
            label_mask[label_start:label_end] = ['I' if not re.match(whitespaces, c) else c for c in label_mask[label_start:label_end]]
        label_mask = ''.join(label_mask)
        
        res = []
        for sentence, sent_mask in zip(re.split(newlines, self.text), re.split(newlines, label_mask)):
            if not re.split(whitespaces, sentence):
                continue
            
            res.append({'words': [], 'labels': []})
            for word, word_mask in zip(re.split(whitespaces, sentence), re.split(whitespaces, sent_mask)):
                res[-1]['words'].append(word)
                res[-1]['labels'].append('I' if 'I' in word_mask else 'O')
        
        return res
    
    def as_multi_label_trainable(self):
        # construct a trainable with sentence-wise samples, sample-wise labels and multi-label format
        labels = list(self.labels)
        labels.sort(key=lambda label: label.start)
        curr = 0
        whitespaces = r'[ \t\n\r]'
        newlines = r'[\n\r]'
        newlines_raw = '\n\r'
        label_mask = [0 if not re.match(whitespaces, c) else c for c in self.text]
        for label_start, label_end, label_value in labels:
            label_mask[label_start:label_end] = [pattern_classmap.str2int(label_value) if not isinstance(c, str) else c for c in label_mask[label_start:label_end]]
        
        res = []
        for sentence, sent_mask in zip(re.split(newlines, self.text), list_split(label_mask, newlines_raw)):
            if not sentence:
                continue
            
            res.append({'text': sentence, 'labels': [0. for _ in range(pattern_classmap.num_classes)]})
            
            has_patterns = False
            for i in sent_mask:
                if not isinstance(i, str) and i:
                    has_patterns = True
                    res[-1]['labels'][i] = 1.
            
            res[-1]['labels'][0] = 1. if has_patterns else 0.
        
        return res
    
    def as_multi_label_trainable_from_regex_schema(self, schema):
        newlines = r'[\n\r]'
        res = []
        for sentence in re.split(newlines, self.text):
            if not sentence:
                continue
            
            res.append({'text': sentence, 'labels': [1.] + [0. for _ in schema]})
            
            for i, topic in enumerate(schema):
                if re.search(schema[topic], sentence):
                    res[-1]['labels'][i + 1] = 1.
                    res[-1]['labels'][0] = 0.
        
        return res


class ArticleList(list):
    def __init__(self):
        self.label_map = {}
        super().__init__()

    def append(self, art):
        for label in art.labels:
            if label.value not in self.label_map:
                self.label_map[label.value] = []
            self.label_map[label.value].append((art, label))
        super().append(art)
    
    @property
    def label_categories(self):
        return list(self.label_map.keys())
    
    def get_label_examples(self, category, n=10):
        if n < 1:
            sample = self.label_map[category]
        else:
            n = min(n, len(self.label_map[category]))
            sample = rn.sample(self.label_map[category], n)
        for art, label in sample:
            print(art.get_line_highlight(label, include_art_id=True), '\n')


def load_article_ids(path='.'):
    return sorted([match.group(1) for file_name in os.listdir(path=path) if (match := re.search(r'article(\d+)\.txt', file_name))])


def load_article_set(path='.', labels_path=None):
    if labels_path is None:
        labels_path = path

    ids = load_article_ids(path=path)
    res = ArticleList()
    for id in ids:
        with open(os.path.join(path, f'article{id}.txt'), 'r', encoding="utf-8") as f:
            text = f.read()
        
        labels = []
        labels_file_name = f'article{id}.labels.tsv'
        if labels_file_name not in os.listdir(labels_path):
            continue
        with open(os.path.join(labels_path, labels_file_name), 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                labels.append(Label(row[2], row[3], row[1]))
        
        res.append(Article(id, text, labels))
        
    return res

def create_master_labels_tsv(file_name, _set):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for article in _set:
            for label in article.labels:
                writer.writerow([article.id, label.start, label.end])

def create_master_articles_csv(file_name, _set):
    with open(file_name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['id', 'text'])
        for article in _set:
            writer.writerow([article.id, article.text])
