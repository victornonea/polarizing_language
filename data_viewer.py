# Victor Nonea 2024

from dataclasses import dataclass
import os
import regex as re
import csv


@dataclass
class Label:
    def __init__(self, start, end, value):
        self.start = int(start)
        self.end = int(end)
        self.value = value
    
    def __iter__(self):
        return iter((self.start, self.end, self.value))

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
    
    def as_lightweight_trainable(self):
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
            res.append({'words': [], 'labels': []})
            for word, word_mask in zip(re.split(whitespaces, sentence), re.split(whitespaces, sent_mask)):
                res[-1]['words'].append(word)
                res[-1]['labels'].append('I' if 'I' in word_mask else 'O')
        
        return res


def load_article_ids(path='.'):
    return set([match.group(1) for file_name in os.listdir(path=path) if (match := re.search(r'article(\d+)\.txt', file_name))])


def load_article_set(path='.'):
    ids = load_article_ids(path=path)
    res = []
    for id in ids:
        with open(os.path.join(path, f'article{id}.txt'), 'r', encoding="utf-8") as f:
            text = f.read()
        
        labels = []
        with open(os.path.join(path, f'article{id}.labels.tsv'), 'r') as f:
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
