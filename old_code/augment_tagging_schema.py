import re
import json
import numpy as np

from data_viewer import load_article_set

# For the 'model' object, check source file 'gensim_fun.py'
# Due to the relatively large load time of the model, the preferred development method was to run the 'gensim_fun.py' source file in interactive mode and reload this file whenever definitions needed to be changed

train = load_article_set('data/train')

with open('ref_keyword_tagging_schema.json', 'r') as file:
    ref_set = json.load(file)

def foo(model):
    avg_dist = {}
    for topic, ref_keys in ref_set.items():
        dist = []
        for i in range(len(ref_keys) - 1):
            for j in range(i + 1, len(ref_keys)):
                key1, key2 = ref_keys[i], ref_keys[j]
                dist.append(model.similarity(key1, key2))
        dist = np.mean(dist)
        avg_dist[topic] = dist
    
    print('avg', avg_dist)
    
    whitespaces = r'[ \t\n\r]'
    new_set = {key: set(value) for key, value in ref_set.items()}
    
    def similarity(a, b):
        nonlocal model
        try:
            return model.similarity(a, b)
        except KeyError:
            return 0
    
    visited = set()
    
    for _, article in enumerate(train):
        print(_)
        for word in re.split(whitespaces, article.text):
            word = word.lower()
            while word and re.match(r'\W', word[0]):
                word = word[1:]
            while word and re.match(r'\W', word[-1]):
                word = word[:-1]
            
            if not word or word in visited:
                continue
            
            visited.add(word)
            
            for topic, ref_keys in ref_set.items():
                if word in new_set[topic]:
                    continue
                if np.mean([similarity(ref_key, word) for ref_key in ref_keys]) >= avg_dist[topic]:
                    new_set[topic].add(word)
    
    new_set = {topic: list(keys) for topic, keys in new_set.items()}
    
    print(new_set)
    return new_set

def dump(new_set):
    with open('aug_keyword_tagging_schema.json', 'w') as file:
        json.dump(new_set, file, indent=4)
