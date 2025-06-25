import evaluate
import numpy as np
from collections import deque
import random as rn

# Credit: HuggingFace https://huggingface.co/docs/transformers/tasks/token_classification#preprocess
def tokenize_and_align_labels(tokenizer):
    def _tokenize_and_align_labels(examples):
        nonlocal tokenizer
        tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True)
        labels = []
        tokens = []

        for i, label in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:  # Set the special tokens to -100.

                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)

                previous_word_idx = word_idx

            labels.append(label_ids)
            tokens.append(tokenized_inputs.tokens(batch_index=i))

        tokenized_inputs["labels"] = labels
        tokenized_inputs["tokens"] = tokens

        return tokenized_inputs
    return _tokenize_and_align_labels

def compute_metrics(label_list, target_label):
    def _compute_metrics(p):
        predictions, labels = p
        predictions = predictions.argmax(axis=2)
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        counts = {(l1, l2): 0 for l1 in label_list for l2 in label_list}
        for example in zip(true_predictions, true_labels):
            for p, l in zip(*example):
                counts[(p, l)] += 1
        
        precision = counts[(target_label, target_label)] / sum(counts[(p, l)] for p, l in counts.keys() if p == target_label)
        recall = counts[(target_label, target_label)] / sum(counts[(p, l)] for p, l in counts.keys() if l == target_label)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = sum(counts[(p, l)] for p, l in counts.keys() if p == l) / sum(counts.values())
        
        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }
        
        if len(label_list) == 2:
            other_label = next(l for l in label_list if l != target_label)
            MCC = (counts[target_label, target_label] * counts[other_label, other_label] - counts[target_label, other_label] * counts[other_label, target_label]) / \
                (sum(counts[target_label, _] for _ in label_list) * sum(counts[other_label, _] for _ in label_list) * sum(counts[_, other_label] for _ in label_list) * sum(counts[_, target_label] for _ in label_list)) ** (1/2)
            results['MCC'] = MCC
        
        return results
    return _compute_metrics

multi_label_metrics = evaluate.combine(["f1", "precision", "recall"])

def sigmoid(x):
   return 1/(1 + np.exp(-x))

def compute_metrics_multi_label(p):
    predictions, labels = p
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int)
    labels = labels.astype(int)
    
    predictions_any = predictions[:, 0]
    labels_any = labels[:, 0]
    
    predictions_specific = predictions[:, 1:].reshape((-1,))
    labels_specific = labels[:, 1:].reshape((-1,))
    
    def append_keys(prefix, _dict):
        return {prefix + key: value for key, value in _dict.items()}
    
    return {**append_keys('any_', multi_label_metrics.compute(predictions_any, labels_any)),
        **append_keys('specific_', multi_label_metrics.compute(predictions_specific, labels_specific))}

def list_split(l, delim_vals):
    res = []
    new = True
    for e in l:
        if isinstance(e, str) and e in delim_vals:
            new = True
            continue
        if new:
            res.append([])
            new = False
        res[-1].append(e)
    return res

class WindowAverage:
    def __init__(self, cap):
        self.cap = cap
        self.sum = 0
        self.data = deque()
    
    def update(self, n):
        self.sum += n
        self.data.append(n)
        if len(self.data) > self.cap:
            self.sum -= self.data.popleft()
    
    @property
    def val(self):
        if not self.data:
            return float('nan')
        return self.sum / len(self.data)

def create_regex_schema_from_keywords(kw_dict):
    schema_dict = {}
    for topic in kw_dict:
        schema_dict[topic] = '|'.join(r'(\b' + keyword + r'\b)' for keyword in kw_dict[topic])
    return schema_dict

class QuickPopList(list):
    def pop(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        
        target_value = self[index]
        
        last_value = self[-1]
        self[index] = last_value
        super().pop()
        
        return target_value

def multi_label_set_balanced_resample(_set, seed=0):
    # we are interested in creating a set with equal amounts of positive samples for each class
    fixed_rn = rn.Random(seed)
    ids_to_labels = {id: _set[id]['labels'] for id in range(len(_set))}
    num_classes = len(_set[0]['labels'])    # any label
    reps = [QuickPopList() for _ in range(num_classes)]
    for sent_id, labels in ids_to_labels.items():
        for i, l in enumerate(labels):
            if l:
                reps[i].append(sent_id)
    
    target_sample_size_per_class = min(len(l) for l in reps)
    counts_to_go = [target_sample_size_per_class for _ in range(num_classes)]
    res_ids = set()
    while any(count > 0 for count in counts_to_go):
        i = 0
        while counts_to_go[i] <= 0:
            i += 1
        
        pick_id = reps[i].pop(fixed_rn.randrange(len(reps[i])))
        while pick_id in res_ids:
            pick_id = reps[i].pop(fixed_rn.randrange(len(reps[i])))
        
        res_ids.add(pick_id)
        for i, l in enumerate(ids_to_labels[pick_id]):
            if l:
                counts_to_go[i] -= 1
    
    return [_set[id] for id in res_ids]

def topics_as_latex_table(mapping):
    rows = [['' for _ in range(len(mapping))] for _ in range(max(len(v) for v in mapping.values()) + 1)]
    rows[0] = list(mapping.keys())
    for i, key in enumerate(mapping.keys()):
        for j, value in enumerate(mapping[key]):
            rows[j + 1][i] = value
    
    lines = []
    for row in rows:
        lines.append('\t' + ' & '.join(row) + r' \\');
    return '\n'.join(lines)

def k_fold(data, fold_idx=0, k=5):
    fold_size = len(data) // k
    bounds = [i * fold_size for i in range(k)] + [len(data)]
    folds = [data[bounds[i]:bounds[i+1]] for i in range(k)]
    return sum([folds[i] for i in range(k) if i != fold_idx], []), folds[fold_idx]
