import evaluate
import numpy as np
from collections import deque

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
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }
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
