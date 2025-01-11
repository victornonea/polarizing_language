
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

# Credit: HuggingFace https://huggingface.co/docs/transformers/tasks/token_classification#evaluate
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
