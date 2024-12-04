
# Credit: HuggingFace https://huggingface.co/docs/transformers/tasks/token_classification#preprocess
def tokenize_and_align_labels(tokenizer):
    def _tokenize_and_align_labels(examples):
        nonlocal tokenizer
        tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True)
        labels = []

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

        tokenized_inputs["labels"] = labels

        return tokenized_inputs
    return _tokenize_and_align_labels

# Credit: HuggingFace https://huggingface.co/docs/transformers/tasks/token_classification#evaluate
def compute_metrics(metric, label_list):
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
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return _compute_metrics
