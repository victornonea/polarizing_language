from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForTokenClassification, PreTrainedTokenizerFast
import pandas as pd
import evaluate
from datasets import Dataset
from datasets import ClassLabel
import regex as re
import os

import data_viewer
from util import tokenize_and_align_labels, compute_metrics

task = "ner"
ref_model_name = "distilbert-base-cased"
# batch_size = 16

classmap = ClassLabel(num_classes=2, names=['O', 'I'])

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return tokenizer

def load_and_preprocess_data(tokenizer):
    def load_in_trainable_form(dir_name):
        articles = data_viewer.load_article_set(dir_name)
        articles_trainable_form = [art.as_lightweight_trainable() for art in articles]
        return [sent for art in articles_trainable_form for sent in art]

    train_sentences = load_in_trainable_form('data/train')
    dev_sentences = load_in_trainable_form('data/dev')

    ds_train = Dataset.from_pandas(pd.DataFrame(data=train_sentences))
    ds_dev = Dataset.from_pandas(pd.DataFrame(data=dev_sentences))

    ds_train = ds_train.map(lambda y: {"labels": classmap.str2int(y["labels"])})
    ds_dev = ds_dev.map(lambda y: {"labels": classmap.str2int(y["labels"])})

    ds_train = ds_train.map(tokenize_and_align_labels(tokenizer), batched=True)
    ds_dev = ds_dev.map(tokenize_and_align_labels(tokenizer), batched=True)
    
    return ds_train, ds_dev

def load_model(option='new'):
    checkpoint_dir = 'tmp_trainer'
    if option == 'new':
        checkpoint_name = ref_model_name
    elif option == 'last':
        ids = set([match.group(1) for file_name in os.listdir(path=checkpoint_dir) if (match := re.search(r'checkpoint-(\d+)', file_name))])
        if ids:
            checkpoint_name = f'{checkpoint_dir}/checkpoint-{max(ids)}'
        else:
            checkpoint_name = ref_model_name
    else:
        checkpoint_name = f'{checkpoint_dir}/checkpoint-{option}'
    
    print(f'Loading model {checkpoint_name}')
    
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint_name,
        id2label={i:classmap.int2str(i) for i in range(classmap.num_classes)},
        label2id={c:classmap.str2int(c) for c in classmap.names},
        finetuning_task=task)
    return model



if __name__ == '__main__':
    tokenizer = load_tokenizer()
    data_collator = DataCollatorForTokenClassification(tokenizer)
    ds_train, ds_dev = load_and_preprocess_data(tokenizer)
    model = load_model('last')
    metric = evaluate.load("seqeval")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir='tmp_trainer', num_train_epochs=1),
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics(metric, classmap.names),
    )
    trainer.train()
    print(trainer.evaluate())
