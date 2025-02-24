from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, PreTrainedTokenizerFast, AdamW
import torch
import pandas as pd
from datasets import Dataset
import regex as re
import os
import json
from matplotlib import pyplot as plt
import random as rn
import numpy as np

import data_viewer
from util import compute_metrics_multi_label

# copy the aschern replication setup except:
# place a custom classification-like head - similar to classification but without softmax
    # I may not need a custom head, the default loss function LabelSmoother applies log softmax itself, implying the head stops at the logistic activations
# override the compute_loss_func parameter to compute a loss that is just absolute difference or squared diff or whatever
# use LabelSmoother https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L540 as reference
# first override compute_loss_func as an introspective hook that reveals the key parameters and breaks execution
# then implement the actual function
# create labels per sentence like:
#   - No Pattern: [0, 1]
#   - Pattern A: [0, 1]
#   - Pattern B: [0, 1]
#   ...
# "No Pattern" is from a learning perspective equivalent to an "Any pattern" logit (both would be redundant) but "No Pattern" assures the norm is always non-zero
# use smt like cosine similarity as eval metric
# distrib of labels:
# specific pattern [34540 - no,   812 - yes] 80 : 1
# any pattern [1331 - no,  633 - yes], 2 : 1
#
# https://huggingface.co/docs/transformers/tasks/sequence_classification
#   looks like I may have to overwrite the data collator, it seems to be resonasible for converting human-readable labels to numeric labels
# https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification
#
# https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/roberta/modeling_roberta.py#L1356
# BCEWithLogitsLoss

ref_model_name = "roberta-base"
checkpoint_dir = 'proto_trainer'
learning_rate = 1e-6   # may need to introduce custom BCEWithLogitsLoss with positive weights
warmup_steps = 0
batch_size = 8    # assume single gpu
num_train_epochs = 23
save_steps = 1000

classmap = data_viewer.pattern_classmap

trainer_params = {
    'output_dir': checkpoint_dir,
    'lr_scheduler_type': 'linear',
    # 'lr_scheduler_kwargs': {'num_warmup_steps': warmup_steps},
    'warmup_steps': warmup_steps,
    'per_device_train_batch_size': batch_size,
    'eval_steps': 500,
    'eval_strategy': 'steps'
}

def load_tokenizer(**kwargs):
    if re.search('roberta', ref_model_name):
        kwargs = {'add_prefix_space': True, **kwargs}
    
    tokenizer = AutoTokenizer.from_pretrained(ref_model_name, cleaup_tokenization_spaces=False, **kwargs)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return tokenizer

def tokenize(example):
    res = tokenizer(example['text'], truncation=True)
    res['labels'] = example['labels']
    return res

def load_and_preprocess_data(tokenizer):
    global ds_train_art_ranges, ds_dev_art_ranges

    def load_in_trainable_form(dir_name):
        articles = data_viewer.load_article_set(dir_name)
        articles_trainable_form = [art.as_multi_label_trainable() for art in articles]
        sents = []
        sent_len = 0
        art_ranges = []
        for art in articles_trainable_form:
            sents.extend(art)
            art_ranges.append((sent_len, sent_len + len(art)))
            sent_len = len(sents)
        return sents, art_ranges

    train_sentences, ds_train_art_ranges = load_in_trainable_form('data/train')
    dev_sentences, ds_dev_art_ranges = load_in_trainable_form('data/dev')

    ds_train = Dataset.from_pandas(pd.DataFrame(data=train_sentences))
    ds_dev = Dataset.from_pandas(pd.DataFrame(data=dev_sentences))

    ds_train = ds_train.map(tokenize)
    ds_dev = ds_dev.map(tokenize)
    
    return ds_train, ds_dev

def get_checkpoint_ids():
    return set([int(match.group(1)) for file_name in os.listdir(path=checkpoint_dir) if (match := re.search(r'checkpoint-(\d+)', file_name))])

def get_checkpoint_name(id):
    return f'{checkpoint_dir}/checkpoint-{id}'

def load_model(option='new'):
    if option == 'new':
        checkpoint_name = ref_model_name
    elif option == 'last':
        ids = get_checkpoint_ids()
        if ids:
            checkpoint_name = get_checkpoint_name(max(ids))
        else:
            checkpoint_name = ref_model_name
    else:
        checkpoint_name = get_checkpoint_name(option)
    
    print(f'Loading model {checkpoint_name}')
    
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_name,
        id2label={i:classmap.int2str(i) for i in range(classmap.num_classes)},
        label2id={c:classmap.str2int(c) for c in classmap.names},
        problem_type='multi_label_classification')
    return model, checkpoint_name

def get_trainer(model, **kwargs):
    return Trainer(
        model=model,
        args=TrainingArguments(**trainer_params, **kwargs),
        optimizers=(AdamW(model.parameters(), lr=learning_rate), None),
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_multi_label,
    )

def train():
    trainer.train(resume_from_checkpoint=(checkpoint_name if model_option != 'new' and checkpoint_name != ref_model_name else None))

def eval_checkpoints():
    res_file_name = 'eval_result.json'
    ids = list(get_checkpoint_ids())
    rn.shuffle(ids)
    for id in ids:
        checkpoint_name = get_checkpoint_name(id)
        print(checkpoint_name)
        if res_file_name in os.listdir(path=checkpoint_name):
            with open(os.path.join(checkpoint_name, res_file_name), 'r') as file:
                checkpoint_res = json.load(file)
        else:
            model, _ = load_model(id)
            trainer = get_trainer(model)
            checkpoint_res = {'train': trainer.evaluate(ds_train), 'dev': trainer.evaluate(ds_dev)}
            print(checkpoint_res)
            with open(os.path.join(checkpoint_name, res_file_name), 'w') as file:
                json.dump(checkpoint_res, file)

def set_total_train_steps():
    t_total = len(ds_train) // batch_size * num_train_epochs # in the aschern implementation there is also a gradient_accumulation_steps variable but it does not seem to be relevant
    print('# Train steps:', t_total)
    # trainer_params['lr_scheduler_kwargs'] = {**trainer_params['lr_scheduler_kwargs'], 'num_training_steps': t_total} # not actually necessary, Trainer() should do the same calculation in the background

def load_objects():
    global tokenizer, ds_train, ds_dev, data_collator, model_option, model, checkpoint_name, trainer, metric, compute_metrics
    tokenizer = load_tokenizer()
    ds_train, ds_dev = load_and_preprocess_data(tokenizer)
    set_total_train_steps()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model_option = 'last'
    model, checkpoint_name = load_model(model_option)
    trainer = get_trainer(model, num_train_epochs=num_train_epochs, save_steps=save_steps)

def predict_verbose(_set, index):
    sample = _set.select([index]).select_columns([c for c in _set.column_names if c != 'labels'])
    res = trainer.predict(sample)
    for t in zip(res.predictions[0], _set[index]['labels']):
        print(t)

from util import util_debug as ud

if __name__ == '__main__':
    load_objects()

    print(trainer.evaluate(ds_dev))
    train()
    print(trainer.evaluate(ds_train))
    eval_checkpoints()
    
    plot_checkpoint_performance(targets=('any_f1', 'any_precision', 'any_recall', 'specific_f1', 'specific_precision', 'specific_recall'))
