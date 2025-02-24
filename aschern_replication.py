from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForTokenClassification, PreTrainedTokenizerFast, AdamW
import pandas as pd
from datasets import Dataset
from datasets import ClassLabel
import regex as re
import os
import json
from matplotlib import pyplot as plt
import random as rn
import numpy as np

import data_viewer
from util import tokenize_and_align_labels, compute_metrics

# model_type: roberta
# config_name: roberta-large
# max_seq_length: 256 <- fixed pad size does not seem to serve any purpose and slows down computation, dynamic (default) padding per batch
# per_gpu_train_batch_size: 8 (V)
# per_gpu_eval_batch_size: 1
# learning_rate: 2e-5 (V)
# warmup_steps: 500 (V)
# num_train_epochs: 27 (V)
# do_lower_case: True (V)
# aschern does BIO tagging, we will not because it would slow down training ('B' tag would be sparse, rebalancing the train set would be difficult), complicate evaluation (F1 for multiclass requires design choices), and there is no reason to encourage the model to assign special meaning to the beggining of a polarized/propaganda pattern

task = "ner"
ref_model_name = "roberta-base"
checkpoint_dir = 'tmp_trainer'
learning_rate = 2e-5
warmup_steps = 500
batch_size = 8    # assume single gpu
num_train_epochs = 23
save_steps = 2000

trainer_params = {
    'output_dir': checkpoint_dir,
    'lr_scheduler_type': 'linear',
    # 'lr_scheduler_kwargs': {'num_warmup_steps': warmup_steps},
    'warmup_steps': warmup_steps,
    'per_device_train_batch_size': batch_size,
}

classmap = ClassLabel(num_classes=2, names=['O', 'I'])
target_label = 'I'

def load_tokenizer(**kwargs):
    if re.search('roberta', ref_model_name):
        kwargs = {'add_prefix_space': True, **kwargs}
    
    tokenizer = AutoTokenizer.from_pretrained(ref_model_name, **kwargs)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return tokenizer

def load_and_preprocess_data(tokenizer):
    global ds_train_art_ranges, ds_dev_art_ranges

    def load_in_trainable_form(dir_name):
        articles = data_viewer.load_article_set(dir_name)
        articles_trainable_form = [art.as_lightweight_trainable() for art in articles]
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

    ds_train = ds_train.map(lambda y: {"labels": classmap.str2int(y["labels"])})
    ds_dev = ds_dev.map(lambda y: {"labels": classmap.str2int(y["labels"])})

    ds_train = ds_train.map(tokenize_and_align_labels(tokenizer), batched=True)
    ds_dev = ds_dev.map(tokenize_and_align_labels(tokenizer), batched=True)
    
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
    
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint_name,
        id2label={i:classmap.int2str(i) for i in range(classmap.num_classes)},
        label2id={c:classmap.str2int(c) for c in classmap.names},
        finetuning_task=task)
    return model, checkpoint_name

def get_trainer(model, **kwargs):
    return Trainer(
        model=model,
        args=TrainingArguments(**trainer_params, **kwargs),
        # optimizer_cls_and_kwargs=(AdamW, {'lr': learning_rate}),
        optimizers=(AdamW(model.parameters(), lr=learning_rate), None),
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

def train():
    trainer.train(resume_from_checkpoint=(checkpoint_name if model_option != 'new' and checkpoint_name != ref_model_name else None))
    print(trainer.evaluate())

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

def plot_checkpoint_performance(train=True, dev=True, targets=('f1', 'precision', 'accuracy')):
    ids = list(get_checkpoint_ids())
    res = {}
    res_file_name = 'eval_result.json'
    if train:
        res.update({'train_' + target: {} for target in targets})
    if dev:
        res.update({'dev_' + target: {} for target in targets})
    
    for id in ids:
        checkpoint_name = get_checkpoint_name(id)
        if not res_file_name in os.listdir(path=checkpoint_name):
            continue
        with open(os.path.join(checkpoint_name, res_file_name), 'r') as file:
            checkpoint_res = json.load(file)
        if train:
            for target in targets:
                res['train_' + target][id] = checkpoint_res['train']['eval_' + target]
        if dev:
            for target in targets:
                res['dev_' + target][id] = checkpoint_res['dev']['eval_' + target]
    
    handles = []
    for i, target in enumerate(targets):
        if train:
            target_res = res['train_' + target]
            handles.append(plt.scatter(list(target_res.keys()), list(target_res.values()), color=f'C{i}', marker='x', label='train_' + target))
        if dev:
            target_res = res['dev_' + target]
            handles.append(plt.scatter(list(target_res.keys()), list(target_res.values()), color=f'C{i}', label='dev_' + target))
    plt.legend(handles=handles)
    plt.show()

def set_total_train_steps():
    t_total = len(ds_train) // batch_size * num_train_epochs # in the aschern implementation there is also a gradient_accumulation_steps variable but it does not seem to be relevant
    print('# Train steps:', t_total)
    # trainer_params['lr_scheduler_kwargs'] = {**trainer_params['lr_scheduler_kwargs'], 'num_training_steps': t_total} # not actually necessary, Trainer() should do the same calculation in the background

def load_objects():
    global tokenizer, data_collator, ds_train, ds_dev, model_option, model, checkpoint_name, trainer, metric, compute_metrics
    tokenizer = load_tokenizer()
    data_collator = DataCollatorForTokenClassification(tokenizer)
    ds_train, ds_dev = load_and_preprocess_data(tokenizer)
    set_total_train_steps()
    model_option = 'last'
    model, checkpoint_name = load_model(model_option)
    compute_metrics = compute_metrics(classmap.names, target_label)
    trainer = get_trainer(model, num_train_epochs=num_train_epochs, save_steps=save_steps)

def stats():
    def quick_stats(seq):
        seq = list(seq)
        return f'mean {np.mean(seq):.2f}, max {max(seq)}, std {np.std(seq):.2f}'

    def ds_stats(ds, art_ranges=None):
        I_word_count = sum(label == 1 for sent in ds for label in sent['labels'])
        O_word_count = sum(label == 0 for sent in ds for label in sent['labels'])
        word_count = I_word_count + O_word_count
        
        res = ''
        if art_ranges:
            res += f"{len(art_ranges)} articles, "
        
        res += f'{len(ds)} samples, {word_count} words, {100 * I_word_count / word_count:.2f}% inside words, {100 * O_word_count / word_count:.2f}% outside words'
        res += f' | token stats per example: {quick_stats(len(sent["tokens"]) for sent in ds)}'
        if art_ranges:
            res += f' | example per article stats: {quick_stats(r[1] - r[0] for r in art_ranges)}'
            res += f' | tokens per article stats: {quick_stats(sum(len(ds[i]["tokens"]) for i in range(*r)) for r in art_ranges)}'
        return res
    
    print('Train set stats:', ds_stats(ds_train, ds_train_art_ranges))
    print('Dev set stats:', ds_stats(ds_dev, ds_dev_art_ranges))

def predict_verbose(_set, index):
    sample = _set.select([index]).select_columns([c for c in _set.column_names if c != 'labels'])
    res = trainer.predict(sample)
    for t in zip(res.predictions[0], _set[index]['tokens'], _set[index]['labels']):
        print(t)

if __name__ == '__main__':
    load_objects()
    
    stats()

    train()
    eval_checkpoints()
    
    plot_checkpoint_performance(targets=('f1', 'precision', 'recall'))
