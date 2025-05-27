from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, PreTrainedTokenizerFast, AdamW, get_linear_schedule_with_warmup
import torch
import pandas as pd
from datasets import Dataset
import regex as re
import os
import json
from matplotlib import pyplot as plt
import random as rn
import numpy as np
from tqdm import tqdm, trange

import data_viewer
from util import compute_metrics_multi_label, WindowAverage, multi_label_set_balanced_resample, create_regex_schema_from_keywords, sigmoid

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
checkpoint_dir = 'proto_trainer_keyword-good'
learning_rate = 3e-6    # 1e-6 appears to work for specific pattern, 1e-8 appears to work for "any" pattern
warmup_steps = 1000
batch_size = 4    # assume single gpu
num_train_epochs = 23
save_epochs = 4
eval_epochs = 4
do_rebalance_sets = True

device = 'cuda'

# ALL rebalancing and special evaluation mechanisms for the original propaganda set
# any_pattern_mask = torch.tensor([1.] + [0.] * 18, device=device)
# spec_pattern_mask = torch.tensor([0.] + [1.] * 18, device=device)
any_pattern_mask = torch.tensor([1.] + [0.] * 4, device=device)
spec_pattern_mask = torch.tensor([0.] + [1.] * 4, device=device)

# intended_label_weights = any_pattern_mask + (1 / 18) * spec_pattern_mask
# learning_rate_ratios = 100 * (any_pattern_mask + 10 * spec_pattern_mask)
# pos_label_weights = 2 * any_pattern_mask + 80 * spec_pattern_mask
# net_label_weights = intended_label_weights * learning_rate_ratios / (pos_label_weights + torch.ones(19, device=device))

#loss_fn = torch.nn.BCEWithLogitsLoss(
#    weight=net_label_weights,
#    pos_weight=pos_label_weights
#)
# any_loss_fn = torch.nn.BCEWithLogitsLoss(
#    weight=net_label_weights * any_pattern_mask,
#     pos_weight=pos_label_weights
#)
#spec_loss_fn = torch.nn.BCEWithLogitsLoss(
#    weight=net_label_weights * spec_pattern_mask,
#    pos_weight=pos_label_weights
#)

# classmap = data_viewer.pattern_classmap
classmap = data_viewer.keyword_classmap
pos_weight = torch.tensor(float(classmap.num_classes), device=device)

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
any_loss_fn = torch.nn.BCEWithLogitsLoss(weight=any_pattern_mask, pos_weight=pos_weight)
spec_loss_fn = torch.nn.BCEWithLogitsLoss(weight=spec_pattern_mask, pos_weight=pos_weight)
global_step = None


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
    
    #trainable_getter = lambda art: art.as_multi_label_trainable()
    with open('aug_keyword_tagging_schema.json', 'r') as file:
        kw_dict = json.load(file)
    schema = create_regex_schema_from_keywords(kw_dict)
    trainable_getter = lambda art: art.as_multi_label_trainable_from_regex_schema(schema)

    def load_in_trainable_form(dir_name):
        articles = data_viewer.load_article_set(dir_name)
        return sum([trainable_getter(art) for art in articles], [])

    train_sentences = load_in_trainable_form('data/train')
    dev_sentences = load_in_trainable_form('data/dev')
    
    if do_rebalance_sets:
        train_sentences = multi_label_set_balanced_resample(train_sentences)
        dev_sentences = multi_label_set_balanced_resample(dev_sentences)

    ds_train = Dataset.from_pandas(pd.DataFrame(data=train_sentences))
    ds_dev = Dataset.from_pandas(pd.DataFrame(data=dev_sentences))

    ds_train = ds_train.map(tokenize)
    ds_dev = ds_dev.map(tokenize)
    
    # DEBUG, pref remove
    ds_train = ds_train.remove_columns(['text'])
    ds_dev = ds_dev.remove_columns(['text'])
    
    return ds_train, ds_dev

def get_checkpoint_ids():
    return set([int(match.group(1)) for file_name in os.listdir(path=checkpoint_dir) if (match := re.search(r'checkpoint-(\d+)', file_name))])

def get_checkpoint_name(id):
    return f'{checkpoint_dir}/checkpoint-{id}'

found_saved_model = False

def load_model(option='last'):
    global global_step, found_saved_model
    
    global_step = 0
    if option == 'new':
        checkpoint_name = ref_model_name
    elif option == 'last':
        ids = get_checkpoint_ids()
        if ids:
            checkpoint_name = get_checkpoint_name(max(ids))
            global_step = max(ids)
            found_saved_model = True
        else:
            checkpoint_name = ref_model_name
    else:
        checkpoint_name = get_checkpoint_name(option)
    
    print(f'Loading model {checkpoint_name}')
    
    with torch.device(device):
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_name,
            id2label={i:classmap.int2str(i) for i in range(classmap.num_classes)},
            label2id={c:classmap.str2int(c) for c in classmap.names},
            problem_type='multi_label_classification')
    return model, checkpoint_name

# train() implementation partially inspired by: 
# https://github.com/aschern/semeval2020_task11/blob/master/span_identification/ner/run_ner.py
def train():
    global global_step
    
    
    dataloader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size,
        collate_fn=data_collator,
        # generator=torch.Generator(device),    # idiots
        shuffle=True,
        num_workers=1,
    )
    
    model.zero_grad()
    avg_loss = WindowAverage(10)
    
    def local_evaluate():
        if global_step % save_steps and global_step < t_total:
            raise RuntimeError('Syncronous save/eval mode')
            # print('train eval:', evaluate(ds_train))
            # print('eval:', evaluate())
        else:
            eval_checkpoint(global_step, model)
    
    def save_model():
        output_path = os.path.join(checkpoint_dir, "checkpoint-{}".format(global_step))
        # model_to_save = model.module if hasattr(model, "module") else model
        model.save_pretrained(output_path)
        print("Saving model checkpoint to %s", output_path)
    
    if not found_saved_model:
        save_model()
    local_evaluate()
    
    with tqdm(total=t_total, initial=global_step, desc="Iteration", position=0, leave=True) as iterator_bar:
        while iterator_bar.n < iterator_bar.total:
            for step, batch in enumerate(dataloader):
                iterator_bar.set_postfix(loss=avg_loss.val)
            
                model.train()
                
                batch = {key: value.to(device) for key, value in batch.items()}
                
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, batch['labels'])
                
                avg_loss.update(float(loss))

                loss.backward()

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                iterator_bar.update()

                if global_step % save_steps == 0:
                    save_model()
                
                if global_step % eval_steps == 0:
                    local_evaluate()
                    
    
    save_model()
    local_evaluate()

eval_file_name = 'eval_result.json'

def eval_checkpoints():
    ids = list(get_checkpoint_ids())
    rn.shuffle(ids)
    for id in ids:
        eval_checkpoint(id)

def eval_checkpoint(id, model=None):
    checkpoint_name = get_checkpoint_name(id)
    print(checkpoint_name)
    if eval_file_name in os.listdir(path=checkpoint_name):
        with open(os.path.join(checkpoint_name, eval_file_name), 'r') as file:
            checkpoint_res = json.load(file)
        # print('cached - ', checkpoint_res)
    else:
        if model is None:
            model, _ = load_model(id)
        checkpoint_res = {'train': evaluate(ds_train), 'dev': evaluate(ds_dev)}
        print(checkpoint_res)
        with open(os.path.join(checkpoint_name, eval_file_name), 'w') as file:
                json.dump(checkpoint_res, file)

def set_total_train_steps():
    global t_total, save_steps, eval_steps
    t_total = len(ds_train) // batch_size * num_train_epochs
    print('# Train steps:', t_total)
    save_steps = int(np.ceil(len(ds_train) // batch_size * save_epochs))
    print('# Save steps:', save_steps)
    eval_steps = int(np.ceil(len(ds_train) // batch_size * eval_epochs))
    print('# Eval steps:', eval_steps)

def load_objects():
    global tokenizer, ds_train, ds_dev, data_collator, model_option, model, checkpoint_name, optimizer, scheduler, metric, compute_metrics
    tokenizer = load_tokenizer()
    ds_train, ds_dev = load_and_preprocess_data(tokenizer)
    set_total_train_steps()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    model_option = 'last'
    model, checkpoint_name = load_model(model_option)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

def predict_verbose(_set, index):
    sample = _set.select([index]).select_columns([c for c in _set.column_names if c != 'labels'])
    res = trainer.predict(sample)
    for t in zip(res.predictions[0], _set[index]['labels']):
        print(t)

def evaluate(_set=None):
    if _set is None:
        _set = ds_dev

    model.eval()
    
    dataloader = torch.utils.data.DataLoader(
        _set,
        batch_size=batch_size,
        collate_fn=data_collator,
        # generator=torch.Generator(device),    # idiots
        num_workers=1,
    )
    
    epoch_iterator = tqdm(dataloader, desc="Iteration", position=0, leave=True)
    all_outs, all_labels, all_loss = [], [], [0, 0, 0]
    for step, batch in enumerate(epoch_iterator):
        batch = {key: value.to(device) for key, value in batch.items()}
        
        outputs = model(**batch)
        
        all_loss[0] += loss_fn(outputs.logits, batch['labels']).detach().cpu().numpy()
        all_loss[1] += any_loss_fn(outputs.logits, batch['labels']).detach().cpu().numpy()
        all_loss[2] += spec_loss_fn(outputs.logits, batch['labels']).detach().cpu().numpy()
        
        all_outs.append(outputs.logits.detach().cpu().numpy())
        all_labels.append(batch['labels'].detach().cpu().numpy())
    
    for i in range(len(all_loss)):
        all_loss[i] = float(all_loss[i]) / len(dataloader)
    
    return {'loss': all_loss[0],
        'any_loss': all_loss[1],
        'spec_loss': all_loss[2],
        **compute_metrics_multi_label((np.concatenate(all_outs), np.concatenate(all_labels)))
    }

def plot_checkpoint_performance(train=True, dev=True, targets=('f1', 'precision', 'accuracy'), aliases=None):
    if aliases is None:
        aliases = targets

    ids = list(get_checkpoint_ids())
    res = {}
    if train:
        res.update({'train_' + target: {} for target in targets})
    if dev:
        res.update({'dev_' + target: {} for target in targets})
    
    for id in ids:
        checkpoint_name = get_checkpoint_name(id)
        if not eval_file_name in os.listdir(path=checkpoint_name):
            continue
        with open(os.path.join(checkpoint_name, eval_file_name), 'r') as file:
            checkpoint_res = json.load(file)
        if train:
            for target in targets:
                res['train_' + target][id] = checkpoint_res['train'][target]
        if dev:
            for target in targets:
                res['dev_' + target][id] = checkpoint_res['dev'][target]
    
    handles = []
    for i, target in enumerate(targets):
        if train:
            target_res = res['train_' + target]
            handles.append(plt.scatter(list(target_res.keys()), list(target_res.values()), color=f'C{i}', marker='x', label='Train ' + aliases[i]))
        if dev:
            target_res = res['dev_' + target]
            handles.append(plt.scatter(list(target_res.keys()), list(target_res.values()), color=f'C{i}', label='Dev ' + aliases[i]))
    plt.xlabel('Train steps')
    plt.legend(handles=handles)
    plt.show()

def pad_matrix(m, pad_elem=-1):
    max_len = max(len(l) for l in m)
    return [l + [pad_elem for _ in range(max_len - len(l))] for l in m]

def torchify(_dict):
    def lambda_raise(e):
        raise e
    key_to_pad_elem = lambda key: 1 if key=='input_ids' else 0 if key=='attention_mask' else -1 if key=='labels' else lambda_raise(Exception(f'Unkwown key {key}'))
    return {key: torch.tensor(pad_matrix(value, pad_elem=key_to_pad_elem(key)), device=device) for key, value in _dict.items()}

def interactive_predict(s):
    tokens = tokenizer([s], truncation=True)
    ins = torchify(tokens)
    outs = model(**ins).logits.detach().cpu().numpy()[0]
    print('Input:')
    print(s)
    print('Prediction:')
    print(json.dumps({name: "{:.4f}".format(sigmoid(value)) for name, value in zip(data_viewer.keyword_classmap.names, outs)}, indent=4))
    print()

if __name__ == '__main__':
    # do_rebalance_sets = True

    load_objects()
    # interactive_predict('Today I tried to go to the icecream store but I was accosted by reporters.')

    # train()
    # eval_checkpoints()
    
    # plot_checkpoint_performance(
    #     targets=('specific_f1', 'specific_precision', 'specific_recall'),
    #    aliases=('F1', 'Precision', 'Recall')
    #)
