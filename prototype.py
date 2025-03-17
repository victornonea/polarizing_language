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
from util import compute_metrics_multi_label, WindowAverage

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
checkpoint_dir = 'proto_trainer_v2'
learning_rate = 1e-6    # 1e-6 appears to work for specific pattern, 1e-8 appears to work for "any" pattern
warmup_steps = 4000
batch_size = 8    # assume single gpu
num_train_epochs = 23
save_steps = 1000
eval_steps = 500

device = 'cuda'
any_pattern_mask = torch.tensor([1.] + [0.] * 18, device=device)
spec_pattern_mask = torch.tensor([0.] + [1.] * 18, device=device)

intended_label_weights = any_pattern_mask + (1 / 18) * spec_pattern_mask
# learning_rate_ratios = any_pattern_mask + 1000 * spec_pattern_mask
learning_rate_ratios = 100 * (any_pattern_mask + 10 * spec_pattern_mask)
pos_label_weights = 2 * any_pattern_mask + 80 * spec_pattern_mask
net_label_weights = intended_label_weights * learning_rate_ratios / (pos_label_weights + torch.ones(19, device=device))

loss_fn = torch.nn.BCEWithLogitsLoss(
    weight=net_label_weights,
    pos_weight=pos_label_weights
)
any_loss_fn = torch.nn.BCEWithLogitsLoss(
    weight=net_label_weights * any_pattern_mask,
    pos_weight=pos_label_weights
)
spec_loss_fn = torch.nn.BCEWithLogitsLoss(
    weight=net_label_weights * spec_pattern_mask,
    pos_weight=pos_label_weights
)
global_step = None

classmap = data_viewer.pattern_classmap

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
    
    # DEBUG, pref remove
    ds_train = ds_train.remove_columns(['text'])
    ds_dev = ds_dev.remove_columns(['text'])
    
    return ds_train, ds_dev

def get_checkpoint_ids():
    return set([int(match.group(1)) for file_name in os.listdir(path=checkpoint_dir) if (match := re.search(r'checkpoint-(\d+)', file_name))])

def get_checkpoint_name(id):
    return f'{checkpoint_dir}/checkpoint-{id}'

def load_model(option='new'):
    global global_step
    global_step = 0
    if option == 'new':
        checkpoint_name = ref_model_name
    elif option == 'last':
        ids = get_checkpoint_ids()
        if ids:
            checkpoint_name = get_checkpoint_name(max(ids))
            global_step = max(ids)
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
    print('eval:', evaluate())
    
    def save_model():
        output_path = os.path.join(checkpoint_dir, "checkpoint-{}".format(global_step))
        # model_to_save = model.module if hasattr(model, "module") else model
        model.save_pretrained(output_path)
        print("Saving model checkpoint to %s", output_path)
    
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
                    print('eval:',evaluate())
    
    save_model()
    print('final eval:', evaluate())


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
    global t_total
    t_total = len(ds_train) // batch_size * num_train_epochs # in the aschern implementation there is also a gradient_accumulation_steps variable but it does not seem to be relevant
    print('# Train steps:', t_total)
    # trainer_params['lr_scheduler_kwargs'] = {**trainer_params['lr_scheduler_kwargs'], 'num_training_steps': t_total} # not actually necessary, Trainer() should do the same calculation in the background

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
        **compute_metrics_multi_label((np.concat(all_outs), np.concat(all_labels)))
    }

if __name__ == '__main__':
    load_objects()

    # train()
    # eval_checkpoints()
    
    # plot_checkpoint_performance(targets=('any_f1', 'any_precision', 'any_recall', 'specific_f1', 'specific_precision', 'specific_recall'))
