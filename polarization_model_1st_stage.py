from transformers import RobertaTokenizer, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, PreTrainedTokenizerFast, AdamW, RobertaForTokenClassification
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
import util as ut


# https://discuss.huggingface.co/t/multi-label-token-classification/16509/7
# https://www.huggingface.co/transformers/v2.9.1/_modules/transformers/modeling_roberta.html#RobertaForTokenClassification
# torch.nn.BCEWithLogitsLoss
# https://huggingface.co/docs/transformers/tasks/token_classification#preprocess


ref_model_name = "roberta-base"
checkpoint_dir = 'checkpt-polarization'
learning_rate = 3e-6
lr_decay_per_epoch = 0.8
batch_size = 4    # assume single gpu
num_train_epochs = 20
save_epochs = 4
eval_epochs = 4

default_pos_weight = 1.
default_neg_weight = 2.  # we apply a negative weight amplifier to encourange the model to guess low when it does not know

global_step = 0

device = 'cuda'

all_patterns = {    # some of these are ignored
    'Loaded_Language',
    'Quote/Paraphrase/Representation',
    'Heavy_Language',
    'Hyperbole/Oversimplification',
    'Emotional_Language',
    'Factual/Technical/Dry_language',
    'Amplifier/Minimizer',
    'Provocative_Unsubstatiated_Claim',
    'Temper/Reel-in_language',
    'Loaded_Question_/_Loaded_Doubt',
    'Inappropriately_Informal_Tone/Irony',
}

agg_patterns = {
    'Heavy_Language': 0,
    'Loaded_Language': 1,
    'Emotional_Language': 1,
    'Amplifier/Minimizer': 2,
    'Hyperbole/Oversimplification': 3,
    'Provocative_Unsubstatiated_Claim': 3,
    'Inappropriately_Informal_Tone/Irony': 3,
}

num_labels = max(agg_patterns.values()) + 1
arch_num_labels = 2 * num_labels

class RobertaForMixedTokenClassification(RobertaForTokenClassification):
    # Roberta model compatible with
    #   - multi-label token classification
    #   - mixed token and sequence classification
    #   - arbitrary loss label weights
    
    func_full_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    func_neg_loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(0., device=device))
    @classmethod
    def func_pos_loss(cls, *args):
        return cls.func_full_loss(*args) - cls.func_neg_loss(*args)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        loss_pos_weights=1.,
        loss_neg_weights=1.,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            pos_loss = self.__class__.func_pos_loss(logits, labels)
            neg_loss = self.__class__.func_neg_loss(logits, labels)
            
            loss = torch.sum(pos_loss * loss_pos_weights + neg_loss * loss_neg_weights)
            
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


def load_objects(model_option='last', checkpoint_dir_overwrite=None):
    global tokenizer, ds_train, ds_dev, data_collator, model, checkpoint_name, optimizer, scheduler, metric, compute_metrics
    
    if checkpoint_dir_overwrite is not None:
        global checkpoint_dir
        checkpoint_dir = checkpoint_dir_overwrite
    
    tokenizer = load_tokenizer()
    ds_train, ds_dev = load_and_preprocess_data()
    set_total_train_steps()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    model, checkpoint_name = load_model(model_option)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    if checkpoint_name != ref_model_name:
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_name, 'optimizer.pt')))
    last_epoch_completed = global_step // epoch_steps - 1
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay_per_epoch, last_epoch_completed)
    

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(ref_model_name, add_prefix_space=True)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return tokenizer

def load_and_preprocess_data():
    rn.seed(42)

    tr = data_viewer.load_article_set('data/train', '../doccano/labels')
    tr = {art.id: art for art in tr}
    # for each article, split into sentences with active patterns and sentences without
    patterns = {}   # tuple(art, sent_range): [label]
    all_pos_sents = set()      # tuple(art, sent_range)
    all_null_sents = set()     # tuple(art, sent_range)
    agg_pattern_counts = {key: 0 for key in agg_patterns.values()}
    for art in tr.values():
        null_sents = set(art.get_all_line_ranges())
        pos_sents = set()
        for _label in art.labels:
            for label in art.split_label_across_lines(_label):
                if label.value not in agg_patterns:
                    continue
                sent = art.get_line(label)
                if sent not in null_sents and sent not in pos_sents:
                    raise RuntimeError(f'Sentence {sent} not found.')
                
                if (art.id, sent) not in patterns:
                    patterns[(art.id, sent)] = []
                patterns[(art.id, sent)].append(label)
                agg_pattern_counts[agg_patterns[label.value]] += 1
                
                if sent in null_sents:
                    null_sents.remove(sent)
                pos_sents.add(sent)
            
        if len(null_sents) < len(pos_sents):
            print(f'Warning: article {art.id} has fewer null sentences {len(null_sents)} than active sentences {len(pos_sents)}')
            
        sample_null_sents = list(null_sents)[:min(len(null_sents), len(pos_sents))]# rn.sample(list(null_sents), min(len(null_sents), len(pos_sents)))
        wrap_art_id = lambda rs: [(art.id, r) for r in rs]
        
        all_pos_sents.update(wrap_art_id(pos_sents))
        all_null_sents.update(wrap_art_id(sample_null_sents))

    all_pos_sents = sorted(list(all_pos_sents))
    all_null_sents = sorted(list(all_null_sents))

    print('all_pos_sents', len(all_pos_sents))
    print('all_null_sents', len(all_null_sents))
    print('all active patterns', sum(len(l) for l in patterns.values()))
    
    null_count = len(all_pos_sents) + len(all_null_sents)   # we assume all sentences contain null labels, then subtract respective positive counts

    ds_pos = parse_sentences(all_pos_sents, patterns, agg_pattern_counts, null_count, tr)
    ds_neg = parse_sentences(all_null_sents, patterns, agg_pattern_counts, null_count, tr)

    rn.shuffle(ds_pos)
    rn.shuffle(ds_neg)
    
    ds_train = ds_pos[:int(len(ds_pos) * 0.8)] + ds_neg[:int(len(ds_neg) * 0.8)]
    ds_dev = ds_pos[int(len(ds_pos) * 0.8):] + ds_neg[int(len(ds_neg) * 0.8):]
    
    return ds_train, ds_dev

def parse_sentences(all_sents, patterns, agg_pattern_counts, null_count, tr):
    ds = []
    
    neg_counter_weight = {key: agg_pattern_counts[key] / (null_count - agg_pattern_counts[key]) for key in agg_pattern_counts}

    for art_id, sent_range in all_sents:
        art = tr[art_id]
        payload = tokenizer(art.text[slice(*sent_range)], add_special_tokens=True, return_offsets_mapping=True)
        
        num_tokens = len(payload['input_ids'])
        
        labels = [[0. for _ in range(arch_num_labels)] for _ in range(num_tokens)]
        payload['labels'] = labels
        
        for raw_label in patterns.get((art_id, sent_range), []):
            labels[0][num_labels + agg_patterns[raw_label.value]] = 1.0 # sentence level label
            for i in range(num_tokens):
                token_offset = payload['offset_mapping'][i]
                if token_offset[1] - token_offset[0] == 0:
                    continue # trivial token
                token_start_in_art = sent_range[0] + token_offset[0]
                if raw_label.start <= token_start_in_art < raw_label.end:
                    labels[i][agg_patterns[raw_label.value]] = 1.0
        
        num_pos_token_labels = sum(sum(tl) for tl in labels[1:])
        pos_token_weight = default_pos_weight / num_pos_token_labels if num_pos_token_labels else 0.
        
        num_neg_token_labels = sum(sum(l == 0. for l in tl) for tl in labels[1:])
        neg_token_weight = 1 / num_neg_token_labels if num_neg_token_labels else 0.
        
        payload['loss_pos_weights'] = [[*[0. for _ in range(num_labels)], *[default_pos_weight for _ in range(num_labels)]], *[[*[pos_token_weight for _ in range(num_labels)], *[0. for _ in range(num_labels)]] for _ in range(num_tokens - 1)]]
        payload['loss_neg_weights'] = [[*[0. for _ in range(num_labels)], *[default_neg_weight * neg_counter_weight[i] for i in range(num_labels)]], *[[*[default_neg_weight * neg_token_weight * neg_counter_weight[i] for i in range(num_labels)], *[0. for _ in range(num_labels)]] for _ in range(num_tokens - 1)]]
        
        ds.append(payload)
    return ds

def set_total_train_steps():
    global t_total, epoch_steps, save_steps, eval_steps
    epoch_steps = int(np.ceil(len(ds_train) / batch_size))
    t_total = epoch_steps * num_train_epochs
    print('# Train steps:', t_total)
    save_steps = int(np.ceil(len(ds_train) // batch_size * save_epochs))
    print('# Save steps:', save_steps)
    eval_steps = int(np.ceil(len(ds_train) // batch_size * eval_epochs))
    print('# Eval steps:', eval_steps)

def get_checkpoint_ids():
    return set([int(match.group(1)) for file_name in os.listdir(path=checkpoint_dir) if (match := re.search(r'checkpoint-(\d+)', file_name))])

def get_checkpoint_name(id):
    return f'{checkpoint_dir}/checkpoint-{id}'

found_saved_model = False

def load_model(option='last'):
    global global_step, found_saved_model
    
    if option != 'new':
        ids = get_checkpoint_ids()
        if ids:
            if option == 'last':
                checkpoint_name = get_checkpoint_name(max(ids))
                global_step = max(ids)
                found_saved_model = True
            else:
                id = option
                checkpoint_name = get_checkpoint_name(id)
                global_step = id
                found_saved_model = True
    
    
    if option == 'new' or not global_step:
        checkpoint_name = ref_model_name
    
    print(f'Loading model {checkpoint_name}')
    
    with torch.device(device):
        model = RobertaForMixedTokenClassification.from_pretrained(checkpoint_name, num_labels=arch_num_labels)
    return model, checkpoint_name

class Dataloader:
    def __init__(self, ds, start_step=0, total=None):
        self.ds = ds
        self.start_step = start_step
        if total is None:
            total = int(np.ceil(len(ds) / batch_size))
        self.total = total
    
    def __len__(self):
        return self.total
    
    def __call__(self):
        ds_view = None
        def get_shuffle(i):
            nonlocal ds_view
            ds_view = list(self.ds)
            rn.seed(i)
            rn.shuffle(ds_view)
        
        epochs_done = self.start_step // epoch_steps
        get_shuffle(epochs_done)

        for step in range(self.start_step, self.total):
            epochs_done = step // epoch_steps
            curr_batch = step - epochs_done * epoch_steps
            if curr_batch == 0:
                get_shuffle(epochs_done)
            
            i1 = curr_batch * batch_size
            i2 = min((curr_batch + 1) * batch_size, len(ds_train))
            
            yield make_batch(ds_view[i1:i2])

def pad_matrix(m, pad_elem=-1):
    max_len = max(len(l) for l in m)
    return [l + [pad_elem for _ in range(max_len - len(l))] for l in m]

def make_batch(rows):
    get_partial_matrix = lambda key, rows: [row[key] for row in rows]
    def chain(key, pad_elem):
        nonlocal rows
        return torch.tensor(pad_matrix(get_partial_matrix(key, rows), pad_elem), device=device)
    
    zero_vector = [0. for _ in range(arch_num_labels)]
    
    pad_map = {
        'input_ids': tokenizer.pad_token_id,
        'labels': zero_vector,
        'loss_pos_weights': zero_vector,
        'loss_neg_weights': zero_vector,
        'attention_mask': 0,
    }
    
    return {key: chain(key, value) for key, value in pad_map.items() if key in rows[0]}

def train():
    global global_step
    
    model.zero_grad()
    avg_loss = ut.WindowAverage(10)
    
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
        torch.save(optimizer.state_dict(), os.path.join(output_path, 'optimizer.pt'))
        torch.save(scheduler.state_dict(), os.path.join(output_path, 'scheduler.pt'))
        print("Saving model checkpoint to %s", output_path)
    
    if not found_saved_model:
        save_model()
    local_evaluate()
    
    with tqdm(total=t_total, initial=global_step, desc="Iteration", position=0, leave=True) as iterator_bar:
        for batch in Dataloader(ds_train, global_step, t_total)():
            
            iterator_bar.set_postfix(loss=avg_loss.val)
        
            model.train()
            
            outputs = model(**batch)
            loss = outputs[0]
            
            avg_loss.update(float(loss))

            loss.backward()
            optimizer.step()
            model.zero_grad()
            
            global_step += 1
            iterator_bar.update()
            
            if global_step % epoch_steps == 0:
                scheduler.step()

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

def evaluate(_set=None):
    if _set is None:
        _set = ds_dev

    model.eval()
    
    dataloader = Dataloader(_set)
    
    epoch_iterator = tqdm(dataloader(), desc="Iteration", position=0, leave=True)
    loss = 0
    class_logits = [[] for _ in range(arch_num_labels)]
    class_labels = [[] for _ in range(arch_num_labels)]
    
    for step, batch in enumerate(epoch_iterator):
        outputs = model(**batch)
        
        loss += outputs[0].detach().cpu().numpy()
        
        logits = torch.nn.Sigmoid()(outputs[1].detach().cpu()).numpy()
        
        for i in range(num_labels):
            class_logits[num_labels + i].extend(logits[:, 0, num_labels + i].reshape(-1))   # sentence level
            class_labels[num_labels + i].extend(batch['labels'].cpu().numpy()[:, 0, num_labels + i].reshape(-1))   # sentence level
            
            class_logits[i].extend(logits[:, 1:, i].reshape(-1))    # token level
            class_labels[i].extend(batch['labels'].cpu().numpy()[:, 1:, i].reshape(-1))    # token level
    
    class_id_to_str = lambda i: f'{i}-TOK' if i < num_labels else f'{i - num_labels}-SEQ'
    
    return {
        'loss': loss / len(dataloader),
        **{'CPP' + class_id_to_str(i): np.corrcoef(np.array([class_logits[i], class_labels[i]]))[0, 1] for i in range(arch_num_labels)}
    }

def interactive_predict(s):
    tokens = tokenizer(s, add_special_tokens=True)
    ins = make_batch([tokens])
    outs = model(**ins)[0].detach().cpu().numpy()[0]
    
    predictor_func = lambda x: ut.sigmoid(x)   # we may apply an additional square to enforce a more conservative signal estimate and increase confidence in positive signals
    
    print('Input:')
    print(s)
    print('Prediction:')
    print(json.dumps({str([key for key, val in agg_patterns.items() if val == i]): "{:.4f}".format(predictor_func(outs[0][num_labels + i])) for i in range(num_labels)}, indent=4))
    print()

if __name__ == '__main__':
    load_objects()
    train()
