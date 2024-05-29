import argparse
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.cluster import KMeans
from args import add_args
from models.encoder import DocREModel
from models.classifier import Linear_Classifier
from utils import set_seed, create_directory, parse_relations
from new_data_loader import matrix_get_data_loader
from evaluation import official_evaluate_new, to_official_new
from tqdm import tqdm
import torch.nn.functional as F
import random
from sampler import data_sampler
from config import Config
from losses import matrix_AS_ATLoss
import torch.nn as nn
import logging
from copy import deepcopy
torch.multiprocessing.set_sharing_strategy('file_system')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def get_class_importance(data):
    labels = []
    for sample in data:
        labels.extend(sample["labels"])
    labels = torch.tensor(labels, dtype=torch.float)

    labels_sum = labels.sum(0)
    labels_all = torch.tensor([labels.shape[0]]*labels.shape[1], dtype=torch.float)
    labels_importance = (labels_all-labels_sum) / labels_all
    negative_importance = labels_sum / labels_all
    return labels_importance, negative_importance


logger = get_logger('log/results.log')


def load_input(batch, device, relation_cls=None,tag="dev"):

    input = {'input_ids': batch[0].to(device),
             'attention_mask': batch[1].to(device),
             'labels': batch[2],
             'entity_pos': batch[3],
             'observe_matrices': batch[10],
             'hts': batch[4],
             'sent_pos': batch[5],
             'sent_labels': batch[6].to(device) if (not batch[6] is None) and (batch[7] is None) else None,
             'teacher_attns': batch[7].to(device) if not batch[7] is None else None,
             'graph': batch[8],
             'titles': batch[9],
             'relation_cls': relation_cls.to(device) if not relation_cls is None else None,
             'tag': tag
             }

    return input


def train(args, config, encoder, classifier, train_features, valid_data, historic_test_data, current_relations, seen_relations, epochs, prev_encoder=None, prev_classifier=None, prev_relations=None):
    def finetune(features, optimizer, num_epoch, num_steps):
        train_dataloader = matrix_get_data_loader(config, features, shuffle=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        scaler = GradScaler()
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        total_loss = 0
        best_score = 0
        loss_fnt = matrix_AS_ATLoss()
        loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")
        loss_fnt_distill = nn.KLDivLoss(reduction="batchmean")
        labels_importance, negative_importance = get_class_importance(train_features)
        distill_criterion = nn.CosineEmbeddingLoss()
        T = config.kl_temp
        prev_relations_index = []
        for rel in prev_relations:
            prev_relations_index.append(seen_relations.index(rel))
        prev_relations_index = torch.tensor(prev_relations_index).to(config.device)
        current_relations_index = []
        for rel in current_relations:
            current_relations_index.append(seen_relations.index(rel))
        current_relations_index = torch.tensor(current_relations_index).to(config.device)
        for epoch in train_iterator:
            losses = []
            progress_bar = tqdm(range(len(train_dataloader)))
            progress_bar.set_description(f'epoch: {epoch} loss: {0:>7f}')
            finish_batch_num = epoch * len(train_dataloader)
            for step, batch in enumerate(train_dataloader, start=1):
                encoder.zero_grad()
                classifier.zero_grad()
                optimizer.zero_grad()
                encoder.train()
                classifier.train()

                inputs = load_input(batch, config.device)

                results = encoder(**inputs)
                reps = torch.cat((results[0][0], results[0][0]), 1)
                normalized_outputs_emb = F.normalize(reps.view(-1, reps.size()[1]), p=2, dim=1)
                logits = classifier(results[0][0],results[0][1])

                labels = [torch.tensor(label, dtype=torch.float32) for label in inputs["labels"]]
                labels = torch.cat(labels, dim=0).to(config.device)
                # if prev_classifier is not None:
                #     pre_logits = prev_classifier(results[0][0], results[0][1])
                #     pred = loss_fnt.get_label(pre_logits, len(prev_relations))
                #     labels[:, 1:len(prev_relations)] = pred[:, 1:]
                #     labels[:, 0] = (labels[:,1:].sum(1) == 0.).to(config.device)

                observe_matrices = [torch.tensor(observe_matrix, dtype=torch.bool) for observe_matrix in inputs["observe_matrices"]]
                observe_matrices = torch.cat(observe_matrices, dim=0).to(config.device)
                loss = [loss_fnt(logits.float(), labels, labels_importance.to(config.device), negative_importance.to(config.device), observe_matrices)]

                if inputs["sent_labels"] is not None:
                    idx_used = torch.nonzero(labels[:, 1:].sum(dim=-1)).view(-1)
                            # evidence retrieval loss (kldiv loss)
                    s_attn = results[1]["s_attn"][idx_used]
                    sent_labels = inputs["sent_labels"][idx_used]
                    norm_s_labels = sent_labels / (sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                    norm_s_labels[norm_s_labels == 0] = 1e-30
                    s_attn[s_attn == 0] = 1e-30
                    evi_loss = loss_fnt_evi(s_attn.log(), norm_s_labels)
                    loss.append(evi_loss * args.evi_lambda)
                if inputs["teacher_attns"] is not None:
                    results[1]["doc_attn"][results[1]["doc_attn"] == 0] = 1e-30
                    inputs["teacher_attns"][inputs["teacher_attns"] == 0] = 1e-30
                    attn_loss = loss_fnt_evi(results[1]["doc_attn"], inputs["teacher_attns"])
                    loss.append(attn_loss * args.attn_lambda)

                if prev_classifier is not None:
                    pre_logits = prev_classifier(results[0][0], results[0][1]).view(-1, 1)
                    sigmoid_pre_logits = torch.cat((torch.sigmoid(pre_logits),1-torch.sigmoid(pre_logits)),dim=-1)
                    sigmoid_pre_logits = torch.clamp(sigmoid_pre_logits,min=config.eps,max=1-config.eps)
                    cur_logits = logits.index_select(1, prev_relations_index).view(-1, 1)
                    sigmoid_cur_logits = torch.cat((torch.sigmoid(cur_logits),1-torch.sigmoid(cur_logits)),dim=-1)
                    sigmoid_cur_logits = torch.clamp(sigmoid_cur_logits,min=config.eps,max=1-config.eps)
                    distill_loss = loss_fnt_distill(F.log_softmax(sigmoid_cur_logits, dim=1), F.softmax(sigmoid_pre_logits,dim=1))
                    loss.append(distill_loss)
                if len(prev_relations) > 0:
                    pred = torch.sigmoid(logits)
                    label_one_hot = torch.clamp(labels, min=1e-4, max=1.0)
                    sys_loss = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1).mean()
                    loss.append(0.1 * sys_loss)


                loss = sum(loss) / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                losses.append(loss.item())
                total_loss += loss.item()
                progress_bar.set_description(f'epoch: {epoch}  loss: {total_loss / (finish_batch_num + step):>7f}')
                progress_bar.update(1)
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    encoder.zero_grad()
                    classifier.zero_grad()
                    num_steps += 1
            history_test_score, history_test_output = evaluate(config, encoder, classifier,
                                                               historic_test_data, seen_relations, seen_relations)
            print("history_test_score:")
            print(history_test_score)
            logger.info('history_test_score:{}\t'.format(history_test_score))
            if best_score < history_test_score:
                best_score = history_test_score
                torch.save(encoder, args.save_path + "encoder.pt")
                torch.save(classifier, args.save_path + "classifier.pt")
            logger.info('Epoch {} loss:{}\t'.format(epoch, np.array(losses).mean()))
        return num_steps

    new_layer = ["extractor", "bilinear", "graph"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in encoder.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in encoder.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
        {"params": [p for n, p in classifier.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in classifier.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_transformer, eps=args.adam_epsilon)
    num_steps = 0
    # set_seed(args)
    encoder.zero_grad()
    classifier.zero_grad()
    finetune(train_features, optimizer, epochs, num_steps)



def train_mem_model(args, config, encoder, classifier, training_data, epochs, prev_encoder, prev_classifier, prev_relations, new_classifier, current_relations, seen_relations, historic_test_data, best_score):
    new_layer = ["extractor", "bilinear", "graph"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in encoder.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in encoder.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
        {"params": [p for n, p in classifier.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in classifier.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_transformer, eps=args.adam_epsilon)
    num_steps = 0
    # set_seed(args)
    encoder.zero_grad()
    classifier.zero_grad()
    train_dataloader = matrix_get_data_loader(config, training_data, shuffle=True)
    train_iterator = range(int(epochs))
    total_steps = int(len(train_dataloader) * epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scaler = GradScaler()
    print("Total steps: {}".format(total_steps))
    print("Warmup steps: {}".format(warmup_steps))
    total_loss = 0
    loss_fnt = matrix_AS_ATLoss()
    labels_importance, negative_importance = get_class_importance(training_data)
    loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")
    loss_fnt_distill = nn.KLDivLoss(reduction="batchmean")
    distill_criterion = nn.CosineEmbeddingLoss()
    T = config.kl_temp
    prev_relations_index = []
    for rel in prev_relations:
        prev_relations_index.append(seen_relations.index(rel))
    prev_relations_index = torch.tensor(prev_relations_index).to(config.device)
    current_relations_index = []
    for rel in current_relations:
        current_relations_index.append(seen_relations.index(rel))
    current_relations_index = torch.tensor(current_relations_index).to(config.device)
    for epoch in train_iterator:
        losses = []
        progress_bar = tqdm(range(len(train_dataloader)))
        progress_bar.set_description(f'epoch: {epoch} loss: {0:>7f}')
        finish_batch_num = epoch * len(train_dataloader)
        for step, batch in enumerate(train_dataloader, start=1):
            encoder.zero_grad()
            classifier.zero_grad()
            optimizer.zero_grad()
            encoder.train()
            classifier.train()

            inputs = load_input(batch, config.device)
            results = encoder(**inputs)
            logits = classifier(results[0][0], results[0][1])
            reps = torch.cat((results[0][0], results[0][0]), 1)
            normalized_outputs_emb = F.normalize(reps.view(-1, reps.size()[1]), p=2, dim=1)
            labels = [torch.tensor(label, dtype=torch.float32) for label in inputs["labels"]]
            labels = torch.cat(labels, dim=0).to(config.device)

            observe_matrices = [torch.tensor(observe_matrix, dtype=torch.bool) for observe_matrix in
                                inputs["observe_matrices"]]
            observe_matrices = torch.cat(observe_matrices, dim=0).to(config.device)
            loss = [loss_fnt(logits.float(), labels, labels_importance.to(config.device),
                             negative_importance.to(config.device), observe_matrices)]
            if inputs["sent_labels"] is not None:
                idx_used = torch.nonzero(labels[:, 1:].sum(dim=-1)).view(-1)
                # evidence retrieval loss (kldiv loss)
                s_attn = results[1]["s_attn"][idx_used]
                sent_labels = inputs["sent_labels"][idx_used]
                norm_s_labels = sent_labels / (sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30
                s_attn[s_attn == 0] = 1e-30
                evi_loss = loss_fnt_evi(s_attn.log(), norm_s_labels)
                loss.append(evi_loss * args.evi_lambda)
            if inputs["teacher_attns"] is not None:
                results[1]["doc_attn"][results[1]["doc_attn"] == 0] = 1e-30
                inputs["teacher_attns"][inputs["teacher_attns"] == 0] = 1e-30
                attn_loss = loss_fnt_evi(results[1]["doc_attn"], inputs["teacher_attns"])
                loss.append(attn_loss * args.attn_lambda)

            if prev_classifier is not None:
                pre_logits = prev_classifier(results[0][0], results[0][1]).view(-1, 1)
                sigmoid_pre_logits = torch.cat((torch.sigmoid(pre_logits), 1 - torch.sigmoid(pre_logits)), dim=-1)
                sigmoid_pre_logits = torch.clamp(sigmoid_pre_logits, min=config.eps, max=1 - config.eps)
                cur_logits = logits.index_select(1, prev_relations_index).view(-1, 1)
                sigmoid_cur_logits = torch.cat((torch.sigmoid(cur_logits), 1 - torch.sigmoid(cur_logits)), dim=-1)
                sigmoid_cur_logits = torch.clamp(sigmoid_cur_logits, min=config.eps, max=1 - config.eps)
                distill_loss = loss_fnt_distill(F.log_softmax(sigmoid_cur_logits, dim=1),
                                                F.softmax(sigmoid_pre_logits, dim=1))
                loss.append(distill_loss)

            if new_classifier is not None:
                new_logits = new_classifier(results[0][0], results[0][1]).index_select(1, current_relations_index).view(-1, 1)
                sigmoid_new_logits = torch.cat((torch.sigmoid(new_logits), 1 - torch.sigmoid(new_logits)), dim=-1)
                sigmoid_new_logits = torch.clamp(sigmoid_new_logits, min=config.eps, max=1 - config.eps)
                cur_logits = logits.index_select(1, current_relations_index).view(-1, 1)
                sigmoid_cur_logits = torch.cat((torch.sigmoid(cur_logits), 1 - torch.sigmoid(cur_logits)), dim=-1)
                sigmoid_cur_logits = torch.clamp(sigmoid_cur_logits, min=config.eps, max=1 - config.eps)
                distill_loss = loss_fnt_distill(F.log_softmax(sigmoid_cur_logits, dim=1),
                                                F.softmax(sigmoid_new_logits, dim=1))
                loss.append(distill_loss)

            loss = sum(loss) / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            losses.append(loss.item())
            total_loss += loss.item()
            progress_bar.set_description(f'epoch: {epoch}  loss: {total_loss / (finish_batch_num + step):>7f}')
            progress_bar.update(1)
            if step % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                encoder.zero_grad()
                classifier.zero_grad()
                num_steps += 1
        history_test_score, history_test_output = evaluate(config, encoder, classifier,
                                                           historic_test_data, seen_relations, seen_relations)
        print("history_test_score:")
        print(history_test_score)
        logger.info('history_test_score:{}\t'.format(history_test_score))
        if best_score < history_test_score:
            best_score = history_test_score
            torch.save(encoder, args.save_path + "encoder.pt")
            torch.save(classifier, args.save_path + "classifier.pt")
        logger.info('Epoch {} loss:{}\t'.format(epoch, np.array(losses).mean()))


def evaluate(config,  encoder, classifier, test_data, current_relations, seen_relations, tag="dev"):

    data_loader = matrix_get_data_loader(config, test_data)
    preds, evi_preds = [], []
    scores, topks = [], []

    loss_fnt = matrix_AS_ATLoss()

    for batch in data_loader:
        encoder.eval()
        classifier.eval()

        inputs = load_input(batch, config.device)

        with torch.no_grad():
            results = encoder(**inputs)
            logits = classifier(results[0][0], results[0][1])
            pred = loss_fnt.get_label(logits, len(seen_relations)).cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    ans = to_official_new(preds, test_data, current_relations,seen_relations)
    best_f1 = 0
    if len(ans) > 0:
        best_f1 = official_evaluate_new(ans, test_data, current_relations, seen_relations)
    output = {
        tag + "_F1": best_f1 * 100,

    }
    return best_f1, output


def select_memory(config, encoder, classifier, training_data, current_relations, seen_relations):
    data_loader = matrix_get_data_loader(config, training_data, batch_size=1)
    encoder.eval()
    classifier.eval()
    rel2titles = {}
    rel2hiddens = {}

    for rel in current_relations:
        if rel != "No relation":
            rel2titles[rel] = []
            rel2hiddens[rel] = []
    for k, batch in enumerate(data_loader):

        inputs = load_input(batch, config.device)

        with torch.no_grad():
            results = encoder(**inputs)
            hiddens = torch.cat([results[0][0], results[0][0]],dim=1)
            for index, label in enumerate(inputs["labels"]):
                for i in range(len(label)):
                    for key in rel2titles.keys():
                        if label[i][seen_relations.index(key)] == 1:
                            rel2hiddens[key].append(hiddens[index].unsqueeze(0).cpu())
                            rel2titles[key].append(inputs["titles"][index])

    select_titles = []
    for rel in rel2hiddens.keys():
        rel2hiddens[rel] = np.concatenate(rel2hiddens[rel], axis=0)
        k = min(config.memory_size, len(rel2hiddens[rel]))
        kmeans = KMeans(n_clusters=k, random_state=0)
        distances = kmeans.fit_transform(rel2hiddens[rel])
        for j in range(k):

            select_idx = np.argmin(distances[:, j])
            select_titles.append(rel2titles[rel][select_idx])
    memory_data = []

    for sample in training_data:

        if sample["title"] in select_titles:
            memory_data.append(sample)

    return memory_data



def merge_memory(history_data, current_data, seen_relations, label_splits):
    memory_data = []
    seen_titles = []
    for history_sample in history_data:
        same_sample = None
        for sample in current_data:
            if sample["title"] == history_sample["title"]:
                same_sample = sample
                break
        if same_sample is None:
            new_labels = [history_sample["labels"][i]+[0]*(len(seen_relations)-len(history_sample["labels"][i])) for i in range(len(history_sample["labels"]))]
            observe_matrices = [
                history_sample["observe_matrix"][i] + [0] * (len(seen_relations) - len(history_sample["observe_matrix"][i]))
                for i in range(len(history_sample["observe_matrix"]))]
        else:
            new_labels = [[0]*len(seen_relations) for _ in range(len(history_sample["labels"]))]
            observe_matrices = [
                history_sample["observe_matrix"][i] + [0] * (
                            len(seen_relations) - len(history_sample["observe_matrix"][i]))
                for i in range(len(history_sample["observe_matrix"]))]
            observe_matrices = [[observe_matrices[i][0]]+(np.clip(np.sum([observe_matrices[i][1:], same_sample["observe_matrix"][i][1:]], axis=0), 0, 1).tolist()) for i in range(len(history_sample["observe_matrix"]))]

            for index, label in enumerate(new_labels):
                new_labels[index][1:label_splits[-2]] = history_sample["labels"][index][1:label_splits[-2]]
                new_labels[index][label_splits[-2]:] = same_sample["labels"][index][label_splits[-2]:]
                if sum(new_labels[index]) == 0:
                    new_labels[index][0] = 1
        new_sample = {
            "input_ids": history_sample["input_ids"],
            "entity_pos": history_sample["entity_pos"],
            "labels": new_labels,
            "observe_matrix": observe_matrices,
            "hts": history_sample["hts"],
            "sent_pos": history_sample["sent_pos"],
            "sent_labels": history_sample["sent_labels"],
            "title": history_sample["title"],
            "graph": history_sample["graph"]
        }
        memory_data.append(new_sample)
        seen_titles.append(history_sample["title"])

    for sample in current_data:
        if sample["title"] not in seen_titles:
            memory_data.append(sample)

    return memory_data




def calculate_correlation(relation_cls):
    threshold = 0.95
    correlation_matrix = torch.zeros([len(relation_cls), len(relation_cls)])
    for i in range(1, len(relation_cls)):
        for j in range(1, len(relation_cls)):
            similarity = torch.cosine_similarity(relation_cls[i], relation_cls[j],dim=0)
            if similarity > threshold and i != j:
                correlation_matrix[i][j] = similarity
    return correlation_matrix



def relabel(config, encoder, classifier, data, trained_relations, label_splits, correlation_matrix, tag):
    data_loader = matrix_get_data_loader(config, data, batch_size=1)
    encoder.eval()
    classifier.eval()
    loss_fnt = matrix_AS_ATLoss()
    relabelled_data = []
    for index, batch in enumerate(data_loader):
        inputs = load_input(batch, config.device)

        with torch.no_grad():
            results = encoder(**inputs)
            logits = classifier(results[0][0], results[0][1])
            pred = loss_fnt.get_label_cor(logits, correlation_matrix, len(trained_relations)).cpu().numpy()

        if tag == "new":
            for sample_index in range(len(inputs["input_ids"])):
                new_labels = [inputs["labels"][sample_index][k][:] for k in range(len(inputs["labels"][sample_index]))]
                observe_matrices = [inputs["observe_matrices"][sample_index][k][:] for k in range(len(inputs["observe_matrices"][sample_index]))]
                for i, label in enumerate(new_labels):
                    new_labels[i][1:label_splits[-1]] = pred[i][1:label_splits[-1]]
                    observe_matrices[i][1:label_splits[-1]] = [1]*(label_splits[-1]-1)
                    new_labels[i][label_splits[-1]:] = inputs["labels"][sample_index][i][label_splits[-1]:]
                    if sum(new_labels[i]) == 0:
                        new_labels[i][0] = 1
                sent_labels = None
                for sample in data:
                    if sample["title"] == inputs["titles"][sample_index]:
                        sent_labels = sample["sent_labels"][:]
                        break
                new_sample = {
                    "input_ids": batch[0][sample_index],
                    "entity_pos": batch[3][sample_index],
                    "labels": new_labels,
                    "observe_matrix": observe_matrices,
                    "hts": inputs["hts"][sample_index],
                    "sent_pos": inputs["sent_pos"][sample_index],
                    "sent_labels": sent_labels,
                    "title": inputs["titles"][sample_index],
                    "graph": inputs["graph"][sample_index]
                }
                relabelled_data.append(new_sample)
        elif tag == "memory":
            for sample_index in range(len(batch[0])):
                new_labels = [[0] * len(trained_relations) for _ in range(len(inputs["labels"][sample_index]))]
                observe_matrices = [[0] * len(trained_relations) for _ in range(len(inputs["observe_matrices"][sample_index]))]
                for i, label in enumerate(new_labels):
                    new_labels[i][1:len(inputs["labels"][sample_index][i])] = inputs["labels"][sample_index][i][1:len(inputs["labels"][sample_index][i])].copy()
                    new_labels[i][len(inputs["labels"][sample_index][i]):] = pred[i][len(inputs["labels"][sample_index][i]):].copy()
                    observe_matrices[i][:len(inputs["labels"][sample_index][i])] = inputs["observe_matrices"][sample_index][i][:len(inputs["labels"][sample_index][i])].copy()
                    observe_matrices[i][len(inputs["labels"][sample_index][i]):] = [1]*(len(trained_relations)-len(inputs["labels"][sample_index][i]))
                    if sum(new_labels[i]) == 0:
                        new_labels[i][0] = 1
                sent_labels = None
                for sample in data:
                    if sample["title"] == inputs["titles"][sample_index]:
                        sent_labels = sample["sent_labels"][:]
                        break
                new_sample = {
                    "input_ids": batch[0][sample_index],
                    "entity_pos": batch[3][sample_index],
                    "labels": new_labels,
                    "observe_matrix": observe_matrices,
                    "hts": inputs["hts"][sample_index],
                    "sent_pos": inputs["sent_pos"][sample_index],
                    "sent_labels": sent_labels,
                    "title": inputs["titles"][sample_index],
                    "graph": inputs["graph"][sample_index]
                }
                relabelled_data.append(new_sample)
    return relabelled_data

def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser.add_argument("--task", default="DocRED", type=str)
    parser.add_argument("--memory_size", default=10, type=str)
    parser.add_argument('--config', default='config.ini')
    parser.add_argument("--model_name_or_path", default="./bert-base-uncased", type=str)
    parser.add_argument("--save_path", default="./save/", type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = Config(args.config)

    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)
    config.memory_size = args.memory_size

    if args.task == "DocRED":
        config.relation_file = "./data/DocRED/rel2id.json"
        config.task_splits_dir = "./data/DocRED/select"
        config.relation_info_file = "./data/DocRED/rel_info_with_descriptions.json"
    else:
        config.relation_file = "./data/ReDocRED/rel2id.json"
        config.task_splits_dir = "./data/ReDocRED/select"
        config.relation_info_file = "./data/DocRED/rel_info_with_descriptions.json"

    result_cur_test = []
    result_whole_test = []
    relation_model = AutoModel.from_pretrained(
        args.model_name_or_path
    )
    for i in range(config.total_round):
        test_cur = []
        test_total = []
        random.seed(config.seed + i * 100)
        sampler = data_sampler(config=config, seed=config.seed + i * 100)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path
        )
        model = AutoModel.from_pretrained(
            args.model_name_or_path
        )

        bert_config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=args.num_class,
        )

        bert_config.transformer_type = args.transformer_type


        bert_config.cls_token_id = tokenizer.cls_token_id
        bert_config.sep_token_id = tokenizer.sep_token_id

        encoder = DocREModel(args, bert_config, model, tokenizer,
                           num_labels=args.num_labels,
                           max_sent_num=args.max_sent_num,
                           evi_thresh=args.evi_thresh).to(config.device)


        create_directory(args.save_path)
        history_test = []
        memory_data = []
        prev_classifier = None
        prev_encoder = None
        prev_relations = []
        history_relations = []
        label_splits = [0]
        for steps, (
        training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(
                sampler):
            print(len(seen_relations))
            print(len(training_data[0]["labels"][0]))
            print(len(valid_data[0]["labels"][0]))
            print(len(test_data[0]["labels"][0]))
            print(test_data[-1]["labels"][0])
            print(historic_test_data[-1]["labels"][0])
            logger.info('Round:{}\t Task:{}\t classes:{}\t'.format(i, steps, len(seen_relations)))
            logger.info('current_labels:{}\t'.format(test_data[-1]["labels"][0]))
            logger.info('history_labels:{}\t'.format(historic_test_data[-1]["labels"][0]))
            history_test.append(test_data)
            history_relations.append(current_relations)
            classifier = Linear_Classifier(num_class=len(seen_relations)).to(config.device)
            if prev_classifier is not None:
                with torch.no_grad():
                    classifier.bilinear.weights[:len(prev_relations),:].copy_(prev_classifier.bilinear.weights)
                    classifier.bilinear.bias[:len(prev_relations)].copy_(prev_classifier.bilinear.bias)
            relabel_training_data = training_data

            seen_relation_tokens, seen_relation_mask = parse_relations(config.relation_info_file, tokenizer,
                                                                       seen_relations)
            seen_relation_output = relation_model(input_ids=seen_relation_tokens, attention_mask=seen_relation_mask)[0]
            seen_relation_cls = seen_relation_output[:, 0, :]

            correlation_matrix = calculate_correlation(seen_relation_cls)
            if prev_classifier is not None:
                relabel_training_data = relabel(config, encoder, prev_classifier, training_data, prev_relations, label_splits, correlation_matrix, tag="new")

            label_splits.append(len(seen_relations))

            if steps != 0:
                train(args, config, encoder, classifier, relabel_training_data, valid_data, historic_test_data, current_relations, seen_relations, config.step1_epochs, prev_encoder, prev_classifier, prev_relations)
            else:
                train(args, config, encoder, classifier, relabel_training_data, valid_data, historic_test_data,
                      current_relations, seen_relations, config.step1_epochs, prev_relations=prev_relations)
            print(f"simple finished")
            logger.info('simple finished')
            encoder = torch.load(args.save_path + "encoder.pt")
            classifier = torch.load(args.save_path + "classifier.pt")
            dev_score, dev_output = evaluate(config, encoder, classifier, valid_data, current_relations,
                                             seen_relations)
            print("dev_score:")
            print(dev_score)
            logger.info('dev_score:{}\t'.format(dev_score))
            test_score, test_output = evaluate(config, encoder, classifier, test_data, current_relations,
                                               seen_relations)
            print("test_score:")
            print(test_score)
            logger.info('test_score:{}\t'.format(test_score))

            history_test_score, history_test_output = evaluate(config, encoder, classifier,
                                                               historic_test_data, seen_relations, seen_relations)
            print("history_test_score:")
            print(history_test_score)
            logger.info('history_test_score:{}\t'.format(history_test_score))
            best_score = history_test_score
            history_scores = []
            for index, data in enumerate(history_test):
                socre, output = evaluate(config, encoder, classifier, data, history_relations[index],
                                         seen_relations)
                history_scores.append(socre)

            logger.info('history_scores:{}\t'.format(history_scores))


            new_memory = select_memory(config, encoder, classifier, relabel_training_data,
                                       current_relations, seen_relations)
            if len(memory_data) > 0:
                relabel_memory_data = relabel(config, encoder, classifier, memory_data, seen_relations, label_splits, correlation_matrix,
                                              tag="memory")
            else:
                relabel_memory_data = memory_data
            relabel_memory_data = merge_memory(relabel_memory_data, new_memory, seen_relations, label_splits)
            memory_data = relabel_memory_data[:]

            if steps != 0:
                current_classifier = deepcopy(classifier)
                train_mem_model(args, config, encoder, classifier, relabel_memory_data, config.step2_epochs,
                                prev_encoder, prev_classifier, prev_relations,current_classifier, current_relations, seen_relations, historic_test_data, best_score)
                print(f"memory finished")
                logger.info('memory finished')
                encoder = torch.load(args.save_path + "encoder.pt")
                classifier = torch.load(args.save_path + "classifier.pt")


            dev_score, dev_output = evaluate(config, encoder, classifier, valid_data, current_relations,
                                             seen_relations)
            print("dev_score:")
            print(dev_score)
            logger.info('dev_score:{}\t'.format(dev_score))
            test_score, test_output = evaluate(config, encoder,  classifier, test_data, current_relations,
                                               seen_relations)
            print("test_score:")
            print(test_score)
            logger.info('test_score:{}\t'.format(test_score))
            test_cur.append(test_score)

            history_test_score, history_test_output = evaluate(config, encoder, classifier,
                                                               historic_test_data, seen_relations, seen_relations)
            print("history_test_score:")
            print(history_test_score)
            logger.info('history_test_score:{}\t'.format(history_test_score))
            test_total.append(history_test_score)
            history_scores = []
            for index, data in enumerate(history_test):
                socre, output = evaluate(config, encoder, classifier, data, history_relations[index],
                                         seen_relations)
                history_scores.append(socre)

            logger.info('history_scores:{}\t'.format(history_scores))

            for rel in current_relations:
                if rel not in prev_relations:
                    prev_relations.append(rel)
            prev_encoder = deepcopy(encoder)
            prev_classifier = deepcopy(classifier)
            torch.cuda.empty_cache()
        result_cur_test.append(np.array(test_cur))
        result_whole_test.append(np.array(test_total))
        print("result_cur_test")
        print(result_cur_test)
        print("result_whole_test")
        print(result_whole_test)
        avg_result_cur_test = np.average(result_cur_test, 0)
        avg_result_all_test = np.average(result_whole_test, 0)
        print("avg_result_cur_test")
        print(avg_result_cur_test)
        print("avg_result_all_test")
        print(avg_result_all_test)
        logger.info('result_cur_test:{}\t'.format(result_cur_test))
        logger.info('result_whole_test:{}\t'.format(result_whole_test))
        logger.info('avg_result_cur_test:{}\t'.format(avg_result_cur_test))
        logger.info('avg_result_all_test:{}\t'.format(avg_result_all_test))




if __name__ == "__main__":
    main()
