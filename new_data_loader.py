import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class data_set(Dataset):

    def __init__(self, data,config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


    def collate_fn(self,batch):
        max_len = max([len(f["input_ids"]) for f in batch])
        max_sent = max([len(f["sent_pos"]) for f in batch])
        input_ids = [list(f["input_ids"]) + [0] * (max_len - len(f["input_ids"])) for f in batch]
        input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
        labels = [f["labels"] for f in batch]
        entity_pos = [f["entity_pos"] for f in batch]
        hts = [f["hts"] for f in batch]
        sent_pos = [f["sent_pos"] for f in batch]
        sent_labels = [f["sent_labels"] for f in batch if "sent_labels" in f]
        attns = [f["attns"] for f in batch if "attns" in f]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)

        # labels = [torch.tensor(label) for label in labels]
        # labels = torch.cat(labels, dim=0)

        if sent_labels != [] and None not in sent_labels:
            sent_labels_tensor = []
            for sent_label in sent_labels:
                sent_label = np.array(sent_label)
                sent_labels_tensor.append(np.pad(sent_label, ((0, 0), (0, max_sent - sent_label.shape[1]))))
            sent_labels_tensor = torch.from_numpy(np.concatenate(sent_labels_tensor, axis=0))
        else:
            sent_labels_tensor = None

        if attns:
            attns = [np.pad(attn, ((0, 0), (0, max_len - attn.shape[1]))) for attn in attns]
            attns = torch.from_numpy(np.concatenate(attns, axis=0))
        else:
            attns = None

        graph = [f["graph"] for f in batch]
        titles = [f["title"] for f in batch]
        # titles = []
        # for f in batch:
        #     for _ in f["labels"]:
        #         titles.append(f["title"])

        output = (input_ids, input_mask, labels, entity_pos, hts, sent_pos, sent_labels_tensor, attns, graph, titles)

        return output


def get_data_loader(config, data, shuffle = False, drop_last = False, batch_size = None):

    dataset = data_set(data, config)
    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader



class matrix_data_set(Dataset):

    def __init__(self, data,config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



    def collate_fn(self,batch):
        max_len = max([len(f["input_ids"]) for f in batch])
        max_sent = max([len(f["sent_pos"]) for f in batch])
        input_ids = [list(f["input_ids"]) + [0] * (max_len - len(f["input_ids"])) for f in batch]
        input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
        labels = [f["labels"] for f in batch]
        observe_matrices = [f["observe_matrix"] for f in batch]
        entity_pos = [f["entity_pos"] for f in batch]
        hts = [f["hts"] for f in batch]
        sent_pos = [f["sent_pos"] for f in batch]
        sent_labels = [f["sent_labels"] for f in batch if "sent_labels" in f]
        attns = [f["attns"] for f in batch if "attns" in f]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)

        # labels = [torch.tensor(label) for label in labels]
        # labels = torch.cat(labels, dim=0)

        if sent_labels != [] and None not in sent_labels:
            sent_labels_tensor = []
            for sent_label in sent_labels:
                sent_label = np.array(sent_label)
                sent_labels_tensor.append(np.pad(sent_label, ((0, 0), (0, max_sent - sent_label.shape[1]))))
            sent_labels_tensor = torch.from_numpy(np.concatenate(sent_labels_tensor, axis=0))
        else:
            sent_labels_tensor = None

        if attns:
            attns = [np.pad(attn, ((0, 0), (0, max_len - attn.shape[1]))) for attn in attns]
            attns = torch.from_numpy(np.concatenate(attns, axis=0))
        else:
            attns = None

        graph = [f["graph"] for f in batch]
        titles = [f["title"] for f in batch]


        output = (input_ids, input_mask, labels, entity_pos, hts, sent_pos, sent_labels_tensor, attns, graph, titles, observe_matrices)

        return output


def matrix_get_data_loader(config, data, shuffle=False, drop_last = False, batch_size = None):

    dataset = matrix_data_set(data, config)
    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader