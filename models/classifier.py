import torch.nn.functional as F
import torch
from torch import nn

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.bias, 0.1)

    def forward(self, input_tensor):
        output = torch.matmul(input_tensor,self.weights.t()) + self.bias
        return output


class Classifier(nn.Module):
    def __init__(self, emb_size=768, block_size=64, num_class=-1):
        super().__init__()
        self.emb_size = emb_size
        self.block_size = block_size
        self.bilinear = nn.Linear(emb_size * block_size, num_class)

    def forward(self, head_embeddings, tail_embeddings):
        b1 = head_embeddings.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = tail_embeddings.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)

        logits = self.bilinear(bl)
        return logits


class Linear_Classifier(nn.Module):
    def __init__(self, emb_size=768, block_size=64, num_class=-1):
        super().__init__()
        self.emb_size = emb_size
        self.block_size = block_size
        self.bilinear = LinearLayer(emb_size * block_size, num_class)

    def forward(self, head_embeddings, tail_embeddings):
        b1 = head_embeddings.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = tail_embeddings.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)

        logits = self.bilinear(bl)
        return logits


class Proto_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.distance_metric = config.distance_metric

    def forward(self, reps, prototypes):
        reps = F.normalize(reps, p=2, dim=1)

        scores = []

        for proto in prototypes:
            proto = F.normalize(proto.to(self.config.device), p=2, dim=1)
            if self.distance_metric == "dot_product":
                class_scores = reps.unsqueeze(0)*proto.unsqueeze(1)
                class_scores = torch.sum(class_scores, dim=-1)
            else:
                class_scores = torch.pow(reps.unsqueeze(0) - proto.unsqueeze(1), 2)
                class_scores = -torch.sum(class_scores, dim=-1)
            class_scores = class_scores.max(dim=0, keepdim=False)[0]
            scores.append(class_scores)
        scores = torch.stack(scores).transpose(0, 1)

        return scores