from torch import nn
from transformers import T5EncoderModel, T5Tokenizer
import torch
import clip
from utils.class_id_map import get_classes
from utils.collect_text import collect_txt
from utils.config import Config


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.device = config.device
        model_, preprocess = clip.load("ViT-B/32", self.device)
        self.model = model_.eval().float()
        self.text = collect_txt(config.dataset)
        self.frozen()
        # generate static part
        with torch.no_grad():
            part_text_list = []
            for part_id in self.text.keys():
                part_text_list.extend([i.strip() for i in self.text[part_id]])
            part_text_token = clip.tokenize(part_text_list).to(self.device)
            part_feature = self.model.encode_text(part_text_token)
            self.fix = part_feature.contiguous().view(5, config.n_classes, -1)
        self.soa = SemanticOffsetAdaptor(config.n_contra_features)
        # froze the model

    def forward(self, part_id, class_ids):
        part_feature_list = []
        for class_id in class_ids:
            part_feature_list.append(self.fix[part_id][class_id].unsqueeze(0))
        part_feature = torch.cat(part_feature_list).to(self.device)
        part_feature = self.soa(part_feature)
        return part_feature

    def frozen(self):
        for param in self.model.parameters():
            param.requires_grad = False

class SemanticOffsetAdaptor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(embed_dim, embed_dim))
    def forward(self, x):
        return x + self.ff(x)