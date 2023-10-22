# This code is based on https://github.com/XunGuangxu/CorNet
# Deep Learning for Extreme Multi-label Text Classification. SIGIR 2017: 115-124

from torch import nn
from methods.template import DeepXMLTemplate
import torch
from utils.networks import DenseSparseAdam


class XMLCNN(DeepXMLTemplate):
    def __init__(self, vocabulary_dims, num_labels, embedding_dims=300, lr=1e-3,max_epoch=30, dynamic_pool_length=8,
                 bottleneck_dim=512, num_filters=128, dropout=0.5, emb_init=None, emb_train=True, padding_idx=0,
                 adjust_lr=False, use_swa=True, swa_warmup=2, verbose=False):
        super(XMLCNN, self).__init__(vocabulary_dims=vocabulary_dims,
                                     num_labels=num_labels,
                                     embedding_dims=None,
                                     emb_train=emb_train,
                                     adjust_lr=adjust_lr,
                                     max_epoch=max_epoch,
                                     use_swa=use_swa,
                                     swa_warmup=swa_warmup,
                                     verbose=verbose)
        self.embedding = nn.Embedding(vocabulary_dims, embedding_dims, padding_idx=padding_idx, sparse=True,
                                      _weight=torch.from_numpy(emb_init).float())
        self.embedding.requires_grad_(emb_train)
        self.padding_idx = padding_idx
        self.ks = 3  # There are three conv nets here
        ## Different filter sizes in xml_cnn than kim_cnn
        self.conv1 = nn.Conv2d(1, num_filters, (2, embedding_dims), padding=(1, 0))
        self.conv2 = nn.Conv2d(1, num_filters, (4, embedding_dims), padding=(3, 0))
        self.conv3 = nn.Conv2d(1, num_filters, (8, embedding_dims), padding=(7, 0))
        self.pool = nn.AdaptiveMaxPool1d(dynamic_pool_length)  # Adaptive pooling
        self.bottleneck = nn.Linear(self.ks * num_filters * dynamic_pool_length, bottleneck_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(bottleneck_dim, num_labels)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.bottleneck.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.num_filters = num_filters
        self.dynamic_pool_length = dynamic_pool_length
        self.optimizer = DenseSparseAdam(self.parameters(), lr=lr, weight_decay=1e-3)
        self.lr = lr
        self.lr_now = lr
        self.cuda()

    def set_forward(self, batch):
        x, _ = batch
        x = torch.from_numpy(x.toarray()).long().cuda()
        emb_out = self.embedding(x)  # (batch, sent_len, embed_dim)
        x = emb_out.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        x = [torch.relu(self.conv1(x)).squeeze(3), torch.relu(self.conv2(x)).squeeze(3),
             torch.relu(self.conv3(x)).squeeze(3)]
        x = [self.pool(i).squeeze(2) for i in x]
        # (batch, channel_output) * ks
        x = torch.cat(x, 1)  # (batch, channel_output * ks)
        x = torch.relu(self.bottleneck(x.view(-1, self.ks * self.num_filters * self.dynamic_pool_length)))
        x = self.dropout(x)
        out = self.fc1(x)  # (batch, target_size)
        return out

    def set_forward_loss(self, batch):
        x, y = batch
        y_pred = self.set_forward(batch)
        y = torch.from_numpy(y.toarray()).float().cuda()
        return nn.BCEWithLogitsLoss()(y_pred, y)

    def get_embedding(self, batch):
        x, _ = batch
        x = torch.from_numpy(x.toarray()).long().cuda()
        emb_out = self.embedding(x)  # (batch, sent_len, embed_dim)
        x = emb_out.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        x = [torch.relu(self.conv1(x)).squeeze(3), torch.relu(self.conv2(x)).squeeze(3),
             torch.relu(self.conv3(x)).squeeze(3)]
        x = [self.pool(i).squeeze(2) for i in x]
        # (batch, channel_output) * ks
        x = torch.cat(x, 1)  # (batch, channel_output * ks)
        x = self.bottleneck(x.view(-1, self.ks * self.num_filters * self.dynamic_pool_length))
        return x
