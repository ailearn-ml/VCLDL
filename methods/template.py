# -*- coding: utf-8 -*-
from torch import nn
import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, vstack
from abc import abstractmethod
from xclib.evaluation.xc_metrics import precision, ndcg, recall, psprecision, psndcg, psrecall
from collections import deque


class DeepXMLTemplate(nn.Module):
    def __init__(self, vocabulary_dims, num_labels, embedding_dims=300, emb_train=True, max_epoch=30,
                 gradient_clip_value=5.0, use_swa=False, swa_warmup=10, val_frequence=100, adjust_lr=False,
                 verbose=False):
        super(DeepXMLTemplate, self).__init__()
        self.vocabulary_dims = vocabulary_dims
        self.num_labels = num_labels
        if embedding_dims is None:
            self.embedding = None
        else:
            self.embedding_dims = embedding_dims
            self.embedding = nn.Linear(vocabulary_dims, embedding_dims)
            self.embedding.requires_grad_(emb_train)
        self.adjust_lr = adjust_lr
        self.max_epoch = max_epoch
        self.verbose = verbose
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)
        self.use_swa = use_swa
        self.global_step = 0
        self.val_frequence = val_frequence
        if self.use_swa:
            self.swa_warmup = swa_warmup
            self.state = {}
        self.max_5 = 0

    @abstractmethod
    def set_forward(self, batch):
        # x -> predict score
        pass

    @abstractmethod
    def set_forward_loss(self, batch):
        # x , y -> loss function
        pass

    def emb_init(self, embedding):
        self.embedding.weight.data = torch.from_numpy(embedding).transpose(0, 1).float().cuda()

    def train_loop(self, epoch, train_loader, log=None, val_loader=None, val_path=None, inv_propesity=None):
        self.train()
        if not log:
            log = print
        self.epoch = epoch
        self.adjust_learning_rate()
        if self.verbose: train_loader = tqdm(train_loader)
        total_loss = 0
        if self.use_swa and epoch == self.swa_warmup:
            self.swa_init()
        for batch in train_loader:
            self.train()
            loss = self.set_forward_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.clip_gradient()
            self.optimizer.step()
            total_loss += loss.item()
            self.global_step += 1
            if (self.global_step % self.val_frequence == 0) and (val_loader is not None):
                assert val_path is not None
                if self.use_swa:
                    self.swa_step()
                    self.swa_swap_params()
                if inv_propesity is None:
                    P = self.test_loop(val_loader, metrics=('ndcg',), verbose=False)
                    n1, n5 = P['ndcg'][0], P['ndcg'][4]
                    if n5 > self.max_5:
                        self.save(val_path, 'best_n5')
                        self.max_5 = n5
                        if not self.verbose:
                            log('best n5, n1 = %.2f, n5 = %.2f' % (n1, n5))
                else:
                    P = self.test_loop(val_loader, metrics=('psprecision',), inv_propesity=inv_propesity, verbose=False)
                    psp1, psp5 = P['psprecision'][0], P['psprecision'][4]
                    if psp5 > self.max_5:
                        self.save(val_path, 'best_psp5')
                        self.max_5 = psp5
                        if not self.verbose:
                            log('best psp5, psp1 = %.2f, psp5 = %.2f' % (psp1, psp5))
                if self.use_swa:
                    self.swa_swap_params()
        torch.cuda.empty_cache()
        log(f'Epoch {epoch} done! Total loss: {total_loss}')

    def test_loop(self, test_loader, inv_propesity=None, metrics=('precision', 'ndcg', 'psprecision', 'psndcg'),
                  save_file=None, return_time=False, return_pred=False, verbose=None):
        if ('psprecision' in metrics or 'psndcg' in metrics or 'psrecall' in metrics) and inv_propesity is None:
            raise ValueError('Missing inv propesity!')
        pred_data = []
        if verbose is None: verbose = self.verbose
        self.eval()
        total_precision, total_ndcg, total_recall = np.zeros(5), np.zeros(5), np.zeros(5)
        total_psprecision, total_psndcg, total_psrecall = np.zeros(5), np.zeros(5), np.zeros(5)
        with torch.no_grad():
            total_samples = 0
            total_time = 0
            if verbose: test_loader = tqdm(test_loader)
            for batch in test_loader:
                y_pred = self.set_forward(batch)
                if type(y_pred) == tuple:
                    y_pred, t = y_pred
                    total_time = total_time + t
                if type(y_pred) == torch.Tensor:
                    y_pred = y_pred.detach().cpu().numpy()
                if type(y_pred) != csr_matrix:
                    y_pred = csr_matrix(y_pred)
                y_true = batch[-1]
                if type(y_true) == torch.Tensor:
                    y_true = y_true.numpy()
                y_true = csr_matrix(y_true)
                if 'precision' in metrics:
                    total_precision = total_precision + precision(y_pred, y_true) * y_pred.shape[0]
                if 'ndcg' in metrics:
                    total_ndcg = total_ndcg + ndcg(y_pred, y_true) * y_pred.shape[0]
                if 'recall' in metrics:
                    total_recall = total_recall + recall(y_pred, y_true) * y_pred.shape[0]
                if 'psprecision' in metrics:
                    total_psprecision = total_psprecision + psprecision(y_pred, y_true, inv_propesity) * y_pred.shape[0]
                if 'psndcg' in metrics:
                    total_psndcg = total_psndcg + psndcg(y_pred, y_true, inv_propesity) * y_pred.shape[0]
                if 'psrecall' in metrics:
                    total_psrecall = total_psrecall + psrecall(y_pred, y_true, inv_propesity) * y_pred.shape[0]
                pred_data.append(y_pred)
                total_samples += y_pred.shape[0]
        torch.cuda.empty_cache()
        pred_data = vstack(pred_data)
        if save_file:
            save_npz(save_file, pred_data)
        total_precision, total_psprecision = total_precision / total_samples, total_psprecision / total_samples
        total_ndcg, total_psndcg = total_ndcg / total_samples, total_psndcg / total_samples
        total_recall, total_psrecall = total_recall / total_samples, total_psrecall / total_samples
        out_score = {}
        for metric in metrics:
            exec(f"out_score['{metric}'] = total_{metric} * 100")

        if return_time:
            avg_time = total_time / total_samples * 1000
            return out_score, avg_time
        else:
            if return_pred:
                return out_score, pred_data
            else:
                return out_score

    def save(self, path, epoch=None, save_optimizer=False):
        os.makedirs(path, exist_ok=True)
        if type(epoch) is str:
            save_path = os.path.join(path, '%s.tar' % epoch)
        elif epoch is None:
            save_path = os.path.join(path, 'model.tar')
        else:
            save_path = os.path.join(path, '%d.tar' % epoch)
        while True:
            try:
                if not save_optimizer:
                    torch.save({'model': self.state_dict(), }, save_path)
                else:
                    torch.save({'model': self.state_dict(),
                                'optimizer': self.optimizer.state_dict(), }, save_path)
                return
            except:
                pass

    def load(self, path, epoch=None, load_optimizer=False):
        if type(epoch) is str:
            load_path = os.path.join(path, '%s.tar' % epoch)
        else:
            if epoch is None:
                files = os.listdir(path)
                files = np.array(list(map(lambda x: int(x.replace('.tar', '')), files)))
                epoch = np.max(files)
            load_path = os.path.join(path, '%d.tar' % epoch)
        tmp = torch.load(load_path)
        self.load_state_dict(tmp['model'])
        if load_optimizer:
            self.optimizer.load_state_dict(tmp['optimizer'])

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))

    def swa_init(self):
        if 'swa' not in self.state:
            print('swa Initializing')
            self.state['swa'] = {'models_num': 1}
            for n, p in self.named_parameters():
                self.state['swa'][n] = p.data.clone().detach()

    def swa_step(self):
        if 'swa' in self.state:
            self.state['swa']['models_num'] += 1
            beta = 1.0 / self.state['swa']['models_num']
            with torch.no_grad():
                for n, p in self.named_parameters():
                    self.state['swa'][n].mul_(1.0 - beta).add_(p.data, alpha=beta)

    def swa_swap_params(self):
        if 'swa' in self.state:
            for n, p in self.named_parameters():
                p.data, self.state['swa'][n] = self.state['swa'][n], p.data

    def disable_swa(self):
        if 'swa' in self.state:
            del self.state['swa']

    def adjust_learning_rate(self):
        epoch = self.epoch + 1
        if self.adjust_lr:
            if epoch <= 5:
                self.lr_now = self.lr * epoch / 5
            elif epoch >= int(self.max_epoch * 0.8):
                self.lr_now = self.lr * 0.01
            elif epoch > int(self.max_epoch * 0.6):
                self.lr_now = self.lr * 0.1
            else:
                self.lr_now = self.lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr_now
        else:
            self.lr_now = self.lr
