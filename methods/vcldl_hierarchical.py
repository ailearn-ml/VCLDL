import torch.nn.functional as F
import numpy as np
from torch import nn
import os
import torch
from scipy.sparse import csr_matrix, save_npz, vstack
from xclib.evaluation.xc_metrics import precision, ndcg, recall, psprecision, psndcg, psrecall
from methods.vcldl import CorNet, kl_normal


class HierarchicalVariationalLDL(nn.Module):
    def __init__(self, n_labels):
        super(HierarchicalVariationalLDL, self).__init__()
        self.n_labels = n_labels
        self.encoder_x = CorNet(n_labels, n_cornet_blocks=1)
        self.encoder_y = CorNet(n_labels, n_cornet_blocks=1)
        self.mlp1 = CorNet(n_labels, n_cornet_blocks=1)
        self.linear11 = CorNet(n_labels, n_cornet_blocks=1)
        self.linear12 = CorNet(n_labels, n_cornet_blocks=1)
        self.mlp2 = CorNet(n_labels, n_cornet_blocks=1)
        self.linear21 = CorNet(n_labels, n_cornet_blocks=1)
        self.linear22 = CorNet(n_labels, n_cornet_blocks=1)
        self.mlp3 = CorNet(n_labels, n_cornet_blocks=1)
        self.linear31 = CorNet(n_labels, n_cornet_blocks=1)
        self.linear32 = CorNet(n_labels, n_cornet_blocks=1)
        self.decoder_x = CorNet(n_labels, n_cornet_blocks=1)
        self.decoder_y = CorNet(n_labels, n_cornet_blocks=1)
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_uniform_(param)

    def forward(self, x):
        return self.decoder_y(self.encoder_x(x))

    def vcldl_forward(self, x, y):
        x_encode = self.encoder_x(x)
        y_encode = self.encoder_y(y)
        d1 = self.mlp1(x_encode + y_encode)
        mu1 = self.linear11(d1)
        var1 = F.softplus(self.linear12(d1)) + 1e-8

        d2 = self.mlp2(d1)
        mu2 = self.linear21(d2)
        var2 = F.softplus(self.linear22(d2)) + 1e-8

        rand = torch.normal(mean=0., std=1., size=mu2.shape).cuda()
        z_down = mu2 + (var2 ** 0.5) * rand

        d3 = self.mlp3(z_down)
        mu3 = self.linear31(d3)
        var3 = F.softplus(self.linear32(d3)) + 1e-8

        mu0 = (mu1 * var3 + mu3 * var1) / (var1 + var3)
        var0 = ((var1 * var3) / (var1 + var3)) ** 2

        x_decode = self.decoder_x(z_down)
        y_decode = self.decoder_y(x_decode)
        return x_encode, mu0, var0, mu2, var2, mu3, var3, x_decode, y_decode


class HierarchicalVCLDL(nn.Module):
    def __init__(self, num_labels, basic_model, fixed=True,
                 alpha=0.1, lr=1e-3, weight_decay=1e-5, val_frequence=100):
        super(HierarchicalVCLDL, self).__init__()
        self.alpha = alpha
        self.fixed = fixed
        self.basic_model = basic_model
        self.HierarchicalVariationalLDL = HierarchicalVariationalLDL(num_labels)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.val_frequence = val_frequence
        self.global_step = 0
        self.max_5 = 0
        self.cuda()

    def set_forward(self, batch):
        x, y = batch
        x = self.basic_model.set_forward((x, y))
        out = self.HierarchicalVariationalLDL(x)
        return out

    def set_forward_loss(self, batch):
        x, y = batch
        y = torch.from_numpy(y.toarray()).float().cuda()
        if self.fixed:
            with torch.no_grad():
                x = self.basic_model.set_forward((x, y))
        else:
            x = self.basic_model.set_forward((x, y))

        x_encode, mu0, var0, mu2, var2, mu3, var3, x_decode, y_decode = self.HierarchicalVariationalLDL.vcldl_forward(x,
                                                                                                                      y)
        loss_cls = nn.BCEWithLogitsLoss()(self.HierarchicalVariationalLDL(x), y)
        loss_tgt = 0.5 * nn.MSELoss()(x_encode, x_decode)
        loss_dec = nn.BCEWithLogitsLoss()(y_decode, y)
        pm, pv = torch.zeros(mu2.shape).cuda(), torch.ones(var2.shape).cuda()
        loss_var = kl_normal(mu2, var2, pm, pv).mean() + kl_normal(mu0, var0, mu3, var3).mean()
        loss_sum = loss_tgt + loss_dec + loss_var
        return loss_cls + self.alpha * loss_sum

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

    def train_loop(self, epoch, train_loader, log=None, val_loader=None, val_path=None, inv_propesity=None):
        self.train()
        if not log:
            log = print
        self.epoch = epoch
        total_loss = 0
        for batch in train_loader:
            self.train()
            loss = self.set_forward_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.global_step += 1
            if (self.global_step % self.val_frequence == 0) and (val_loader is not None):
                assert val_path is not None
                if inv_propesity is None:
                    P = self.test_loop(val_loader, metrics=('ndcg',))
                    n1, n5 = P['ndcg'][0], P['ndcg'][4]
                    if n5 > self.max_5:
                        self.save(val_path, 'best_n5')
                        self.max_5 = n5
                        log('best n5, n1 = %.2f, n5 = %.2f' % (n1, n5))
                else:
                    P = self.test_loop(val_loader, metrics=('psprecision',), inv_propesity=inv_propesity)
                    psp1, psp5 = P['psprecision'][0], P['psprecision'][4]
                    if psp5 > self.max_5:
                        self.save(val_path, 'best_psp5')
                        self.max_5 = psp5
                        log('best psp5, psp1 = %.2f, psp5 = %.2f' % (psp1, psp5))
        torch.cuda.empty_cache()
        log(f'Epoch {epoch} done! Total loss: {total_loss}')

    def test_loop(self, test_loader, inv_propesity=None, metrics=('precision', 'ndcg', 'psprecision', 'psndcg'),
                  save_file=None, return_time=False, return_pred=False):
        # metrics = ('precision', 'ndcg', 'recall', 'psprecision', 'psndcg', 'psrecall')
        if ('psprecision' in metrics or 'psndcg' in metrics or 'psrecall' in metrics) and inv_propesity is None:
            raise ValueError('Missing inv propesity!')
        pred_data = []
        self.eval()
        total_precision, total_ndcg, total_recall = np.zeros(5), np.zeros(5), np.zeros(5)
        total_psprecision, total_psndcg, total_psrecall = np.zeros(5), np.zeros(5), np.zeros(5)
        with torch.no_grad():
            total_samples = 0
            total_time = 0
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
