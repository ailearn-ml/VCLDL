import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from xclib.evaluation.xc_metrics import precision, ndcg, recall, psprecision, psndcg, psrecall


def evaluation_xml(y_true, y_pred, inv_propesity):
    out_score = {}
    out_score['psprecision'] = psprecision(y_pred, y_true, inv_propesity) * 100
    out_score['psndcg'] = psndcg(y_pred, y_true, inv_propesity) * 100
    out_score['psrecall'] = psrecall(y_pred, y_true, inv_propesity) * 100
    out_score['precision'] = precision(y_pred, y_true) * 100
    out_score['ndcg'] = ndcg(y_pred, y_true) * 100
    out_score['recall'] = recall(y_pred, y_true) * 100
    return out_score


def evaluation(y_true, y_prob, y_pred=None):
    if type(y_prob) == csr_matrix:
        y_prob = y_prob.toarray()
    if type(y_prob) == torch.Tensor:
        y_prob = y_prob.detach().cpu().numpy()
    if type(y_true) == csr_matrix:
        y_true = y_true.toarray()
    if type(y_true) == torch.Tensor:
        y_true = y_true.detach().cpu().numpy()

    if y_pred is None:
        if np.min(y_prob) < 0:
            y_pred = np.zeros_like(y_prob)
            y_pred[y_prob > 0] = 1
        else:
            y_pred = np.zeros_like(y_prob)
            y_pred[y_prob > 0.5] = 1

    CF1 = f1_score(y_true, y_pred, average='macro')
    OF1 = f1_score(y_true, y_pred, average='micro')

    return {'MacroF1': CF1, 'MicroF1': OF1}