import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import json
from utils.datasets import SparseTensorDataset, collate_fn_for_sparse_data_loader
import numpy as np
from torch.utils.data.dataloader import DataLoader
from utils.utils import get_inv_propesity, Logger
from utils.utils import set_seed
from scipy.sparse import load_npz
from utils.metrics import evaluation
from methods.xmlcnn import XMLCNN
from methods.vcldl import VCLDL


def get_model(params, vocabulary_dims, num_labels, embedding_dims, emb_init):
    model = XMLCNN(vocabulary_dims=vocabulary_dims,
                   num_labels=num_labels,
                   embedding_dims=embedding_dims,
                   lr=1e-3,
                   dynamic_pool_length=params['model_params']['dynamic_pool_length'],
                   bottleneck_dim=params['model_params']['bottleneck_dim'],
                   num_filters=params['model_params']['num_filters'],
                   dropout=params['model_params']['dropout'],
                   emb_init=emb_init,
                   emb_train=params['model_params']['emb_trainable'],
                   padding_idx=0,
                   use_swa=True,
                   swa_warmup=params['swa_warmup'],
                   verbose=False)
    return model


def main_train():
    print('-----Training-----')
    basic_model = get_model(params=params,
                            vocabulary_dims=embedding.shape[0],
                            num_labels=y_train.shape[1],
                            embedding_dims=embedding.shape[1],
                            emb_init=embedding,
                            )
    basic_model.load(basic_train_dir, epoch='best_psp5')
    model = VCLDL(num_labels=y_train.shape[1],
                  fixed=fixed,
                  basic_model=basic_model,
                  alpha=alpha,
                  lr=lr)
    train_loader = DataLoader(dataset=SparseTensorDataset(x_train, y_train),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn_for_sparse_data_loader)
    val_loader = DataLoader(dataset=SparseTensorDataset(x_val, y_val),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn_for_sparse_data_loader)
    for epoch in range(max_epoch):
        model.train_loop(epoch, train_loader, log=print,
                         val_loader=val_loader, val_path=train_dir, inv_propesity=inv_propesity)
    print('-----------------------------------------------------')
    model.save(train_dir, epoch=max_epoch - 1)


def main_test():
    print('-----Testing-----')
    basic_model = get_model(params=params,
                            vocabulary_dims=embedding.shape[0],
                            num_labels=y_train.shape[1],
                            embedding_dims=embedding.shape[1],
                            emb_init=embedding,
                            )
    model = VCLDL(num_labels=y_train.shape[1],
                  basic_model=basic_model,
                  alpha=alpha)
    model.load(train_dir, epoch='best_psp5')
    test_loader = DataLoader(dataset=SparseTensorDataset(x_test, y_test),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             collate_fn=collate_fn_for_sparse_data_loader)
    P, pred = model.test_loop(test_loader,  # save_file = f'{test_dir}/result',
                              return_time=False, inv_propesity=inv_propesity,
                              metrics=('psprecision', 'psndcg', 'psrecall', 'precision', 'ndcg', 'recall'),
                              return_pred=True)
    for key in P.keys():
        print('%s@1: %.2f\t%s@3: %.2f\t%s@5: %.2f' % (key[:3], P[key][0], key[:3], P[key][2], key[:3], P[key][4]))
    result = evaluation(y_test, pred)
    for key in result.keys():
        print(f'{key}: {result[key]}')


seed = 0
set_seed(seed)
basic_method_name = 'xmlcnn'
method_name = 'vcldl-xmlcnn'
dataset_name = 'Reuters-21578'
alpha = 0.01
lr = 1e-5
fixed = True


params = json.load(open(
    os.path.join(os.getcwd(), './configs', method_name.replace('vcldl-', ''),
                 dataset_name + '.json')))
max_epoch = 10
num_workers = 8
batch_size = params['batch_size']

basic_train_dir = os.getcwd() + f'/save/{basic_method_name}/{dataset_name}/train/{seed}'
train_dir = os.getcwd() + f'/save/{method_name}/{dataset_name}/train/{alpha}_{lr}_{fixed}_{seed}'
test_dir = os.getcwd() + f'/save/{method_name}/{dataset_name}/test/{alpha}_{lr}_{fixed}_{seed}'
log_dir = os.getcwd() + f'/save/{method_name}/{dataset_name}/log/{alpha}_{lr}_{fixed}_{seed}'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

dataset_path = os.getcwd() + f'/../data/MLTC/{dataset_name}'
x_train = load_npz(f'{dataset_path}/x_train.npz')[:, :500]
x_val = load_npz(f'{dataset_path}/x_val.npz')[:, :500]
x_test = load_npz(f'{dataset_path}/x_test.npz')[:, :500]
y_train = load_npz(f'{dataset_path}/y_train.npz')
y_val = load_npz(f'{dataset_path}/y_val.npz')
y_test = load_npz(f'{dataset_path}/y_test.npz')
embedding = np.load(f'{dataset_path}/emb_init.npy')

idx = np.arange(x_train.shape[0])[np.array((x_train.sum(1) != 0)).squeeze()]
x_train = x_train[idx]
y_train = y_train[idx]

inv_propesity = get_inv_propesity(dataset_name, y_train, file=f'{test_dir}/inv_propesity.npy')

print = Logger(f'{log_dir}/log.txt').logger.warning
print(f'{dataset_name}, {method_name}, {seed}')

if not os.path.exists(os.path.join(train_dir, f'{max_epoch - 1}.tar')):
    main_train()
main_test()
