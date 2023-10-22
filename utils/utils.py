import numpy as np
import torch
import random
from scipy import sparse
import os
import logging
from logging import handlers
from xclib.evaluation.xc_metrics import compute_inv_propesity
import six
import torch.nn.functional as F
from xclib.evaluation.xc_metrics import precision, ndcg, recall, psprecision, psndcg, psrecall
import warnings

warnings.filterwarnings("ignore")

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # multi-gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


def load_data(dataset_path):
    x_train = sparse.load_npz(os.path.join(dataset_path, 'x_train.npz'))
    y_train = sparse.load_npz(os.path.join(dataset_path, 'y_train.npz'))
    x_test = sparse.load_npz(os.path.join(dataset_path, 'x_test.npz'))
    y_test = sparse.load_npz(os.path.join(dataset_path, 'y_test.npz'))
    return x_train, y_train, x_test, y_test


def load_embedding(dataset_path, n_features=40000, embedding_dims=300):
    if os.path.exists(f'{dataset_path}/fasttextB_embeddings_512d.npy'):
        embedding = np.load(f'{dataset_path}/fasttextB_embeddings_512d.npy')
    elif os.path.exists(f'{dataset_path}/fasttextB_embeddings_300d.npy'):
        embedding = np.load(f'{dataset_path}/fasttextB_embeddings_300d.npy')
    else:
        print('Embedding file not found! Using random embedding!')
        embedding = torch.from_numpy(np.zeros((n_features, embedding_dims)))
        torch.nn.init.kaiming_uniform_(embedding)
        embedding = embedding.numpy()
    return embedding


# numpy.ndarray/scipy sparse matrix to pytorch sparse matrix
def sparse_tensor(tensor):
    '''
    numpy/scipy sparse matrix to pytorch sparse matrix
    :param tensor:  numpy.ndarray or scipy.sparse.matrix
    :return: pytorch coo_matrix
    '''
    if type(tensor) != sparse.coo_matrix:
        tensor = sparse.coo_matrix(tensor)
    indices = np.vstack((tensor.row, tensor.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(tensor.data)
    tensor = torch.sparse.FloatTensor(i, v, torch.Size(tensor.shape))
    return tensor


def get_inv_propesity(dataset_name, y_train, file=None, retrain=False):
    if file is not None and (os.path.exists(file) or os.path.exists(file + '.npy')) and not retrain:
        try:
            return np.load(file)
        except:
            return np.load(file + '.npy')
    # y_train:[N,c], sparse matrix
    if 'Wikipedia' in dataset_name or 'WikiTitles' in dataset_name:
        A, B = 0.5, 0.4
    elif 'Amazon' in dataset_name:
        A, B = 0.6, 2.6
    else:
        A, B = 0.55, 1.5
    print(f'dataset_name: {dataset_name}, inv_propesity: A-{A}, B-{B}')
    inv_propesity = compute_inv_propesity(y_train, A, B)
    if file is not None:
        np.save(file, inv_propesity)
    return inv_propesity


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


def csr_from_arrays(indices, values, shape=None, dtype='float32'):
    """
    Convert given indices and their corresponding values
    to a csr_matrix

    indices[i, j] => value[i, j]

    Arguments:
    indices: np.ndarray
        array with indices; type should be int
    values: np.ndarray
        array with values
    shape: tuple, optional, default=None
        Infer shape from indices when None
        * Throws error in case of invalid shape
        * Rows in indices and vals must match the given shape
        * Cols in indices must be less than equal to given shape
    """
    assert indices.shape == values.shape, "Shapes for ind and vals must match"
    num_rows, num_cols = indices.shape[0], np.max(indices) + 1
    if shape is not None:
        assert num_rows == shape[0], "num_rows_inferred != num_rows_given"
        assert num_cols <= shape[1], "num_cols_inferred > num_cols_given"
    else:
        shape = (num_rows, num_cols)
    # Need the last values (hence +1)
    indptr = np.arange(0, indices.size + 1, indices.shape[1])
    data = values.flatten()
    indices = indices.flatten()
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the beginning or the end
    if padding='post.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def to_categorical(label, num_classes=None):
    '''
    label to one-hot label vector
    Args:
        label: np.ndarray, label
        num_classes: total number of classes
    Returns: one-hot label vector
    '''
    label = np.array(label, dtype='int')
    if num_classes is not None:
        assert num_classes > label.max()
    else:
        num_classes = label.max() + 1
    if len(label.shape) == 1:
        y = np.eye(num_classes, dtype='int64')[label]
        return y
    elif len(label.shape) == 2 and label.shape[1] == 1:
        y = np.eye(num_classes, dtype='int64')[label.squeeze()]
        return y
    else:
        raise ValueError('Dimension Error!')


def report_result(y_pred, y_true, inv_propesity):
    P = {}
    P['precision'] = precision(y_pred, y_true) * 100
    P['ndcg'] = ndcg(y_pred, y_true) * 100
    P['recall'] = recall(y_pred, y_true) * 100
    P['psprecision'] = psprecision(y_pred, y_true, inv_propesity) * 100
    P['psndcg'] = psndcg(y_pred, y_true, inv_propesity) * 100
    P['psrecall'] = psrecall(y_pred, y_true, inv_propesity) * 100
    for key in P.keys():
        print('%s@1: %.2f\t%s@3: %.2f\t%s@5: %.2f' % (key[:3], P[key][0], key[:3], P[key][2], key[:3], P[key][4]))

def cosine_similarity(x, y):
    '''
    Cosine Similarity of two tensors
    Args:
        x: torch.Tensor, m x d
        y: torch.Tensor, n x d
    Returns:
        result, m x n
    '''
    assert x.size(1) == y.size(1)
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return x @ y.transpose(0, 1)


def mahalanobis_dist(x, y):
    # x: m x d
    # y: n x d
    # return: m x n
    assert x.size(1) == y.size(1)
    cov = torch.cov(x)  # [m,m]
    x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
    y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
    delta = x - y  # [m,n,d]
    return torch.einsum('abc,abc->ab', torch.einsum('abc,ad->abc', delta, torch.inverse(cov)), delta)

def euclidean_dist(x, y):
    # x: m x d
    # y: n x d
    # return: m x n
    assert x.size(1) == y.size(1)
    x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
    y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
    return torch.pow(x - y, 2).sum(2)