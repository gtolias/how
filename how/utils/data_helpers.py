"""Data manipulation helpers"""

import os.path
import pickle

from cirtorch.datasets.datahelpers import cid2filename
from cirtorch.datasets.testdataset import configdataset


def load_dataset(dataset, data_root=''):
    """Return tuple (image list, query list, bounding boxes, gnd dictionary)"""

    if isinstance(dataset, dict):
        root = os.path.join(data_root, dataset['image_root'])
        images, qimages = None, None
        if dataset['database_list'] is not None:
            images = [path_join(root, x.strip("\n")) for x in open(dataset['database_list']).readlines()]
        if dataset['query_list'] is not None:
            qimages = [path_join(root, x.strip("\n")) for x in open(dataset['query_list']).readlines()]
        bbxs = None
        gnd = None

    elif dataset == 'train':
        training_set = 'retrieval-SfM-120k'
        db_root = os.path.join(data_root, 'train', training_set)
        ims_root = os.path.join(db_root, 'ims')
        db_fn = os.path.join(db_root, '{}.pkl'.format(training_set))
        with open(db_fn, 'rb') as f:
            db = pickle.load(f)['train']
        images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
        qimages = []
        bbxs = None
        gnd = None

    elif dataset == 'val_eccv20':
        db_root = os.path.join(data_root, 'train', 'retrieval-SfM-120k')
        fn_val_proper = db_root+'/retrieval-SfM-120k-val-eccv2020.pkl' # pos are all with #inl >=3 & <= 10
        with open(fn_val_proper, 'rb') as f:
            db = pickle.load(f)
        ims_root = os.path.join(db_root, 'ims')
        images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
        gnd = db['gnd']
        qidx = db['qidx']
        qimages = [images[x] for x in qidx]
        bbxs = None

    elif "/" in dataset:
        with open(dataset, 'rb') as handle:
            db = pickle.load(handle)
        images, qimages, bbxs, gnd = db['imlist'], db['qimlist'], None, db['gnd']

    else:
        cfg = configdataset(dataset, os.path.join(data_root, 'test'))
        images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
        qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
        if 'bbx' in cfg['gnd'][0].keys():
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        else:
            bbxs = None
        gnd = cfg['gnd']

    return images, qimages, bbxs, gnd


def path_join(root, name):
    """Perform os.path.join by default; if asterisk is present in root, substitute with the name.

    >>> path_join('/data/img_*.jpg', '001')
    '/data/img_001.jpg'
    """
    if "*" in root.rsplit("/", 1)[-1]:
        return root.replace("*", name)
    return os.path.join(root, name)


class AverageMeter:
    """Compute and store the average and last value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the counter by a new value"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
