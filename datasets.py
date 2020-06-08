from __future__ import print_function
import sys

import os
from PIL import Image
import gzip
import pickle
import urllib
import cv2
import codecs
import warnings
import gdown
from shutil import unpack_archive

from torch._six import int_classes as _int_classes
from torch.utils.data import Sampler
from torchvision import transforms as T
from torchvision.datasets.utils import download_url, makedir_exist_ok, download_and_extract_archive
import torch.nn.functional as F
import torch
from torch.utils import data
import numpy as np
from skimage.util import pad
import bisect

class SYNDIGITS(data.Dataset):
    """`
    Args:
        root (string): Root directory of dataset where directory
            ``SYNDIGIT`` exists.
        split (string): One of {'train', 'test'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    NB: I DO NOT own this dataset.
    This dataset is owned by the authors of the paper: "Unsupervised Domain Adaptation by Backpropagation, ICML'15"
    Please visit their project website: http://sites.skoltech.ru/compvision/projects/grl/ for more details.
    To view a copy of their license, visit http://creativecommons.org/licenses/by/4.0/.
    """

    filename = "synthetic_digits.tar.gz"
    url = "https://drive.google.com/uc?id=1g8rAvvIbl2T5SGD87gVyHEF-YLsO0vvW"

    split_list = {
        'train': ["synth_train_32x32.mat"],
        'test': ["synth_test_32x32.mat"]}

    def __init__(self, root, sample_mask=None, split='train', transform=None, download=False, domain_label=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split  # training set or test set
        self.domain_label = domain_label
        self.sample_mask = sample_mask

        if download:
            self.download()

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.filename = self.split_list[split][0]

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.split_list[self.split][0]))
        #print(loaded_mat.keys())

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()
        
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        if self.sample_mask is not None:
            # sub sample from entire dataset
            self.labels = self.labels[self.sample_mask]
            self.data = self.data[self.sample_mask]

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.split_list['train'][0]))
                and os.path.exists(os.path.join(self.root, self.split_list['test'][0])))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target_label, rot_label) where target is index of the target class.
        """
        img, target_label = self.data[index], int(self.labels[index])
        target_label = torch.tensor(target_label)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        domain_label = self.domain_label

        if self.transform is not None:
            img = self.transform(img)

        return img, target_label, domain_label

    def __len__(self):
        return len(self.data)

    def download(self):

        if self._check_exists():
            return

        makedir_exist_ok(self.root)

        print('Downloading Synthetic Digits...')
        gdown.download(self.url, os.path.join(self.root, self.filename), quiet=False)

        # untar the compressed files
        unpack_archive(os.path.join(self.root, self.filename), self.root)

class USPS(data.Dataset):

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    filename = "usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False, domain_label=0):
        """Init the USPS data set"""
        self.root = os.path.expanduser(root)
        self.train = train
        # Num of train = 7438, Num of test = 1860
        self.transform = transform
        self.dataset_size = None
        self.domain_label = domain_label

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset no found." + 
                " You can use download=True to download it.")

        self.train_data, self.train_labels = self.load_samples()

        self.train_data = self.train_data.transpose((0, 2, 3, 1)) # NCHW

    def __getitem__(self, index):
        """ Get images and target labels for data loader 
        
        Args:
            index (int): Index
        Returns:
            tuple (image, target): where target is the index of the class
        """

        img, label = self.train_data[index], self.train_labels[index]
        domain_label = self.domain_label

        if self.transform is not None:
            img = self.transform(img)

        label = torch.squeeze(torch.LongTensor([np.int64(label).item()]))

        return img, label, domain_label

    def __len__(self):
        """ Return the size of the dataset """
        return self.dataset_size

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("Done")
        return

    def load_samples(self):
        filename =os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()

        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels

class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]

    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    @property
    def train_labels(self):
        warnings.warn('train labels has been renamed targets')
        return self.targets

    @property
    def test_labels(self):
        warnings.warn('test labels has been renamed targets')
        return self.targets

    def __init__(self, root, sample_mask=None, train=True, transform=None, download=False, domain_label=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.domain_label = domain_label
        self.sample_mask = sample_mask

        if download:
            self.download()

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        if self.sample_mask is not None:
            self.data, self.targets = self.data[self.sample_mask], self.targets[self.sample_mask]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target_label, rot_label) where target_label is index of the target class
            and rot_label is the rotation index
        """
        img, target_label = self.data[index], self.targets[index]
        domain_label = self.domain_label

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target_label, domain_label

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @ property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file))
                and os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        "Download the MNIST data if it doesnt exist already in the processed folder"

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

class MNISTM(data.Dataset):
    """
    `MNIST-M Dataset.

    Adopted from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cogan/mnistm.py
    """

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'mnist_m_train.pt'
    test_file = 'mnist_m_test.pt'

    def __init__(self, root, mnist_root='mnist', sample_mask=None, train=True, transform=None, download=False, domain_label=0):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.join(os.path.dirname(self.root), mnist_root)
        self.transform = transform
        self.train = train  # training set or test set
        self.domain_label = domain_label
        self.sample_mask = sample_mask

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + 'You can use download=True to download it.')

        if self.train:
            self.train_data, self.train_labels = \
                torch.load(os.path.join(self.processed_folder,
                                        self.training_file))
            if self.sample_mask is not None:
                self.train_data, self.train_labels = self.train_data[self.sample_mask], self.train_labels[self.sample_mask]
        else:
            self.test_data, self.test_labels = \
                torch.load(os.path.join(self.processed_folder,
                                        self.test_file))
            if self.sample_mask is not None:
                self.test_data, self.test_labels = self.test_data[self.sample_mask], self.test_labels[self.sample_mask]

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode='RGB')
        domain_label = self.domain_label

        if self.transform is not None:
            img = self.transform(img)

        return img, target, domain_label

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file))
                and os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        "Download MNIST-M if it does not exists and put into processed folder"

        # import packages
        import gzip
        import pickle
        from torchvision import datasets

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.raw_folder, filename)
        download_and_extract_archive(self.url, download_root=self.raw_folder)

        # process and save as torch files
        print('Processing...')

        # load MNIST-M images from pkl file
        with open(file_path.replace('.gz', ''), 'rb') as f:
            mnistm_m_data = pickle.load(f, encoding='bytes')

        mnistm_m_train_data = torch.ByteTensor(mnistm_m_data[b"train"])
        mnistm_m_test_data = torch.ByteTensor(mnistm_m_data[b"test"])

        # get MNIST labels
        mnist_train_labels = MNIST(root=self.mnist_root, train=True, download=True).train_labels
        mnist_test_labels = MNIST(root=self.mnist_root, train=False, download=True).test_labels

        # save MNIST-M dataset
        training_set = (mnistm_m_train_data, mnist_train_labels)
        test_set = (mnistm_m_test_data, mnist_test_labels)
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


class SVHN(data.Dataset):
    """
    `SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`
    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root, sample_mask=None, split='train', transform=None, download=False, domain_label=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split  # training set or test set
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]
        self.domain_label = domain_label
        self.sample_mask = sample_mask

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        if download:
            self.download()

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        if self.sample_mask is not None:
            self.data, self.labels = self.data[self.sample_mask], self.labels[self.sample_mask]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target_label, rot_label) where target is index of the target class.
        """
        img, target_label = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        domain_label = self.domain_label
        target_label = torch.tensor(target_label)

        if self.transform is not None:
            img = self.transform(img)

        return img, target_label, domain_label

    def __len__(self):
        return len(self.data)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')}
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

class ConcatDataset_(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class WeightedRandomSampler(Sampler):
    """Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, weights, num_samples, replacement=True):
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples

class BatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class ConcatDataset(torch.utils.data.Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        #print([x for x in self.datasets[dataset_idx][sample_idx]])
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes
