import collections
import os
import numpy as np


def load_data(args):
    train_data, eval_data, test_data, user_history_dict = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    ripple_set = get_ripple_set(args, kg, user_history_dict)
    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = './data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    # n_user = len(set(rating_np[:, 0]))
    # n_item = len(set(rating_np[:, 1]))
    return dataset_split(rating_np)


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data, user_history_dict


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = './data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def get_ripple_set(args, kg, user_history_dict):
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set


from torch.utils import data
import torch

class CustomDataset(data.Dataset):
    def __init__(self, args, data,ripple_set):
        super(CustomDataset, self).__init__()
        self.args = args
        self.data = data
        self.ripple_set = ripple_set
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return index


class _CustomDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def get_feed_dict(self, args, data, ripple_set, start, end):
        items = torch.LongTensor(data[start:end, 1])
        labels = torch.LongTensor(data[start:end, 2])
        memories_h, memories_r, memories_t = [], [], []
        for i in range(args.n_hop):
            memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[start:end, 0]]))
            memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[start:end, 0]]))
            memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[start:end, 0]]))
        if args.use_cuda:
            items = items.cuda()
            labels = labels.cuda()
            memories_h = list(map(lambda x: x.cuda(), memories_h))
            memories_r = list(map(lambda x: x.cuda(), memories_r))
            memories_t = list(map(lambda x: x.cuda(), memories_t))
        return items, labels, memories_h, memories_r,memories_t
    
    def fetch(self, possible_batched_index):
        start = possible_batched_index[0]
        end = possible_batched_index[-1]
        
        
    
        return self.get_feed_dict(self.dataset.args, self.dataset.data, self.dataset.ripple_set, start, end)


from typing import Any, Callable, TypeVar, Generic, Sequence, List, Optional

class _BaseDataLoaderIter(object):
    def __init__(self, loader: data.DataLoader) -> None:
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._pin_memory = loader.pin_memory and torch.cuda.is_available()
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = "enumerate(DataLoader)#{}.__next__".format(self.__class__.__name__)

    def __iter__(self) -> '_BaseDataLoaderIter':
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                self._reset()
            data = self._next_data()
            self._num_yielded += 1
            # if self._dataset_kind == _CustomDatasetKind.Iterable and \
            #         self._IterableDataset_len_called is not None and \
            #         self._num_yielded > self._IterableDataset_len_called:
            #     warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
            #                 "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
            #                                                       self._num_yielded)
            #     if self._num_workers > 0:
            #         warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
            #                      "IterableDataset replica at each worker. Please see "
            #                      "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
            #     warnings.warn(warn_msg)
            return data

    next = __next__  # Python 2 compatibility

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)

from torch.utils.data import _utils

class _CustomDatasetKind(object):
    @staticmethod
    def create_fetcher(dataset, auto_collation, collate_fn, drop_list):
        return _CustomDatasetFetcher(dataset, auto_collation, collate_fn, drop_list)

class CustomDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(CustomDataLoaderIter, self).__init__(loader)

        assert self._timeout == 0
        assert self._num_workers == 0
        
        self._dataset_fetcher = _CustomDatasetKind.create_fetcher(
            self._dataset, self._auto_collation,self._collate_fn, self._drop_last
        )

    def _next_data(self):
        index = self._next_index()
        data = self._dataset_fetcher.fetch(index)
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data


from torch.utils.data import Dataset, Sampler
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]

class CustomDataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, sampler, collate_fn, pin_memory):
        super(CustomDataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size,sampler=sampler,
            shuffle=shuffle, collate_fn=collate_fn, 
            pin_memory=pin_memory
        )
            
    
    def _get_iterator(self):
            return CustomDataLoaderIter(self)
        