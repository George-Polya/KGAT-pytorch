from data_loader.loader_base import *
from data_loader.loader_base import _BaseDataLoaderIter


class KG_Dataset(data.Dataset):
    def __init__(self, loaderbase, seed, phase="train_user_set"):
        self.loaderbase = loaderbase
        self.user_dict = {
            "train_user_set":self.loaderbase.train_user_dict,
            "test_user_set":self.loaderbase.test_user_dict
        }
        self.phase=phase
        self.seed=seed
    def __len__(self):
        return len(self.user_dict[self.phase])
    
    def __getitem__(self, index):
        # exist_users = self.user_dict[self.phase]
        # exist_users_dict = {i:key for i, key in enumerate(exist_users)}
        # return exist_users_dict[index]
        return index


class _CustomDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last
    
    def fetch(self, possible_batched_index):
        random.seed(self.dataset.seed)
        np.random.seed(self.dataset.seed)
        kg_dict = self.dataset.loaderbase.train_kg_dict
        
        batch_size = len(possible_batched_index)
        highest_neg_idx = self.dataset.loaderbase.n_users_entities
        
        return self.dataset.loaderbase.generate_kg_batch(kg_dict, batch_size, highest_neg_idx)

class KG_DataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, collate_fn, pin_memory=False):
        super(KG_DataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size,
            shuffle=shuffle, collate_fn=collate_fn, 
            pin_memory=pin_memory
        )
            
    
    def _get_iterator(self):
            return CustomDataLoaderIter(self)


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
        # print(index)
        data = self._dataset_fetcher.fetch(index)
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data

