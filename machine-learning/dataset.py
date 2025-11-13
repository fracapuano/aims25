from datasets import load_dataset


class EEG2MEGDataloader:
    """A Jax-based wrapper around dataset to perform data loading."""
    def __init__(self, dataset_id: str, split:str="train", batch_size:int=32, seed=42):
        self.batch_size = batch_size

        self.dataset = load_dataset(dataset_id, split=split).with_format("jax")
        self.dataset_iter = self.dataset.shuffle(seed=seed).iter(batch_size=self.batch_size)
    
    def _prepare_batch(self, batch: dict) -> dict:
        raise NotImplementedError("Subclasses of this dataloader must implement this functionality.")

    def __iter__(self):
        for batch in self.dataset_iter:
            yield self._prepare_batch(batch)


class MiniEEG2MEGDataloader(EEG2MEGDataloader):
    def _prepare_batch(self, batch: dict) -> dict:
        """Applies (minimal) preprocessing steps to a batch pulled from the upstream dataset.
        In this case collapses the data recorded on the different eeg/meg sensors shrinking down the
        number of channels."""
        eeg = batch["eeg"].mean(axis=1)  # (B, C, T) -> (B, T)
        meg = batch["meg"].mean(axis=1)

        return {"eeg": eeg, "meg": meg}

