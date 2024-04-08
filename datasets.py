import pynever.datasets as pyn_datasets
import torch.utils.data as pyt_data
import pandas


class ComponentDegradationAD(pyn_datasets.Dataset, pyt_data.Dataset):

    """
        Dataset compatible with pynever and pytorch for the "One Year Industrial Component Degradation" dataset.
        This particular dataset is for the task of Anomaly Detection (i.e., to recognize when the blade is degraded).
        Moreover, it is developed for the use with an autoencoder (i.e., the outputs corresponds to the inputs).

        Attributes
        ----------

    """

    def __init__(self, filepath: str, columns_to_drop: list = None, is_training: bool = True):

        df = pandas.read_csv(filepath)
        if columns_to_drop is None:
            df = df.drop(labels=['timestamp', 'mode'], axis='columns')
        else:
            df = df.drop(labels=columns_to_drop, axis='columns')

        # We define an arbitrary cutoff assuming that the first 200000 samples represent the behaviour of a new blade.
        new_cutoff = 200000
        if is_training:
            self.df = df[0:new_cutoff]
        else:
            self.df = df  # [new_cutoff:]

    def __len__(self):

        return self.df.__len__()

    def get_features_ids(self):
        return self.df.columns

    def __getitem__(self, index: int):

        if index >= self.__len__():
            raise IndexError

        sample = self.df.iloc[index].values
        target = self.df.iloc[index].values
        return sample, target
