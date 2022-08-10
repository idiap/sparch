#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is where the dataloaders and defined for the HD and SC datasets.
"""
import logging
import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchaudio_augmentations import ComposeMany
from torchaudio_augmentations import Gain
from torchaudio_augmentations import Noise
from torchaudio_augmentations import PolarityInversion
from torchaudio_augmentations import RandomApply
from torchaudio_augmentations import Reverb

logger = logging.getLogger(__name__)


class HeidelbergDigits(Dataset):
    """
    Dataset class for the original non-spiking Heidelberg Digits (HD)
    dataset. Generated mel-spectrograms use 40 bins by default.

    Arguments
    ---------
    data_folder : str
        Path to folder containing the Heidelberg Digits dataset.
    split : str
        Split of the HD dataset, must be either "train" or "test".
    use_augm : bool
        Whether to perform data augmentation or not.
    min_snr, max_snr : float
        Minimum and maximum amounts of noise if augmentation is used.
    p_noise : float in (0, 1)
        Probability to apply noise if augmentation is used, i.e.,
        proportion of examples to which augmentation is applied.
    """

    def __init__(
        self,
        data_folder,
        split,
        use_augm,
        min_snr,
        max_snr,
        p_noise,
    ):

        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split {split}")

        # Get paths to all audio files
        self.data_folder = data_folder
        filename = self.data_folder + "/" + split + "_filenames.txt"
        with open(filename, "r") as f:
            self.file_list = f.read().splitlines()

        # Data augmentation
        if use_augm and split == "train":
            transforms = [
                RandomApply([PolarityInversion()], p=0.8),
                RandomApply([Noise(min_snr, max_snr)], p_noise),
                RandomApply([Gain()], p=0.3),
                RandomApply([Reverb(sample_rate=16000)], p=0.6),
            ]
            self.transf = ComposeMany(transforms, num_augmented_samples=1)
        else:
            self.transf = lambda x: x.unsqueeze(dim=0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        # Read waveform
        filename = self.file_list[index]
        x = self.data_folder + "/audio/" + filename
        x, _ = torchaudio.load(x)

        # Apply augmentation
        x = self.transf(x).squeeze(dim=0)

        # Compute acoustic features
        x = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=40)

        # Get label (digits 0-9 in eng and germ)
        y = int(filename[-6])
        if filename[5] == "g":
            y += 10

        return x, y

    def generateBatch(self, batch):

        xs, ys = zip(*batch)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)

        return xs, xlens, ys


class SpeechCommands(Dataset):
    """
    Dataset class for the original non-spiking Speech Commands (SC)
    dataset. Generated mel-spectrograms use 40 bins by default.

    Arguments
    ---------
    data_folder : str
        Path to folder containing the Heidelberg Digits dataset.
    split : str
        Split of the HD dataset, must be either "train" or "test".
    use_augm : bool
        Whether to perform data augmentation or not.
    min_snr, max_snr : float
        Minimum and maximum amounts of noise if augmentation is used.
    p_noise : float in (0, 1)
        Probability to apply noise if augmentation is used, i.e.,
        proportion of examples to which augmentation is applied.
    """

    def __init__(
        self,
        data_folder,
        split,
        use_augm,
        min_snr,
        max_snr,
        p_noise,
    ):

        if split not in ["training", "validation", "testing"]:
            raise ValueError(f"Invalid split {split}")

        # Get paths to all audio files
        self.data_folder = data_folder
        EXCEPT_FOLDER = "_background_noise_"

        def load_list(filename):
            filepath = os.path.join(self.data_folder, filename)
            with open(filepath) as f:
                return [os.path.join(self.data_folder, i.strip()) for i in f]

        if split == "training":
            files = sorted(str(p) for p in Path(data_folder).glob("*/*.wav"))
            exclude = load_list("validation_list.txt") + load_list("testing_list.txt")
            exclude = set(exclude)
            self.file_list = [
                w for w in files if w not in exclude and EXCEPT_FOLDER not in w
            ]
        else:
            self.file_list = load_list(str(split) + "_list.txt")

        self.labels = sorted(next(os.walk("./" + data_folder))[1])[1:]

        # Data augmentation
        if use_augm and split == "training":
            transforms = [
                RandomApply([PolarityInversion()], p=0.8),
                RandomApply([Noise(min_snr, max_snr)], p_noise),
                RandomApply([Gain()], p=0.3),
                RandomApply([Reverb(sample_rate=16000)], p=0.6),
            ]
            self.transf = ComposeMany(transforms, num_augmented_samples=1)
        else:
            self.transf = lambda x: x.unsqueeze(dim=0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        # Read waveform
        filename = self.file_list[index]
        x, _ = torchaudio.load(filename)

        # Apply augmentation
        x = self.transf(x).squeeze(dim=0)

        # Compute acoustic features
        x = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=40)

        # Get label
        relpath = os.path.relpath(filename, self.data_folder)
        label, _ = os.path.split(relpath)
        y = torch.tensor(self.labels.index(label))

        return x, y

    def generateBatch(self, batch):

        xs, ys = zip(*batch)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)

        return xs, xlens, ys


def load_hd_or_sc(
    dataset_name,
    data_folder,
    split,
    batch_size,
    shuffle=True,
    use_augm=False,
    min_snr=0.0001,
    max_snr=0.9,
    p_noise=0.1,
    workers=0,
):
    """
    This function creates a dataloader for a given split of
    the HD or SC dataset.

    Arguments
    ---------
    dataset_name : str
        The name of the dataset, either hd or sc.
    data_folder : str
        Path to folder containing the desired dataset.
    split : str
        Split of the desired dataset, must be either "train" or "test" for hd
        and "training", "validation" or "testing" for sc.
    batch_size : int
        Number of examples in a single generated batch.
    shuffle : bool
        Whether to shuffle examples or not.
    use_augm : bool
        Whether to perform data augmentation or not.
    min_snr, max_snr : float
        Minimum and maximum amounts of noise if augmentation is used.
    p_noise : float in (0, 1)
        Probability to apply noise if augmentation is used, i.e.,
        proportion of examples to which augmentation is applied.
    workers : int
        Number of workers.
    """
    if dataset_name not in ["hd", "sc"]:
        raise ValueError(f"Invalid dataset name {dataset_name}")

    if split not in ["train", "valid", "test"]:
        raise ValueError(f"Invalid split name {split}")

    if dataset_name == "hd":

        if split in ["valid", "test"]:
            split = "test"
            logging.info("\nHD uses the same split for validation and testing.\n")

        dataset = HeidelbergDigits(
            data_folder, split, use_augm, min_snr, max_snr, p_noise
        )

    else:
        if split == "train":
            split = "training"
        elif split == "valid":
            split = "validation"
        else:
            split = "testing"

        dataset = SpeechCommands(
            data_folder, split, use_augm, min_snr, max_snr, p_noise
        )

    logging.info(f"Number of examples in {dataset_name} {split} set: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.generateBatch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )
    return loader
