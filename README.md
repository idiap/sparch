<!--
SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>

SPDX-License-Identifier: BSD-3-clause

This file is part of the sparch package
--->

# SpArch: Spiking Architectures for Speech Technology

This [PyTorch](https://pytorch.org/) based toolkit is for developing spiking neural networks (SNNs) by training and testing them on speech command recognition tasks. It was published as part of the following paper: [A Surrogate Gradient Spiking Baseline for Speech Command Recognition](https://doi.org/10.3389/fnins.2022.865897) by A. Bittar and P. Garner (2022).


## Data

### Spiking data sets

In order to rectify the absence of free spike-based benchmark datasets, [Cramer et al. (2020)](https://doi.org/10.1109/TNNLS.2020.3044364) have recently released two spiking datasets for speech command recognition using [LAUSCHER](https://github.com/electronicvisions/lauscher), a biologically plausible model to convert audio waveforms into spike trains based on physiological processes.

- The Spiking Heidelberg Digits (SHD) dataset contains spoken digits from 0 to 9 in both English and German (20 classes). The train and test sets contain 8332 and 2088 examples respectively (there is no validation set).

- The Spiking Speech Commands (SSC) dataset is based on the Google Speech Commands v0.2 dataset and contains 35 classes from a larger number of speakers. The number of examples in the train, validation and test splits are 75466, 9981 and 20382 respectively.

Both data sets can be downloaded from the [Zenke Lab website](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).

### Non-spiking data sets

The original, non-spiking versions of the SHD and SSC datasets are also available and can be used in this framework. With our approach, acoustic features are extracted from the waveforms and directly fed into spiking neural networks. The conversion from filterbank features to spike trains therefore happens in a trainable fashion inside the neuronal dynamics of the first hidden layer. Moreover, the initial (non-trainable) transformation of the audio waveforms into filterbank features is fast enough to be performed during training, so that no preliminary processing of the audio is required.

- The Heidelberg Digits (HD) data set can also be downloaded from the same [website](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) as its spiking counterpart.

- The second version of the (non-spiking) Google Speech Command (SC) data set, introduced by [Warden (2018)](https://arxiv.org/abs/1804.03209), can be found on the [TensorFlow website](https://www.tensorflow.org/datasets/catalog/speech_commands).

Note that the training/validation/testing splits are different for the SC and SSC data sets. The SC has a 80% / 10% / 10% partition instead of 70% / 10% / 20% for the SSC, which makes a direct comparison impossible between the accuracies on the two tasks.

## Installation

    git clone https://github.com/idiap/sparch.git
    cd sparch
    pip install -r requirements.txt
    python setup.py install

### Run experiments

All experiments on the speech command recognition datasets can be run from the `run_exp.py` script. The experiment configuration can be specified using parser arguments. Run the command `python run_exp.py -h` to get the descriptions of all possible options. For instance, if you want to run a new SNN experiment with adLIF neurons on the SC dataset,

    python run_exp.py --model_type adLIF --dataset_name sc \
        --data_folder <PATH-TO-DATASET-FOLDER> --new_exp_folder <OUTPUT-PATH>

You can also continue training from a checkpoint

    python run_exp.py --use_pretrained_model 1 --load_exp_folder <OUTPUT-PATH> \
        --dataset_name sc --data_folder <PATH-TO-DATASET-FOLDER> \
        --start_epoch <LAST-EPOCH-OF-PREVIOUS-TRAINING>


## Usage

Spiking neural networks (SNNs) based on the surrogate gradient approach are defined in `sparch/models/snn_models.py` as PyTorch modules. We distinguish between four types of spiking neuron models based on the linear Leaky Integrate and Fire (LIF),

- LIF: LIF neurons without layer-wise recurrent connections
- RLIF: LIF neurons with layer-wise recurrent connections
- adLIF: adaptive LIF neurons without layer-wise recurrent connections
- RadLIF: adaptive LIF neurons with layer-wise recurrent connections.

An SNN can then be simply implemented as a PyTorch module:

    from sparch.models.snns import SNN

    # Build input
    batch_size = 4
    nb_steps = 100
    nb_inputs = 20
    x = torch.Tensor(batch_size, nb_steps, nb_inputs)
    nn.init.uniform_(x)

    # Define model
    model = SNN(
        input_shape=(batch_size, nb_steps, nb_inputs),
        neuron_type="adLIF",
        layer_sizes=[128, 128, 10],
        normalization="batchnorm",
        dropout=0.1,
        bidirectional=False,
        use_readout_layer=False,
        )

    # Pass input through SNN
    y, firing_rates = model(x)


and used for other tasks. Note that by default, the last layer of the SNN is a readout layer that produces non-sequential outputs. For sequential outputs, simply set `use_readout_layer=False`. Moreover, the inputs do not have to be binary spike trains.

Standard artificial neural networks (ANNs) with non-spiking neurons are also defined in `sparch/models/ann_models.py`, in order to have a point of comparison for the spiking baseline. We implemented the following types of models: MLPs, RNNs, [LiGRUs](https://doi.org/10.1109/TETCI.2017.2762739) and [GRUs](https://doi.org/10.48550/arXiv.1406.1078).

## Structure of the git repository

```
.
├── sparch
│   ├── dataloaders
|   |   ├── nonspiking_datasets.py
│   │   └── spiking_datasets.py
│   ├── models
|   |   ├── anns.py
│   │   └── snns.py
│   ├── parsers
|   |   ├── model_config.py
│   │   └── training_config.py
│   └── exp.py
|
└── run_exp.py
```

## Citation
If you use this framework in your research, please cite it as
```
@article{bittar2022surrogate,
  title={A surrogate gradient spiking baseline for speech command recognition},
  author={Bittar, Alexandre and Garner, Philip N},
  journal={Frontiers in Neuroscience},
  volume={16},
  year={2022},
  publisher={Frontiers},
  doi={10.3389/fnins.2022.865897}
}
```
