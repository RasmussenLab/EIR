<p align="center">
  <img src="docs/source/_static/img/EIR_logo.png">
</p>

<p align="center">
    <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-APGL-5B2D5B.svg" /></a>
  
  <a href="https://www.biorxiv.org/content/10.1101/2021.06.11.447883" alt="bioRxiv">
        <img src="https://img.shields.io/badge/Paper-bioRxiv-B5232F.svg" /></a>
  
  <a href="https://www.python.org/downloads/" alt="Python">
        <img src="https://img.shields.io/badge/python-3.11-blue.svg" /></a>
  
   <a href="https://pypi.org/project/eir-dl/" alt="Python">
        <img src="https://img.shields.io/pypi/v/eir-dl.svg" /></a>
  
  <a href="https://codecov.io/gh/arnor-sigurdsson/EIR" alt="Coverage">
        <img src="https://codecov.io/gh/arnor-sigurdsson/EIR/branch/master/graph/badge.svg" /></a>
  
  <a href='https://eir.readthedocs.io/'>
        <img src='https://readthedocs.org/projects/eir/badge/?version=latest' alt='Documentation Status' /></a>
  
       
</p>

---

Supervised modelling on genotype, tabular, sequence, image, array and binary data.

**WARNING:** This project is in alpha phase. Expect backwards incompatible changes and API changes.

## Install

`pip install eir-dl`

**Important:** The latest version of EIR supports [Python 3.11](https://www.python.org/downloads/). Using an older version of Python will install a old version of EIR, which likely be incompatible with the current documentation and might contain bugs. Please ensure that you are installing EIR in a Python 3.11 environment.

## Usage

Please refer to the [Documentation](https://eir.readthedocs.io/en/latest/index.html) for examples and information.

## Use Cases

EIR allows for training and evaluating various deep-learning models directly from the command line. This can be useful for:

- Quick prototyping and iteration when doing supervised modelling on new datasets.
- Establishing baselines to compare against other methods.
- Fitting on data sources such as large-scale genomics, where DL implementations are not commonly available.

If you are an ML/DL researcher developing new models, etc., it might not fit your use case. However, it might provide a quick baseline for comparison to the cool stuff you are developing, and there is some degree of [customization](https://eir.readthedocs.io/en/latest/tutorials/tutorial_index.html#customizing-eir) possible.

## Features

- Train models directly from the command line through `.yaml` configuration files.
- Training on [genotype](https://eir.readthedocs.io/en/latest/tutorials/01_basic_tutorial.html), [tabular](https://eir.readthedocs.io/en/latest/tutorials/02_tabular_tutorial.html), [sequence](https://eir.readthedocs.io/en/latest/tutorials/03_sequence_tutorial.html), [image](https://eir.readthedocs.io/en/latest/tutorials/05_image_tutorial.html), [array](https://eir.readthedocs.io/en/latest/tutorials/08_array_tutorial.html) and [binary](https://eir.readthedocs.io/en/latest/tutorials/06_raw_bytes_tutorial.html) input data, with various modality-specific settings available.
- Seamless multi-modal (e.g., combining [text + image + tabular data](https://eir.readthedocs.io/en/latest/tutorials/07_multimodal_tutorial.html), or any combination of the modalities above) training.
- Train multiple features extractors on the same data source, e.g., [combining vanilla transformer, Longformer and a pre-trained BERT variant](https://eir.readthedocs.io/en/latest/tutorials/04_pretrained_sequence_tutorial.html) for text classification.
- Supports continuous (i.e., regression) and categorical (i.e., classification) targets.
- [Multi-task / multi-label](https://eir.readthedocs.io/en/latest/tutorials/07_multimodal_tutorial.html#appendix-b-multi-modal-multi-task-learning) prediction supported out-of-the-box.
- Model explainability for genotype, tabular, sequence and image data built in.
- Computes and graphs various evaluation metrics (e.g., RMSE, PCC and R2 for regression tasks, accuracy, ROC-AUC, etc. for classification tasks) during training.
- [Many more settings](https://eir.readthedocs.io/en/latest/api_reference.html) and configurations (e.g., augmentation, regularization, optimizers) available.

## Citation

If you use `EIR` in a scientific publication, we would appreciate if you could use the following citation:

```
@article{sigurdsson2021deep,
  title={Deep integrative models for large-scale human genomics},
  author={Sigurdsson, Arnor Ingi and Westergaard, David and Winther, Ole and Lund, Ole and Brunak, S{\o}ren and Vilhjalmsson, Bjarni J and Rasmussen, Simon},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Acknowledgements

Massive thanks to everyone publishing and developing the [packages](https://eir.readthedocs.io/en/latest/acknowledgements.html) this project directly and indirectly depends on.
