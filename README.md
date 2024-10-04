# BackwardLens

## Description
This is a demo provided for the paper: Backward Lens: Projecting Language Model Gradients
into the Vocabulary Space

Try our demo: [![Colab BackwardLens Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o_dKmxkCMNzyvbztxXKEkjQGr_s0GoE6?usp=sharing)

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)

## Installation

### Prerequisites
Make sure you have Python 3.10 installed on your machine.

All files must be in the same folder

### Installing Dependencies
In your Python environment, install the required dependencies using `pip` and the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Usage

Make sure all files are in the same folder and run `backward_lens_demo.ipynb`.

This script demonstrates how to obtain the backward lens projection from GPT-2 models.

Additionally, the final blocks of the script show how to exclude tokens' rankings from VJPs and how to observe the differences in VJPs' norms between layers.


## Forward Pass Shift

Use the submodule memit_for_BackwardLens or directly use the repo [here](https://github.com/shacharKZ/memit_for_BackwardLens).



## Citing

```bibtex
@article{katz2024backward,
  title={Backward lens: Projecting language model gradients into the vocabulary space},
  author={Katz, Shahar and Belinkov, Yonatan and Geva, Mor and Wolf, Lior},
  journal={arXiv preprint arXiv:2402.12865},
  year={2024}
}
```
