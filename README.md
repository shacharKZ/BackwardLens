# BackwardLens

## Description
This is a demo provided for the paper: Backward Lens: Projecting Language Model Gradients
into the Vocabulary Space

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
