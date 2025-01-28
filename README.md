# METAGENE-1: Metagenomic Foundation Model for Pandemic Monitoring

This repository contains serving code for METAGENE-1.

## Introduction

METAGENE-1 is a 7-billion-parameter autoregressive transformer language model, which we
refer to as a metagenomic foundation model, pretrained on a novel corpus of diverse
metagenomic DNA and RNA reads sequenced from wastewater.

This repository contains code for pretraining METAGENE-1. It aims to provide a reference
for future pretraining efforts. Note that the metagenomic pretraining dataset is not yet
public (see [data details](#data-details) below). However, this repository will be
updated in the future as the metagenomic data is publically released.

## Quick Tour

- [`serve`](serve/): serving code for METAGENE-1.
    - Embed
        - [`embed.py`](serve/embed/embed.py): containing template to extract embeddings from METAGENE-1.
    - Evaluate
        - [`evaluate_mteb.py`](serve/evaluate/evaluate_mteb.py): containing evaluation code for METAGENE-1 on [gene-mteb](https://github.com/metagene-ai/gene-mteb) tasks.
    - Quantize
        - [`quantize_awq.py`](serve/quantize/quantize_awq.py): containing code to quantize METAGENE-1 using [AutoAWQ](https://github.com/casper-hansen/AutoAWQ).
        - [`quantize_bnb.py`](serve/quantize/quantize_bnb.py): containing code to quantize METAGENE-1 using [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).
        - [`quantize_quanto.py`](serve/quantize/quantize_quanto.py): containing code to quantize METAGENE-1 using [quanto](https://github.com/huggingface/optimum-quanto).

## Installation

To install dependencies, run:

```bash
conda create -n metagene python=3.10 -y && conda activate metagene
./scripts/set_env.sh
```

## Evaluation on Gene-MTEB 

Run the following command to evaluate METAGENE-1 on gene-mteb tasks:
```bash
./scripts/run_evaluate.sh
```

## Quantization

Run the following command to quantize METAGENE-1:
```bash
./scripts/run_quantize.sh
```