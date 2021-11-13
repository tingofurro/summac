# SummaC: Summary Consistency Detection

This repository contains the code for TACL2021 paper: SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization

We release: (1) the trained SummaC models, (2) the SummaC Benchmark and data loaders, (3) training and evaluation scripts.

<p align="center">
  <img width="460" height="354" src="https://tingofurro.github.io/images/tacl2021_summac.png">
</p>

## Trained SummaC Models

The two trained models SummaC-ZS and SummaC-Conv are implemented in `model_summac.py` ([link](https://github.com/tingofurro/summac/blob/master/model_summac.py)):

- *SummaC-ZS* does not require a model file (as the model is zero-shot and not trained): it can be used as seen at the bottom of the `model_summac.py`.
- *SummaC-Conv* requires a `start_file` which contains the trained weight for the convolution layer. The default `start_file` used to compute results is available in this repository ( `summac_conv_vitc_sent_perc_e.bin` [download link](https://github.com/tingofurro/summac/raw/master/summac_conv_vitc_sent_perc_e.bin)).

## SummaC Benchmark

