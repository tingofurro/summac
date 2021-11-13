# SummaC: Summary Consistency Detection

This repository contains the code for TACL2021 paper: SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization

We release: (1) the trained SummaC models, (2) the SummaC Benchmark and data loaders, (3) training and evaluation scripts.

<p align="center">
  <img width="400" src="https://tingofurro.github.io/images/tacl2021_summac.png">
</p>

## Trained SummaC Models

The two trained models SummaC-ZS and SummaC-Conv are implemented in `model_summac.py` ([link](https://github.com/tingofurro/summac/blob/master/model_summac.py)):

- *SummaC-ZS* does not require a model file (as the model is zero-shot and not trained): it can be used as seen at the bottom of the `model_summac.py`.
- *SummaC-Conv* requires a `start_file` which contains the trained weight for the convolution layer. The default `start_file` used to compute results is available in this repository ( `summac_conv_vitc_sent_perc_e.bin` [download link](https://github.com/tingofurro/summac/raw/master/summac_conv_vitc_sent_perc_e.bin)).

## SummaC Benchmark

The SummaC Benchmark consists of 6 summary consistency datasets that have been standardized to a binary classification task. The datasets included are:

<p align="center">
  <img width="500" src="https://tingofurro.github.io/images/tacl2021_summac_benchmark.png">
</p>

The data-loaders for the benchmark are included in `utils_summac_benchmark.py` ([link](https://github.com/tingofurro/summac/blob/master/utils_summac_benchmark.py)). Because the dataset relies on previously published work, the dataset requires the manual download of several datasets. For each of the 6 tasks, the link and instruction to download are present as a comment in the file. Once all the files have been compiled, the benchmark can be loaded and standardized by running:
```
from utils_summac_benchmark import SummaCBenchmark
benchmark_validation = SummaCBenchmark(benchmark_folder="/path/to/summac_benchmark/", cut="val")
```

Note: we have a plan to streamline the process by further improving to automatically download necessary files if not present, if you would like to participate please let us know. If encoutering an issue in the manual download process, please contact us.

## Cite the work

If you make use of the code, models, or algorithm, please cite our paper:
```
@inproceedings{laban2021summac,
  title={SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization},
  author={Philippe Laban and Tobias Schnabel and Paul N. Bennett and Marti A. Hearst},
  booktitle={Proceedings of the Transactions of the Association for Computational Linguistics},
  volume={1},
  year={2020}
}
```

## Contributing

If you'd like to contribute, or have questions or suggestions, you can contact us at phillab@berkeley.edu. All contributions welcome, for example helping make the benchmark more easily downloadable, or improving model performance on the benchmark.
