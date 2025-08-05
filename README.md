# Exploring the performance of Kolmogorov–Arnold Networks on continual learning tasks


## Motivation
Kolmogorov-Arnold Networks (KANs) are a recently introduced alternative to multi layer perceptrons (MLPs), motivated by interpretability and approximation efficiency. Previous work conjectured that KANs may be suitable for continual learning, which is a learning paradigm where data is presented to a learning algorithm sequentially and may change over time. This conjecture was supported by initial results for learning a single-variable target function in a continual learning setting, where the KAN demonstrated negligible forgetting compared to the baseline MLP. Utilizing the pykan package, we investigated the performance of KANs for more difficult continual learning settings than considered in previous work: we evaluated KANs with deeper architectures using more difficult datasets, including multi-variable target functions and Split-MNIST (a standard digit recognition benchmark for continual learning). When learning simple functions with small KANs, we reproduce the previous conclusion that KANs achieve minor forgetting compared to MLPs. However, on Split-MNIST, both KANs and MLPs suffered from catastrophic forgetting.

## Getting Started
To install required python dependencies, run
```sh
pip3 install -r requirements.txt
```

<br>
<br>

To run all experiments, run
```sh
./scripts/run_experiments.sh
```
or, to only run a single experiment, run
```sh
python3 -m experiments.XX_experiment_name_here
```

<br>
<br>

To launch the dashboard and look at run results, run
```sh
./scripts/launch_dashboard.sh
```
to instead convert experiment results to plots, run
```sh
./scripts/export_figures.sh
```
Generated figures are accessible in the `figures` directory.