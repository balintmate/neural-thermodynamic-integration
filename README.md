# Neural TI

This repository contains the implementation of the paper
> [Neural Thermodynamic Integration: Free Energies from Energy-based Diffusion Models](https://arxiv.org/abs/2406.02313) by Bálint Máté, François Fleuret and Tristan Bereau.

## Environment
The ```install.sh``` script will create a virtualenv necessary to run the experiments. The only requirement for this is python>=3.9.

## Toy experiment
The notebook ```toy_example.ipynb``` contains a simple experiment that demonstrates the idea on a 1D Gaussian mixture.

## 3D Lennard-Jones experiment

The ```run_exp.sh``` activates the virtualenv created by ```install.sh``` and then executes ```experiments/main.py``` using the configs ```experiments/config.yaml``` and ```experiments/LJ3D.yaml```. When executing for the first time, it begins with generating the training data using MCMC. The samples are then dumped to files and loaded in later runs.


## Logging
All the plots and metrics are also logged to the ```experiments/wandb```directory by default. If you create a file at ```experiments/wandb.key``` containing your weights and biases key, then all the logs will be pushed to your wandb account.

## Citation
If you find our paper or this repository useful, consider citing us at

```
@misc{máté2024neural,
        title={Neural Thermodynamic Integration: Free Energies from Energy-based Diffusion Models}, 
        author={Bálint Máté and François Fleuret and Tristan Bereau},
        year={2024},
        eprint={2406.02313},
        archivePrefix={arXiv},
        primaryClass={cond-mat.stat-mech}
        url={https://arxiv.org/abs/2406.02313}
}
```
