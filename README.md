# VODCON

## Running experiments

Experiments are run using `run.py` with the corresponding config file as argument.

```
cd src/navigation
python run.py configs/<config_file>.ini
```

Any unspecified arguments are loaded from `default_parameters.ini` and those parameters absent from this default config file are loaded from `arguments.py`. Certain script parameters are specified in the `SCRIPT_PARMS` section and their purpose is explained in the corresponding section in `default_parameters.ini`.

To start off, a simple config for running a single job on SLURM's debug queue is provided in `configs/single_debug_job.ini`. Run this initially as a sanity check.

```
python run.py configs/single_debug_job.ini.ini
```

An example hyperparameter sweep config file is provided in `configs/core_hyperparam_sweep.ini`.
