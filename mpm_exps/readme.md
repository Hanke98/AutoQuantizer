## This folder contains the experiments conducted on MLS-MPM algorithgms

#### Files in this folder

```shell
config/ 	# configurations for the scaling & initial conditions experiments
engine/ 	# code for mpm simulator
grads/		# the gradient data
ranges/		# the range data
inputs/		# some input of meshes and particle positions.

mpm_config.py # parameters for mpm simulator

# The scripts for running the experiments
run_2d_exp.py # 2d experiments scripts
run_3d_scaling_and_init_conditions.py # 3d experiments scripts of the scaling & initial conditions

# code for each single experiment
run_mpm_dithering_exp.py 
run_mpm_elastic_2d.py	 
run_mpm_elastic_3d.py
run_mpm_fluid_2d.py
run_mpm_fluid_3d.py
run_mpm_fluid_3d_scaling.py
run_mpm_optimality_check.py
```
To get the optimal fraction bits for simulation, we need to first compute the simulation gradients and ranges of each variables. We offer our gradients and ranges data in the corresponding folders. You may also recompute the data by sending args to the scripts. 

For example, to compute the gradient of 2D elastic body simultion:

```python
# use -a arg to enable auto-diff
python run_mpm_elastic_2d.py -a
```

To compute the ranges of 3D fluid simultion:

```python
# use -r arg to record ranges
python run_mpm_fluid_3d.py -r
```
For more details, please see the `arg_parse` function in each python file.

## Reporduce commands
#### Reporduce the 2D error bound experiments

```
python -u run_2d_exp.py -T 0
```

#### Reporduce the 2D memory bound experiments

```
python -u run_2d_exp.py -T 1
```

#### Reporduce the 2D dithering experiments

```
python -u run_2d_exp.py -T 2
```

#### Reporduce the 2D optimality check experiments

```
python -u run_2d_exp.py -T 3
```

#### Reporduce the large-scale quantized simulation of elastic body

```
python run_mpm_elastic_3d.py -f
```

#### Reporduce the large-scale quantized simulation of fluids

```
python run_mpm_fluid_3d.py -f
```

#### Reporduce the scaling and initial condition experiments

```
python run_3d_scaling_and_init_conditions.py
```