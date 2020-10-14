# Fast Adversarial Robustness Certification of Nearest Prototype Classifiers for Arbitrary Seminorms
Methods for adversarial robustness certification aim to provide an upper bound 
on the test error of a classifier under adversarial manipulation of its input. 
Current certification methods are computationally expensive and limited to 
attacks that optimize the manipulation with respect to a norm. We overcome 
these limitations by investigating the robustness properties of Nearest 
Prototype Classifiers (NPCs) like learning vector quantization and large 
margin nearest neighbor. For this purpose, we study the hypothesis margin. We 
prove that if NPCs use a dissimilarity measure induced by a seminorm, the 
hypothesis margin is a tight lower bound on the size of adversarial attacks 
and can be calculated in constant time—this provides the first adversarial 
robustness certificate calculable in reasonable time. Finally, we show that 
each NPC trained by a triplet loss maximizes the hypothesis margin and is 
therefore optimized for adversarial robustness. In the presented evaluation, 
we demonstrate that NPCs optimized for adversarial robustness are competitive 
with state-of-the-art methods and set a new benchmark in certification speed.

## Requirements

The package requires Python 3.6 and we recommend to use a virtual 
environment or Docker image. To install the Python requirements use the 
following command:

```setup
pip install -r requirements.txt
```

## Training

The `train.py` script is available for training NPCs similar to the ones 
presented in the paper. To run the training script with the exact same 
parameters as the models in the paper, the `--replicate` parameter is 
available. Otherwise, see the available descriptions for defined arguments 
to define a model of your choice.

The following set of commands will train the models from the paper (specify 
the desired output directory by adding the argument `--save_dir` followed by
the output path):


```train
python train.py --model glvq --p_norm inf --dataset mnist --replicate
python train.py --model glvq --p_norm 2 --dataset mnist --replicate
python train.py --model rslvq --dataset mnist --replicate
python train.py --model gtlvq --dataset mnist --replicate

python train.py --model glvq --p_norm inf --dataset cifar10 --replicate
python train.py --model glvq --p_norm 2 --dataset cifar10 --replicate
python train.py --model rslvq --dataset cifar10 --replicate
python train.py --model gtlvq --dataset cifar10 --replicate

python train.py --dataset breast_cancer --replicate
python train.py --dataset diabetes --replicate
python train.py --dataset cod_rna --replicate
```

### Pre-trained Models

If you wish to evaluate the models presented in the paper directly, without 
retraining them, they can be found in the `weight_files` folder. The model 
corresponding weight file is automatically loaded if `--weights` argument 
is `None` (default) and the evaluation flag `--eval` is set. For example:
 ```
 python train.py --model glvq --p_norm inf --dataset mnist --replicate --eval
 python train.py --dataset diabetes --replicate --eval
 ```
 

## Evaluation
To evaluate an NPC two scripts are available. One for evaluating the 
robustness of a model, and one for evaluating the discussed rejection 
strategy. The evaluation scripts are designed to be used on newly 
trained models. For replicating the results of the paper the `--replicate` 
parameters is available.

### Robustness evaluation
The `evaluation_robustness.py` reports the CTE, URTE, and LRTE for a given 
model. The script has a number of parameters for which the default value is 
dependent on a number of other parameters.

To obtain the results presented in the main part of the paper run the 
following commands:

```eval
python evaluation_robustness.py --save_dir ./output --model glvq --model_norm inf --dataset mnist --replicate
python evaluation_robustness.py --save_dir ./output --model rslvq --dataset mnist --replicate
python evaluation_robustness.py --save_dir ./output --model glvq --model_norm 2 --dataset mnist --replicate
python evaluation_robustness.py --save_dir ./output --model gtlvq --dataset mnist --replicate

python evaluation_robustness.py --save_dir ./output --model glvq --model_norm inf --dataset cifar10 --replicate
python evaluation_robustness.py --save_dir ./output --model rslvq --dataset cifar10 --replicate
python evaluation_robustness.py --save_dir ./output --model glvq --model_norm 2 --dataset cifar10 --replicate
python evaluation_robustness.py --save_dir ./output --model gtlvq --dataset cifar10 --replicate
```

**Note that the certificate (without the empirical robustness evaluation) can 
be computed by the command presented in the Section Pre-trained Models.**

### Rejection evaluation
The `evaluation_rejection.py` script determines the false reject rate of an 
NPC classifier based on the strategy discussed in the paper. With the `--plot` 
parameter set, it also produces a similar plot as presented in the paper. 
 
To obtain the results and plots as presented in the paper, run the 
following command:
 
```reject
python evaluation_rejection.py --save_dir ./output --replicate
```

## Plots
In Figure 1 of the paper, three plots are presented. To replicate the plots 
the following scripts are available. Where possible, the scripts are designed 
to be reused with newly trained models. 

### URTE for different losses (Left)
The script `plot_losses.py` recreates the left plot of Figure 1. To reconstruct 
the 
plot, the script has to run the MNIST test dataset through 4 different models. 
To reduce the time this takes, the `--number_of_samples` parameter can be used.

To obtain the plot as presented in the paper, run: 

```losses
python plot_losses.py --save_dir ./output  --replicate
```

### Training logs (Middle)
The script `plot_training.py` can be used to create an overview of the 
development of URTE, CTE, and loss during training. To do this, it requires 
the path to the csv files with training logs to be given using the 
`--csv_file` parameter. 

To replicate the plot from the paper, run:

```training
python plot_training.py --save_dir ./output --replicate
```

### False rejection rate (Right)
The final plot in Figure 1 is created using the `evaluation_rejection.py` 
script. To reconstruct the plot presented in the paper run the same command 
as for the full evaluation: 

```reject_plot
python evaluation_rejection.py --save_dir ./output  --plot --replicate
```
