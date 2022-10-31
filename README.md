# Post-breach Recovery: Protection against White-box Adversarial Examples for Leaked DNN Models

### ABOUT

This repository contains code implementation of the paper "[Post-breach Recovery: Protection against White-box Adversarial Examples for Leaked DNN Models](https://www.shawnshan.com/files/publication/salt.pdf)", at *ACM CCS 2022*. 
This recovery tool for model breaches is developed by researchers at [SANDLab](https://sandlab.cs.uchicago.edu/), University of Chicago.  

### Note
The code base currently supports CIFAR10 dataset. Adapting to new datasets is possible by editing the data loading and model loading code. 

### DEPENDENCIES

Our code is implemented and tested on the dependency below. The code will likely to work on other versions as well. 

- `torch==1.12.1`
- `tensorflow==2.4.1`

### Recovery pipeline 

This code base includes code to generate hidden distribution, train model variants, and detect adversarial examples. 

#### Step 1: Generate hidden distributions

This code will generate the hidden data distributions (used for model training), and save the hidden distribution in a local directory. 

`python3 gen_hidden.py`

You can overwrite the default output directory using `--hidden-dir`. 

#### Step 2: Train models

To steamline the eval process, we are going to train multiple models at first. 

Run `python3 train_cifar10.py`. This will generate 10 CIFAR models, each trained using different hidden distribution. 


#### Step 3: Eval recovery

This code simulate the scenario where a number of models are breached. 

`python3 eval_recovery.py`. 

The script will predict out the final attack success rate (attacks that transfer to the recovered model and evade our detection). We select a number of breached model each time, and jointly optimize the attack on all of selected model. 

This code takes awhile to run, mostly because ensemble attack with multiple models is very expensive. 

### Citation
```
@inproceedings{shan2022poison,
  title={Post-breach Recovery: Protection against White-box Adversarial Examples for Leaked DNN Models},
  author={Shan, Shawn and Ding, Wenxin and Wenger, Emily and Zheng, Haitao and Zhao, Ben Y},
  journal={Proc. of CCS},
  year={2022}
}
```
