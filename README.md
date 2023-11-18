# README

It is a group project for HKU STAT7008.

Topic: A Study of Adversarial Attack

Introduction
An adversarial attack refers to a method used to manipulate input data with the goal of misleading machine learning models. These manipulations, known as perturbations, are carefully crafted to be undetectable by humans, but they can cause the model to produce incorrect outputs. 

To counter these attacks, various techniques have been developed to defend against adversarial attacks. However, attackers continuously adapt their methods to overcome these defenses, creating an ongoing challenge in the field. 

In this project, our focus will be on studying both adversarial attacks and defense mechanisms using the CIFAR-10 dataset. It is a widely used dataset in machine learning research, particularly for tasks involving image classification. 

Projected Gradient Descent:

A white-box attack, the attacker has access to the model gradients, so can design specifically to fool the trained model.
It is a constrained optimisation problem: PGD attempts to find the perturbation that maximises the loss of a model on particular input while keeping the size of the perturbation smaller than a specified amount named epsilon.

![image](https://github.com/tklsimon/adversarial-attack/assets/46237598/30dfc32b-95af-400b-a6a5-97e23e0cdef4)
