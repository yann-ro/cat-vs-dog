<h1 align=center>Cat vs Dog</br>Binary Classification on reduced dataset</h1> 

![](figures/cat%20%26%20dog%20-%20test.png)

# Baseline
### Random classifier
> **Note**</br>
> Test accuracy: 0.50 (balanced dataset)

### Resnet50 without any experiments
![](figures/Resnet50%20-%20basic%20augmentations.png)
> **Note**</br>
> Test accuracy: 0.81

# Improve performances vs avoid overfitting

## Choice of feature extractor

### Resnet
- easy to implement 
- easily scalable on different size

## Learning rate and optimizer
- Optimizer: Adam
> create a variable learning rate for all layers between 0 and defined learning rate.

- Select the most intersting lr : 
    - [sgugger blog](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html#how-do-you-find-a-good-learning-rate)

    ![](figures/find_lr.png)
    > **Note**</br>
    > Suggested LR: 1.67E-07

## Image size

## Nb Epochs

## Batch size
Constraints:
* small batch size:
    > convergence time
* big batch size:
    > computer storage limiation

## Transfer learning
> **Warning**</br>
> Pretrain on imagenet was not considered because imagenet contains images of cats and dog.

## Augmentations
![](figures/cat%20%26%20dog%20-%20train.png)
- select relevant augmentation with data (photos in a casual context)
    - need to be invariant to 3D rotation, scale, zoom, crop, color, brightness
    - can be many instances on the pictures (many times the same class, human, ...)

![](figures/Resnet18%20-%20advanced%20augmentations.png)
> **Note**</br>
> Test accuracy: 0.94

## Other tracks
### Self-supervised learning
- pretrain model using a pretext task: the more relevant task should be predicting relative position of image patches ([ref.](https://atcold.github.io/pytorch-Deep-Learning/en/week10/10-1/))
    - easy to implement and easy to understand
