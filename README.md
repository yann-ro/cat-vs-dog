<h1 align=center>Cat vs Dog</br>Binary Classification on reduced dataset</h1> 

![](figures/cat%20%26%20dog%20-%20test.png)

# Baseline
### Random classifier
> **Note**</br>
> Test accuracy: 0.50 (balanced dataset)

### Resnet50 without any experiments
![](figures/Resnet50%20-%20basic%20augmentations.png)
> **Note**</br>
> Test accuracy: 0.8095

# Improve performances vs avoid overfitting

## Choice of feature extractor

### Resnet
- easy to implement 
- easily scalable on different size
- compromise between accuracy and overfitting

## Learning rate and optimizer


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
> Test accuracy: 0.8660

## Other tracks
