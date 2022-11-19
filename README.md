# Cat vs Dog Classification

![](figures/cat%20%26%20dog%20-%20test.png)

## Baseline

![](figures/Resnet50%20-%20basic%20augmentations.png)
> **Note**
> 
> accuracy on test 0.8095

## Choice of feature extractor

### Resnet
- easy to implement 
- easily scalable on different size
- compromise between accuracy and overfitting

## Improve performances

### Transfer learning
> **Warning**
> 
> Pretrain on imagenet was not considered bcs it contain images of cats and dog so inject new data

## Avoid overfitting

![](figures/Resnet18%20-%20advanced%20augmentations.png)

### Augmentations
![](figures/cat%20%26%20dog%20-%20train.png)
- select relevant with 

## Other tracks
