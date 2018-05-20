# virtual-adversarial-training
Pytorch implementation of "Virtual Adversarial Training: a Regularization Method for Supervised and Semi-Supervised Learning" http://arxiv.org/abs/1704.03976

## For reproducing semi-supervised learning results for SVHN :
```python main.py --dataroot=<dataroot> --dataset=svhn --method=vatent```
  
## For reproducing semi-supervised learning results for CIFAR10 :
```python main.py --dataroot=<dataroot> --dataset=cifar10 --method=vatent --num_epochs=500 -epoch_decay_start=460 --epsilon=10.0 --top_bn=False```
