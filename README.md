## NODE-GAMLSS: Differentiable Additive Model for Interpretable Distributional Deep Learning: 

NodeGAMLSS is an interpretable deep distributional learning GAM model based on the Node-GAM framework and the NAMLSS/GAMLSS framework: 
[NODE GAM: Differentiable Generalized Additive Model for Interpretable Deep Learning](https://arxiv.org/abs/2106.01613)
[Neural Additive Models for Location Scale and Shape: A Framework for Interpretable Neural Regression Beyond the Mean](https://arxiv.org/pdf/2301.11862.pdf).
In short, it trains a GAMLSS model by multi-layer differentiable trees and minimizes the negative log-likelihood of a given distribution.
The distributional parameter restrictions (e.g. postiive variance in a normal distribution) are handled in place.
The code is taken from the [Node-GAM Github](https://github.com/zzzace2000/nodegam/tree/main) implementation and adjusted to account for distributional regression approaches.

## Installation

```bash
pip install nodegamlss
```

## NodeGAM Training

### Sklearn interface

To simply minimize the negative log-likelihood of a normal distribution on a given dataset just use:
```python
from nodegamlss.sklearn import NodeGAMLSS

model = NodeGAMLSS(
    in_features=X.shape[1],
    family="normal",
    device="cpu",
    verbose=False,
    max_steps=5000,
    lr=0.001,
    report_frequency=100,
    num_trees=75,
)


record = model.fit(X, y)
```

See nodegamlss/distributions for implemented distributions.
Note, that the visualizations are not yet prettified. Thus the visualizations are simply the NodeGAM visualization for each distributional parameter in order of the parameters as defined in the torch.distribution.

```python
fig, axes, df = model.visualize(X)
```

Visualize the loss and the evaluation metric simply via:


```python
plt.figure(figsize=[18, 6])
plt.subplot(1, 2, 1)
plt.plot(record['train_losses'])
plt.title('Training Loss')
plt.xlabel('Steps')
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(record['val_metrics'])
plt.title('Validation Metrics')
plt.xlabel('Steps in every 100 step')
plt.grid()
plt.show()
```