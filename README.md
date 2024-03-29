
## Node-GAMLSS: Differentiable Additive Model for Interpretable Distributional Deep Learning

Node-GAMLSS integrates the Node-GAM framework and the NAMLSS/GAMLSS framework to train Generalized Additive Models for Location, Scale, and Shape (GAMLSS) using multi-layer differentiable trees. This model minimizes the negative log-likelihood for a variety of distributions, with built-in support for handling distributional parameter constraints. The implementation adapts the Node-GAM approach to accommodate distributional regression, making it suitable for deep learning applications that require interpretability. In short, it trains a GAMLSS model by multi-layer differentiable trees and minimizes the negative log-likelihood of a given distribution.


Relevant Papers:
- [Node-GAM: Differentiable Generalized Additive Model for Interpretable Deep Learning](https://arxiv.org/abs/2106.01613)
- [Neural Additive Models for Location Scale and Shape: A Framework for Interpretable Neural Regression Beyond the Mean](https://arxiv.org/pdf/2301.11862.pdf).

The codebase is inspired by and adapted from the [Node-GAM Github repository](https://github.com/zzzace2000/nodegam/tree/main).

### Installation
Install Node-GAMLSS directly from the repository using pip:
```bash
pip install git+https://github.com/AnFreTh/NodeGAMLSS.git
```

## NodeGAM Training

### Sklearn interface

To simply minimize the negative log-likelihood of a normal distribution on a given dataset just use:
```python
from nodegamlss.model import NodeGAMLSS

model = NodeGAMLSS(
    in_features=X.shape[1],
    family="normal",
    device="cpu", #or "cuda"
    verbose=False,
    max_steps=5000,
    lr=0.001,
    report_frequency=100,
    num_trees=75,
)


record = model.fit(X, y)
```
The "family" parameter defines the distribution that is used.

See nodegamlss/distributions for details regarding the distributions and their parameters.
So far, the following distributions are implemented. Feel free to raise an Issue when crucial distributions are missing:
1. Normal 
2. Poisson 
3. Gamma
4. Inverse Gamma
5. Dirichlet
6. Beta
7. StudentT
8. Negative Binomial 
9.  Categorical

After fitting, the individual feature and parameter effects are easily visualizable.
There are multiple ways to visualize the model.
First, we can simply leverage the Node-GAM visualizations and create a Node-GAM-style plot for each parameter.

```python
fig, axes, df = model.visualize_nodegam(X)
```
This will create as many plots as you have distributional parameters and then create for each feature and feature interaction a subplot.

Additionally, you can create dash plots for each distributional parameter. E.g. for a normal distribution, you can visualize the mean predictions via:
```python
model.plot_single_feature_effects(X, parameter="mean")
```

And the variance via:
```python
model.plot_single_feature_effects(X, parameter="variance")
```

The interaction plots are similarly created via:
```python
model.plot_interaction_effects(X, port=8051, parameter="mean")
```

Furthermore, it is similar to Node-GAM easy and user-friendly to visualize the loss and the evaluation metric simply via:
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