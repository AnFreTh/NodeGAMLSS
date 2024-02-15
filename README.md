# NODE GAM: Differentiable Generalized Additive Model for Interpretable Deep Learning: 

NodeGAMLSS is an interpretable deep distributional learning GAM model based on the Node-GAM Paper and the NAMLSS Paper: 
[NODE GAM: Differentiable Generalized Additive Model for Interpretable Deep Learning](https://arxiv.org/abs/2106.01613)
[Neural Additive Models for Location Scale and Shape: A Framework for Interpretable Neural Regression Beyond the Mean](https://arxiv.org/pdf/2301.11862.pdf).
In short, it trains a GAMLSS model by multi-layer differentiable trees to be accurate, interpretable, and 
differentiable. The code is taken from the [Node-GAM Github](https://github.com/zzzace2000/nodegam/tree/main) implementation and adjusted to account for disrtibutional regression approaches.

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
    in_features=3,
    objective="mse",
    family="normal",
    device="cpu",
    verbose=False,
    problem="LSS",
    max_steps=300,
    lr=0.0001,
    num_trees=25,
    l2_lambda=0.01
)


record = model.fit(X, y)
```

See nodegamlss/distributions for implemented distributions.
Note, that the visualizations are not yet implemented for most distributions and are still a work in progress.