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

To simply use it on your dataset, just run:
```python
from nodegamlss.sklearn import NodeGAMLSS

model = NodeGAMLSS( 
    in_features=X.shape[1],
    objective="LSS",
    family="normal",
    )

model.fit(X, y)
```