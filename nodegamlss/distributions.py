import torch
import torch.distributions as dist
import numpy as np


import torch
import torch.distributions as dist


class BaseDistribution:
    def __init__(self, name, param_names):
        self._name = name
        self.param_names = param_names
        self.param_count = len(param_names)
        # Predefined transformation functions accessible to all subclasses
        self.predefined_transforms = {
            "positive": torch.nn.functional.softplus,
            "none": lambda x: x,
            "square": lambda x: x**2,
            "exp": torch.exp,
            "sqrt": torch.sqrt,
            "probabilities": torch.softmax,
            "log": lambda x: torch.log(
                x + 1e-6
            ),  # Adding a small constant for numerical stability
        }

    @property
    def name(self):
        return self._name

    @property
    def parameter_count(self):
        return self.param_count

    def get_transform(self, transform_name):
        """
        Retrieve a transformation function by name, or return the function if it's custom.
        """
        if callable(transform_name):
            # Custom transformation function provided
            return transform_name
        return self.predefined_transforms.get(
            transform_name, lambda x: x
        )  # Default to 'none'

    def compute_loss(self, predictions, y_true):
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate_nll(self, y_true, y_pred):

        # Convert numpy arrays to torch tensors
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

        # Compute NLL using the provided loss function
        nll_loss_tensor = self.compute_loss(y_true_tensor, y_pred_tensor)

        # Convert the NLL loss tensor back to a numpy array and return
        return nll_loss_tensor.detach().numpy()


class NormalDistribution(BaseDistribution):
    def __init__(self, name="Normal", mean_transform="none", var_transform="positive"):
        param_names = [
            "mean",
            "variance",
        ]
        super().__init__(name, param_names)

        self.mean_transform = self.get_transform(mean_transform)
        self.var_transform = self.get_transform(var_transform)

    def compute_loss(self, predictions, y_true):
        mean = self.mean_transform(predictions[:, self.param_names.index("mean")])
        variance = self.var_transform(
            predictions[:, self.param_names.index("variance")]
        )

        normal_dist = dist.Normal(mean, variance)

        nll = -normal_dist.log_prob(y_true).mean()
        return nll


class PoissonDistribution(BaseDistribution):
    def __init__(self, name="Poisson", rate_transform="positive"):
        param_names = ["rate"]  # Specify parameter name for Poisson distribution
        super().__init__(name, param_names)
        # Retrieve transformation function for rate
        self.rate_transform = self.get_transform(rate_transform)

    def compute_loss(self, predictions, y_true):
        rate = self.rate_transform(predictions[:, self.param_names.index("rate")])

        # Define the Poisson distribution with the transformed parameter
        poisson_dist = dist.Poisson(rate)

        # Compute the negative log-likelihood
        nll = -poisson_dist.log_prob(y_true).mean()
        return nll


class InverseGammaDistribution(BaseDistribution):
    def __init__(
        self,
        name="InverseGamma",
        shape_transform="positive",
        scale_transform="positive",
    ):
        param_names = [
            "shape",
            "scale",
        ]
        super().__init__(name, param_names)

        self.shape_transform = self.get_transform(shape_transform)
        self.scale_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):
        shape = self.shape_transform(predictions[:, self.param_names.index("shape")])
        scale = self.scale_transform(predictions[:, self.param_names.index("scale")])

        inverse_gamma_dist = dist.InverseGamma(shape, scale)
        # Compute the negative log-likelihood
        nll = -inverse_gamma_dist.log_prob(y_true).mean()
        return nll


class BetaDistribution(BaseDistribution):
    def __init__(
        self,
        name="Beta",
        shape_transform="positive",
        scale_transform="positive",
    ):
        param_names = [
            "alpha",
            "beta",
        ]
        super().__init__(name, param_names)

        self.alpha_transform = self.get_transform(shape_transform)
        self.beta_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):
        alpha = self.alpha_transform(predictions[:, self.param_names.index("alpha")])
        beta = self.beta_transform(predictions[:, self.param_names.index("beta")])

        beta_dist = dist.Beta(alpha, beta)
        # Compute the negative log-likelihood
        nll = -beta_dist.log_prob(y_true).mean()
        return nll


class DirichletDistribution(BaseDistribution):
    def __init__(self, name="Dirichlet", concentration_transform="positive"):
        # For Dirichlet, param_names could be dynamically set based on the dimensionality of alpha
        # For simplicity, we're not specifying individual names for each concentration parameter
        param_names = ["concentration"]  # This is a simplification
        super().__init__(name, param_names)
        # Retrieve transformation function for concentration parameters
        self.concentration_transform = self.get_transform(concentration_transform)

    def compute_loss(self, predictions, y_true):
        # Apply the transformation to ensure all concentration parameters are positive
        # Assuming predictions is a 2D tensor where each row is a set of concentration parameters for a Dirichlet distribution
        concentration = self.concentration_transform(predictions)

        dirichlet_dist = dist.Dirichlet(concentration)

        nll = -dirichlet_dist.log_prob(y_true).mean()
        return nll


class GammaDistribution(BaseDistribution):
    def __init__(
        self, name="Gamma", shape_transform="positive", rate_transform="positive"
    ):
        param_names = ["shape", "rate"]
        super().__init__(name, param_names)

        self.shape_transform = self.get_transform(shape_transform)
        self.rate_transform = self.get_transform(rate_transform)

    def compute_loss(self, predictions, y_true):

        shape = self.shape_transform(predictions[:, self.param_names.index("shape")])
        rate = self.rate_transform(predictions[:, self.param_names.index("rate")])

        # Define the Gamma distribution with the transformed parameters
        gamma_dist = dist.Gamma(shape, rate)

        # Compute the negative log-likelihood
        nll = -gamma_dist.log_prob(y_true).mean()
        return nll


class StudentTDistribution(BaseDistribution):
    def __init__(
        self,
        name="StudentT",
        df_transform="positive",
        loc_transform="none",
        scale_transform="positive",
    ):
        param_names = ["df", "loc", "scale"]
        super().__init__(name, param_names)

        self.df_transform = self.get_transform(df_transform)
        self.loc_transform = self.get_transform(loc_transform)
        self.scale_transform = self.get_transform(scale_transform)

    def compute_loss(self, predictions, y_true):

        df = self.df_transform(predictions[:, self.param_names.index("df")])
        loc = self.loc_transform(predictions[:, self.param_names.index("loc")])
        scale = self.scale_transform(predictions[:, self.param_names.index("scale")])

        student_t_dist = dist.StudentT(df, loc, scale)

        nll = -student_t_dist.log_prob(y_true).mean()
        return nll


class NegativeBinomialDistribution(BaseDistribution):
    def __init__(
        self,
        name="NegativeBinomial",
        mean_transform="positive",
        dispersion_transform="positive",
    ):
        param_names = ["mean", "dispersion"]
        super().__init__(name, param_names)

        self.mean_transform = self.get_transform(mean_transform)
        self.dispersion_transform = self.get_transform(dispersion_transform)

    def compute_loss(self, predictions, y_true):
        # Apply transformations to ensure mean and dispersion parameters are positive
        mean = self.mean_transform(predictions[:, self.param_names.index("mean")])
        dispersion = self.dispersion_transform(
            predictions[:, self.param_names.index("dispersion")]
        )

        # Calculate the probability (p) and number of successes (r) from mean and dispersion
        # These calculations follow from the mean and variance of the negative binomial distribution
        # where variance = mean + mean^2 / dispersion
        r = 1 / dispersion
        p = r / (r + mean)

        # Define the Negative Binomial distribution with the transformed parameters
        negative_binomial_dist = dist.NegativeBinomial(total_count=r, probs=p)

        # Compute the negative log-likelihood
        nll = -negative_binomial_dist.log_prob(y_true).mean()
        return nll


class CategoricalDistribution(BaseDistribution):
    def __init__(self, name="Categorical", prob_transform="probabilites"):
        param_names = ["probs"]  # Specify parameter name for Poisson distribution
        super().__init__(name, param_names)
        # Retrieve transformation function for rate
        self.probs_transform = self.get_transform(prob_transform)

    def compute_loss(self, predictions, y_true):
        probs = self.probs_transform(predictions[:, self.param_names.index("probs")])

        # Define the Poisson distribution with the transformed parameter
        cat_dist = dist.Categorical(probs)

        # Compute the negative log-likelihood
        nll = -cat_dist.log_prob(y_true).mean()
        return nll
