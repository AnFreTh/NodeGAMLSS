import torch
import torch.distributions as dist
import numpy as np


class BaseDistribution:
    def __init__(self):
        self.num_params = None  # To be defined in each subclass

    def compute_loss(self, predictions, y_true):
        """
        Compute the distribution-specific loss.
        Subclasses must override this method.

        Args:
        - predictions (Tensor): The predicted parameters of the distribution with shape (n, num_params).
        - y_true (Tensor): The actual target values.

        Returns:
        - loss (Tensor): The computed loss.
        """
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
    def __init__(self):
        super().__init__()
        self.num_params = 2  # mean and log variance

    def compute_loss(self, predictions, y_true):
        y_pred_mean = predictions[:, 0]
        y_pred_logvar = predictions[:, 1]

        # Transform log variance to variance using Softplus
        var = torch.nn.functional.softplus(y_pred_logvar)

        # Define the normal distribution with the predicted mean and variance
        normal_dist = dist.Normal(y_pred_mean, var)

        # Compute the log probability of the actual values
        log_prob = normal_dist.log_prob(y_true)

        # The negative log-likelihood is the negative of the log probability
        nll_loss = -log_prob

        return nll_loss.mean()


class PoissonDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self.num_params = 1  # rate parameter (lambda)

    def compute_loss(self, predictions, y_true):
        # Assuming predictions contain the log of rate parameter (for stability)
        rate = torch.exp(predictions)
        poisson_dist = torch.distributions.Poisson(rate)
        log_prob = poisson_dist.log_prob(y_true)
        nll_loss = -log_prob
        return nll_loss.mean()


class GammaDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self.num_params = 2  # shape (k) and scale (θ)

    def compute_loss(self, predictions, y_true):
        # Ensure parameters are positive
        shape = torch.nn.functional.softplus(predictions[:, 0])
        scale = torch.nn.functional.softplus(predictions[:, 1])
        gamma_dist = torch.distributions.Gamma(shape, scale)
        log_prob = gamma_dist.log_prob(y_true)
        nll_loss = -log_prob
        return nll_loss.mean()


class NegativeBinomialDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self.num_params = (
            2  # mean and dispersion (or mean and inverse logits of dispersion)
        )

    def compute_loss(self, predictions, y_true):
        mean = predictions[:, 0]
        # Transform the second parameter to ensure the dispersion is positive
        dispersion = torch.nn.functional.softplus(predictions[:, 1])
        probs = mean / (dispersion + mean)

        neg_binomial_dist = dist.NegativeBinomial(total_count=dispersion, probs=probs)
        log_prob = neg_binomial_dist.log_prob(y_true)

        # The negative log-likelihood is the negative of the log probability
        nll_loss = -log_prob

        return nll_loss.mean()  # Return the mean loss over the batch


class StudentTDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self.num_params = 3  # mean, log-scale, and log-degrees of freedom

    def compute_loss(self, predictions, y_true):
        mean = predictions[:, 0]
        log_scale = predictions[:, 1]
        log_df = predictions[:, 2]

        # Ensure scale and degrees of freedom are positive
        scale = torch.nn.functional.softplus(log_scale)
        df = torch.nn.functional.softplus(log_df)

        # Define the Student's t-distribution with the predicted parameters
        student_t_dist = dist.StudentT(df, mean, scale)

        # Compute the log probability of the actual values
        log_prob = student_t_dist.log_prob(y_true)

        # The negative log-likelihood is the negative of the log probability
        nll_loss = -log_prob

        return nll_loss.mean()  # Return the mean loss over the batch


class InverseGammaDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self.num_params = 2  # log-shape and log-scale

    def compute_loss(self, predictions, y_true):
        log_shape = predictions[:, 0]
        log_scale = predictions[:, 1]

        # Ensure shape and scale are positive
        shape = torch.nn.functional.softplus(log_shape)
        scale = torch.nn.functional.softplus(log_scale)

        gamma_dist = dist.Gamma(shape, 1.0 / scale)  # Using Gamma distribution
        log_prob = gamma_dist.log_prob(
            1.0 / y_true
        )  # Transform y_true for Inverse Gamma

        log_prob_adj = log_prob - 2.0 * torch.log(y_true)

        # The negative log-likelihood is the negative of the adjusted log probability
        nll_loss = -log_prob_adj

        return nll_loss.mean()  # Return the mean loss over the batch
