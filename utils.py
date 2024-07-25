import torch
import os

def download_weights(model_capacity):
    try:
        from urllib.request import urlretrieve
    except ImportError:
        from urllib import urlretrieve

    weight_file = 'crepe-{}.pth'.format(model_capacity)
    base_url = 'https://github.com/sweetcocoa/crepe-pytorch/raw/models/'

    # In all other cases, decompress the weights file if necessary
    package_dir = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(package_dir, weight_file)
    if not os.path.isfile(weight_path):
        print('Downloading weight file {} from {} ...'.format(weight_path, base_url + weight_file))
        urlretrieve(base_url + weight_file, weight_path)

def to_local_average_cents(salience, center=None):
    """
    Find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # The bin number-to-cents mapping
        to_local_average_cents.cents_mapping = (
                torch.linspace(0, 7180, 360, dtype=salience.dtype, device=salience.device) + 1997.3794084376191)

    if salience.ndim == 1:
        if center is None:
            center = int(torch.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience_segment = salience[start:end]
        mapping_segment = to_local_average_cents.cents_mapping[start:end]
        product_sum = torch.sum(salience_segment * mapping_segment)
        weight_sum = torch.sum(salience_segment)
        return product_sum / weight_sum
    elif salience.ndim == 2:
        return torch.stack([to_local_average_cents(salience[i, :]) for i in range(salience.shape[0])])

    raise Exception("Label should be either 1D or 2D Tensor")

def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch continuity.

    Note: This is NOT implemented with PyTorch.
    """
    from hmmlearn import hmm
    import numpy as np

    # Uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # Transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # Emission probability = fixed probability for self, evenly distribute the others
    self_emission = 0.1
    emission = (np.eye(360) * self_emission + np.ones((360, 360)) * ((1 - self_emission) / 360))

    # Fix the model parameters because we are not optimizing the model
    model = hmm.MultinomialHMM(n_components=360)
    model.startprob_ = starting
    model.transmat_ = transition
    model.emissionprob_ = emission

    # Find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), lengths=[len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in range(len(observations))])

def to_freq(activation, viterbi=False):
    if viterbi:
        cents = to_viterbi_cents(activation.detach().cpu().numpy())
        cents = torch.tensor(cents, dtype=torch.float32, device=activation.device)
    else:
        cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[torch.isnan(frequency)] = 0
    return frequency
