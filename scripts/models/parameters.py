import numpy as np


# TODO: Finish documentation
class Parameters:
    """
    A class representing parameters of multi-agent algorithms.

    Parameters
    ----------
    n: int
        Number of agents in multi-agent network.
    adj: np.matrix
        Adjacency matrix for agents.
    productivity: np.matrix
        Productivity matrix for agents.
    theta_hat: np.matrix
        Estimations?
    static_system: bool
        If system static (no new tasks during work) or not
    """
    n: int
    adj: np.matrix
    product: np.matrix
    theta_hat: np.matrix = np.matrix([[0], [0], [0]])
    neib_add: int
    add_neib_val: float
    static_system: bool

    params_dict: dict
