import numpy as np
from scripts.models.parameters import Parameters
from scripts.algorithms.accelerated_local_voting_protocol import AcceleratedLVP


def test_accelerated_local_voting_protocol():
    generate = True
    num_agents = 5
    productivities = [10] * num_agents
    num_steps = 20

    pars = Parameters()
    pars.n = num_agents
    pars.theta_hat = []
    Adj = np.array([
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0]
    ])
    pars.b = Adj / 2

    pars.neib_add = 5
    pars.add_neib_val = 0.3
    pars.static_system = False
    pars.params_dict = {
        "L": 7.1,
        "mu": 0.9,
        "h": 0.2,
        "eta": 0.8,
        "gamma": [[0.07, 0.09, 0.11][0]],
        "alpha": [0.07, 0.09, 0.11][1]
    }

    alg_lvp = AcceleratedLVP(params=pars)
    alg_lvp.run(num_steps=num_steps, generate=generate, productivities=productivities)
