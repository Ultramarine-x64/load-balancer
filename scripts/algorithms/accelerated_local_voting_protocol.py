import numpy as np

from scripts.algorithms.local_voting_protocol import LocalVotingProtocol
from scripts.jobs.alvp_parallel import AlvpParallel
from scripts.models.parameters import Parameters

DEFAULT_CACHE_PATH = "/cache/"
DEFAULT_LOGGS_PATH = DEFAULT_CACHE_PATH + "logs/"
DEFAULT_NEIGH_FILE = DEFAULT_CACHE_PATH + "alg_params/neigh.pkl"
DEFAULT_NOISE_FILE = DEFAULT_CACHE_PATH + "alg_params/noise.pkl"

NOISE_AVG = 0
NOISE_DISTR = 10


class AcceleratedLVP(LocalVotingProtocol):
    def __init__(self, params: Parameters, **kwargs):
        """
        :param parameters: parameters of class Parameters
        """
        super().__init__(params, **kwargs)

        # Accelerated local voting protocol variables
        self.nesterov_step = np.array([[0]] * self.n)
        self.L = params.params_dict.get("L", None)
        self.mu = params.params_dict.get("mu", None)
        self.eta = params.params_dict.get("eta", None)
        self.gamma = params.params_dict.get("gamma", [])
        self.alpha = params.params_dict.get("alpha", None)

        # Technical variables
        self.alvp_parallel = AlvpParallel(self.alpha, self.mu, self.eta, self.h, self.L)

    # --------------------------- Algorithm step -------------------------------------

    def compute_change(self, step):
        u_lvp = self.acc_local_voting_protocol(self.theta_hat, step)
        return u_lvp.round()

    def acc_local_voting_protocol(self, x: np.array, step) -> np.array:
        self.gamma = [self.gamma[-1]]
        self.gamma.append((1 - self.alpha) * self.gamma[0] + self.alpha * (self.mu - self.eta))

        params_list = []
        for agent_id in range(self.n):
            params_list.append(
                (
                    agent_id,
                    (x.T + self.noise_mat[step, agent_id]).T,
                    self.nesterov_step.item((agent_id, 0)),
                    self.gamma,
                    self.D[agent_id],
                    self.b[agent_id],
                    self.loggs_path,)
            )
        response = self.alvp_parallel.run(args_list=params_list)

        self.nesterov_step = AlvpParallel.extract_response_to_array(response, "nesterov_step", range(self.n))

        return AlvpParallel.extract_response_to_array(response, "x", range(self.n))
