import numpy as np

from scripts.algorithms.abstract import LbAlgorithm
from scripts.models.parameters import Parameters
from scripts.models.task_pool import TaskPool, UNIFORM


class RoundRobin(LbAlgorithm):
    def __init__(self, params: Parameters, **kwargs):
        """
        :param parameters: parameters of class Parameters
        """
        super().__init__(params, **kwargs)

        # Algorithm parameter
        self.h = params.params_dict.get("h", None)

    # --------------------------- Prepare before run ---------------------------------

    def create_task_pool(self, num_steps: int, generate: bool):
        return TaskPool(self.n, num_steps, generate, distribute=False, distribution=UNIFORM)

    def prepare(self, generate, neigh_file, num_steps):
        pass

    # --------------------------- Extract step info ----------------------------------

    def extract_step_info(self, step: int) -> None:
        """
        Extract information about neighbours and productivity according to the step
        :param step:
        """
        for agent in self.agents:
            agent.produc = agent.prods[step]

    # --------------------------- Algorithm step -------------------------------------

    def algorithm_step(self, is_logging, step):
        # Compute local voting protocol
        self.theta_hat = np.matrix([[agent.theta_hat] for agent in self.agents])
        self.log(is_logging, f"Current distribution on {step} step: {self.theta_hat}")

        for agent in self.agents:
            agent.update_theta_hat()
            self.log(is_logging, f"Agent {agent.id}: {agent.theta_hat}")
