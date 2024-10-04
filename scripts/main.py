from scripts.algorithms.accelerated_local_voting_protocol import AcceleratedLVP
from scripts.algorithms.local_voting_protocol import LocalVotingProtocol
from scripts.algorithms.round_robin import RoundRobin

LVP = "LVP"
ALVP = "ALVP"
ROUND_ROBIN = "ROUND_ROBIN"

METHODS_TO_CLASSES = {
    LVP: LocalVotingProtocol,
    ALVP: AcceleratedLVP,
    ROUND_ROBIN: RoundRobin
}


def run_load_balancing(method, num_steps, params, generate, productivities):
    load_balancing = METHODS_TO_CLASSES[method](params=params)
    load_balancing.run(num_steps=num_steps, generate=generate, productivities=productivities)
    return load_balancing
