class Task:
    """
    A class representing a task in a multi-agent system.

    Parameters
    ----------
    step: int
        The step at which the task appeared.
    complexity: int
        Task's complexity (number of steps needed to complite the task).
    completed_step: int
        Step at which the task was completed.
    """

    def __init__(self, step=None, complexity=0):
        """
        Initializes the task.
        """
        self.step = step
        self.complexity = complexity
        self.completed_step = None

    def to_dict(self):
        """
        Converts the task to dictionary.
        """
        return {
            "step": self.step,
            "complexity": self.complexity,
            "completed_step": self.completed_step
        }
