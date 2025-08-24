from enum import StrEnum


class ListableEnum(StrEnum):
    """Base enum class with utility method to list all values.

    Extends StrEnum to provide a convenient method for getting all enum values as a list.
    """

    @classmethod
    def list(cls) -> list[str]:
        """Return a list of all enum values.

        Returns:
            list[str]: A list containing all enum values.
        """
        return [item.value for item in cls]


class ExperimentMetricType(ListableEnum):
    """Enumeration of experiment metric types.

    Defines the different types of metrics that can be calculated for experiments.
    """

    RATIO = "ratio"
    PROPORTION = "proportion"
    AVG = "avg"


class UserMetricType(ListableEnum):
    """Enumeration of user metric aggregation types.

    Defines the different aggregation methods for user-level metrics.
    """

    COUNT = "count"
    SUM = "sum"
    FLG = "flg"
    AVG = "avg"


class ExperimentMode(ListableEnum):
    """Enumeration of experiment operation modes.

    Defines the available modes for experiment operations.
    """

    CREATE = "Create New Experiment"
    UPDATE = "Update Existing Experiment"


class ExperimentStatus(ListableEnum):
    """Enumeration of experiment statuses.

    Defines the lifecycle states of an experiment.
    """

    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class ComputationLevel(ListableEnum):
    """Levels at which computations can be performed."""

    EXPERIMENT = "experiment"
    USER = "user"


class JobStatus(ListableEnum):
    """Possible states of a calculation job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CalculationPurpose(ListableEnum):
    """Different purposes for which calculations can be performed."""

    PLANNING = "planning"
    REGULAR = "regular"


class PageMode(ListableEnum):
    """Different modes for page interactions."""

    LIST = "List"
    CREATE = "Create"
    UPDATE = "Update"
