from dataclasses import dataclass, field
from enum import Enum

from hydra_zen import make_custom_builds_fn


class ClusterMode(Enum):
    dev = "DEV"
    prod = "PROD"


class LocalMode(Enum):
    shell = "SHELL"
    cluster = "CLUSTER"


class ExecutionLocation(Enum):
    local = "LOCAL"
    remote = "REMOTE"


@dataclass
class ClusterConfig:
    mode: ClusterMode = field(default_factory=lambda: ClusterMode.dev)


@dataclass
class LocalConfig:
    mode: LocalMode = field(default_factory=lambda: LocalMode.shell)
    cluster_config: ClusterConfig = field(default_factory=ClusterConfig)


@dataclass
class ExecutionMode:
    """
    Constructs configurations for each leaf node marked with a `#` in the supported
    execution config tree:

        execution_config = {
            "LOCAL": {
                "SHELL": "LOCAL_SHELL", #
                "CLUSTER": {
                    "DEV": "LOCAL_CLUSTER_DEV", #
                    "PROD": "LOCAL_CLUSTER_PROD" #
                }
            },
            "REMOTE": {
                "DEV": "REMOTE_DEV", #
                "PROD": "REMOTE_PROD" #
            }
        }
    """

    location: ExecutionLocation = field(
        default_factory=lambda: ExecutionLocation.remote
    )
    local_config: LocalConfig = field(default_factory=LocalConfig)
    remote_config: ClusterConfig = field(default_factory=ClusterConfig)


fbuilds = make_custom_builds_fn(populate_full_signature=True)
ClusterConfigConf = fbuilds(ClusterConfig)
LocalConfigConf = fbuilds(LocalConfig)
ExecutionModeConf = fbuilds(ExecutionMode)


# Default Execution Configuration
default_execution_config = ExecutionModeConf()

# Local Shell Configuration
local_shell_config = ExecutionModeConf(
    location=ExecutionLocation.local,
    local_config=LocalConfigConf(
        mode=LocalMode.shell,
        cluster_config=None,
    ),
    remote_config=None,
)

# Local Cluster Dev Configuration
local_cluster_dev_config = ExecutionModeConf(
    location=ExecutionLocation.local,
    local_config=LocalConfigConf(
        mode=LocalMode.cluster,
        cluster_config=ClusterConfigConf(mode=ClusterMode.dev),
    ),
    remote_config=None,
)

# Local Cluster Prod Configuration
local_cluster_prod_config = ExecutionModeConf(
    location=ExecutionLocation.local,
    local_config=LocalConfigConf(
        mode=LocalMode.cluster,
        cluster_config=ClusterConfigConf(mode=ClusterMode.prod),
    ),
    remote_config=None,
)

# Remote Dev Configuration
remote_dev_config = ExecutionModeConf(
    location=ExecutionLocation.remote,
    local_config=None,
    remote_config=ClusterConfigConf(mode=ClusterMode.dev),
)

# Remote Prod Configuration
remote_prod_config = ExecutionModeConf(
    location=ExecutionLocation.remote,
    local_config=None,
    remote_config=ClusterConfigConf(mode=ClusterMode.prod),
)

if __name__ == "__main__":
    from pprint import pprint

    from hydra_zen import instantiate

    def ipprint(x):
        pprint(instantiate(x))

    ipprint(default_execution_config)
    ipprint(local_shell_config)
    ipprint(local_cluster_dev_config)
    ipprint(local_cluster_prod_config)
    ipprint(remote_dev_config)
    ipprint(remote_prod_config)

"""
Permissively typed version of the dataclasses above for debugging purposes

@dataclass
class ClusterConfig:
    mode: Any


@dataclass
class LocalConfig:
    mode: Any
    cluster_config: Any = MISSING


@dataclass
class ExecutionMode:
    location: Any
    local_config: Any = MISSING
    remote_config: Any = MISSING
"""
