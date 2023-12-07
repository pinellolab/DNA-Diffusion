import importlib
import inspect
import logging
import os
import pkgutil
import queue
import re
import secrets
import sys
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from textwrap import dedent
from typing import Any, Dict, List, Tuple, Union

import plumbum
from dataclasses_json import dataclass_json
from flytekit import WorkflowExecutionPhase
from flytekit.core.base_task import PythonTask
from flytekit.core.workflow import WorkflowBase
from flytekit.exceptions.system import FlyteSystemException
from flytekit.exceptions.user import FlyteTimeout
from flytekit.remote import FlyteRemote
from flytekit.remote.executions import FlyteWorkflowExecution
from hydra.conf import HelpConf, HydraConf, JobConf
from hydra_zen import ZenStore, builds, make_custom_builds_fn


@dataclass_json
@dataclass
class EntityConfig:
    inputs: Dict[str, Any]
    module_name: str = "lrwine"
    entity_name: str = "training_workflow"
    entity_type: str = "WorkflowBase"


# @dataclass_json
# @dataclass
# class EntityConfigs:
#     entities: Dict[str, Any] = MISSING


fbuilds = make_custom_builds_fn(populate_full_signature=True)


def generate_entity_configs(
    parent_module_path: str, entity_store: ZenStore, logger: logging.Logger
) -> None:
    """
    Generates and stores configurations for entities found in a specified
    module.

    This function iterates over all submodules in the parent module defined by
    `parent_module_path`. For each submodule, it looks for entities that are
    instances of either `WorkflowBase` or `PythonTask` and generates
    configurations for them using `EntityConfig`. These configurations are then
    stored in `entity_store`.

    Args:
        parent_module_path (str): The import path of the parent module to search
        for entities.
        entity_store (ZenStore): The store where generated entity configurations
        will be kept.
        logger (logging.Logger): Logger for debugging and logging information.

    Returns:
        None: The function does not return anything. It populates the
        `entity_store` with configurations for each entity of type EntityTypes.
    """
    parent_module = importlib.import_module(parent_module_path)
    EntityTypes = (WorkflowBase, PythonTask)

    # iterate over submodules in the parent module
    for submodule_info in pkgutil.iter_modules(
        parent_module.__path__, parent_module.__name__ + "."
    ):
        # import the submodule
        submodule = importlib.import_module(submodule_info.name)
        logger.debug(f"Checking submodule: {submodule_info.name}")

        # import entities that are instances of EntityTypes
        entities = inspect.getmembers(
            submodule,
            # TODO: validate that PythonTasks function as expected
            lambda member: isinstance(member, EntityTypes),
            # lambda member: isinstance(member, WorkflowBase),
        )

        for entity_name, entity in entities:
            logger.debug(f"Found entity: {entity_name}")

            # construct an instance (or a configuration) of the entity
            module_name = submodule_info.name.split(".")[-1]
            entity_inputs = generate_entity_inputs(entity)
            entity_instance = fbuilds(
                EntityConfig,
                inputs=builds(dict, entity_inputs, hydra_convert="all"),
                module_name=module_name,
                entity_name=entity_name,
                entity_type=type(entity).__name__,
            )

            # store the entity instance in the entity_store
            composed_name = module_name + "_" + entity_name
            entity_store(entity_instance, name=composed_name)
            logger.debug(f"Stored entity: {composed_name} in entity_store")


def generate_entity_inputs(
    entity: Union[WorkflowBase, PythonTask],
) -> Dict[str, Any]:
    """
    Generates a dictionary of inputs for a given entity.

    This function inspects the signature of the provided `entity`, which can be
    either a `WorkflowBase` or a `PythonTask`. For each parameter in the
    signature, it determines the type and default value (if any). If the type is
    a built-in type, it directly uses the default value. For custom types, it
    dynamically imports and constructs a configuration object using `fbuilds`.

    Args:
        entity (Union[WorkflowBase, PythonTask]): The entity for which to
        generate input configurations.

    Returns:
        Dict[str, Any]: A dictionary with keys that are the names of the entity
        inputs and values hydra-zen configurations that will build their
        respective default values.
    """
    inputs = {}

    for name, param in inspect.signature(entity).parameters.items():
        param_type = param.annotation
        default = param.default

        # check if the type is a built-in type
        if isinstance(param_type, type) and param_type.__module__ == "builtins":
            inputs[name] = default
        else:
            # dynamically import the type if it's not a built-in type
            type_module = importlib.import_module(param_type.__module__)
            custom_type = getattr(type_module, param_type.__name__)

            inputs[name] = fbuilds(custom_type)

    return inputs


def random_alphanumeric_suffix(input_string: str = "", length: int = 3) -> str:
    return input_string.join(
        secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789")
        for _ in range(length)
    )


def check_required_env_vars(
    required_vars: List[str], logger: logging.Logger
) -> bool:
    """
    Checks required environment variables for workflow configuration.
    """

    missing_vars = [var for var in required_vars if os.environ.get(var) is None]
    if missing_vars:
        missing_vars_str = ", ".join(missing_vars)
        logger.error(
            f"Missing required environment variables: {missing_vars_str}"
        )
        return False
    return True


def git_info_to_workflow_version(
    logger: logging.Logger,
) -> Tuple[str, str, str]:
    """
    Retrieves git information for workflow versioning using plumbum.

    This function extracts repository name, current branch name, and short SHA.
    It handles the case where the Git repository is in a detached HEAD state,
    common in CI environments like GitHub Actions for pull requests.

    Args:
        logger (logging.Logger): Logger object for logging messages.

    Returns:
        Tuple[str, str, str]: A tuple containing the repository name,
                               branch name, and short SHA commit.

    Raises:
        ValueError: If unable to extract source commit SHA from commit message.
        ProcessExecutionError: If a git command fails.

    Example:
        >>> import logging
        >>> logger = logging.getLogger()
        >>> # Assuming this test is run in a Git repository
        >>> repo_name, branch, short_sha = git_info_to_workflow_version(logger)
        >>> print(isinstance(repo_name, str), isinstance(branch, str), isinstance(short_sha, str))
        True True True
    """
    try:
        git = plumbum.local["git"]
        git_branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()

        if git_branch.lower() == "head":
            git("fetch", "origin", "+refs/heads/*:refs/remotes/origin/*")

            commit_message = git("log", "-1", "--pretty=%B")
            match = re.search(r"Merge ([0-9a-f]{40}) into", commit_message)
            if match:
                source_commit_sha = match.group(1)

                git_branch_list = (
                    git("branch", "-r", "--contains", source_commit_sha)
                    .strip()
                    .split("\n")
                )
                git_branch = (
                    next(
                        (
                            branch
                            for branch in git_branch_list
                            if "HEAD" not in branch
                        ),
                        "",
                    )
                    .replace("origin/", "")
                    .strip()
                )
            else:
                git_branch_from_detached_head_failure = (
                    "Unable to extract source commit SHA from commit message."
                )
                raise ValueError(git_branch_from_detached_head_failure)

        git_short_sha = git("rev-parse", "--short", "HEAD").strip()
        remote_url = git("config", "--get", "remote.origin.url").strip()
        repo_name = remote_url.split("/")[-1].rstrip(".git")

        for string in [repo_name, git_branch, git_short_sha]:
            if any(char.isupper() for char in string):
                logger.warning(
                    f"String '{string}' contains capitalized characters. Converting to lowercase."
                )

        return repo_name.lower(), git_branch.lower(), git_short_sha.lower()

    except plumbum.commands.processes.ProcessExecutionError as e:
        logger.error(f"Error obtaining git information: {e}")
        raise


def generate_hydra_config() -> HydraConf:
    return HydraConf(
        defaults=[
            {"output": "default"},
            {"launcher": "basic"},  # joblib
            {"sweeper": "basic"},
            {"help": "default"},
            {"hydra_help": "default"},
            {"hydra_logging": "none"},  # default
            {"job_logging": "none"},  # default
            {"callbacks": None},
            {"env": "default"},
        ],
        help=HelpConf(
            header=dedent(
                """
                This is the ${hydra.help.app_name} help accessible via `${hydra.help.app_name} -h`.

                Use `${hydra.help.app_name} -c job` to view the ${hydra.help.app_name} configuration alone.
                See the end of this help page for instructions on how to install shell tab completion for
                configuration overrides.

                ${hydra.help.app_name} is the CLI of a template designed to illustrate the integration of:

                  * hydra-zen (https://mit-ll-responsible-ai.github.io/hydra-zen/),
                  * hydra (https://hydra.cc/), and
                  * omegaconf (https://omegaconf.readthedocs.io/),

                which provide configuration management, with

                  * flyte(kit) (https://flyte.org/),

                which manages the registration and execution of Flyte workflows.
                ${hydra.help.app_name} can be adapted as an auxiliary component of any python package,
                enhancing its capabilities in managing complex workflow configuration
                and execution.

                Running `${hydra.help.app_name} -c job` displays the current configuration of ${hydra.help.app_name}.
                This reflects what will be executed if `dnadiffusion` is run without arguments.

                Use `${hydra.help.app_name} -c hydra` to view the associated hydra configuration.

                """
            ),
            footer=dedent(
                """
                You can test CLI configuration overrides after `-c job`, e.g.:

                  * `${hydra.help.app_name} -c job execution_context=prod`
                  * `${hydra.help.app_name} -c job entity_config==example_wf`
                  # The inputs arguments in the following example must correspond
                  # to those supported by the specified entity_config.
                  * `${hydra.help.app_name} -c job \\
                       entity_config=lrwine_training_workflow \\
                       entity_config.inputs._args_.0.logistic_regression.max_iter=1200`
                  # The following two examples are only meant for task testing.
                  # This example only works in the local_shell execution context.
                  # See: https://github.com/flyteorg/flyte/issues/4275
                  #      https://github.com/flyteorg/flyte/issues/1312
                  * `${hydra.help.app_name} -c job \\
                       execution_context=local_shell \\
                       entity_config=lrwine_process_data \\
                       entity_config.inputs._args_=[]"`
                  # This will fail without specifying the inputs as it does not
                  # automatically instantiate the default arguments of the task.
                  * `${hydra.help.app_name} -c job \\
                       execution_context=local_cluster_dev \\
                       entity_config=lrwine_process_data \\
                       entity_config.inputs._args_.0.data.data="[[12.0, 0],[13.0, 1],[9.5, 2]]" \\
                       entity_config.inputs._args_.0.data.columns="[ash, target]"`


                This will generate `== Config ==` above resolved in context of the command line overrides.
                Removing the `-c job` flag will execute the workflow with the specified configuration.
                The resolved configuration will be stored in the `outputs` or `multirun` directories.

                Use `${hydra.help.app_name} --hydra-help` to view the hydra
                help. This contains, for example, the commands to install shell
                tab completion. For example in bash or zsh, if the active
                configuration has path
                `entity_config.inputs._args_.0.logistic_regression`
                representing the parameters of a
                sklearn.linear_model.LogisticRegression instance:

                > eval "$$(dnadiffusion -sc install=bash)"
                > dnadiffusion entity_config.inputs._args_.0.logistic_regression.[TAB]
                entity_config.inputs._args_.0.logistic_regression.C=
                entity_config.inputs._args_.0.logistic_regression._target_=
                entity_config.inputs._args_.0.logistic_regression.class_weight=
                entity_config.inputs._args_.0.logistic_regression.dual=
                ..."""
            ),
            template=dedent(
                """
                ${hydra.help.header}
                == Configuration groups ==
                First override default group values (group=option)

                $APP_CONFIG_GROUPS

                == Config ==
                Then override any element in the config (foo.bar=value)
                that is not set exclusively by an environment variable [see doc(strings)]

                $CONFIG
                ${hydra.help.footer}
                """
            ),
        ),
        job=JobConf(name="dnadiffusion"),
    )


def get_user_input(input_queue):
    """
    Gets user input and puts it in the queue.
    """
    user_input = input("Terminate workflow execution? (y/N after 1 min.): ")
    input_queue.put(user_input)


def wait_for_workflow_completion(
    execution: FlyteWorkflowExecution,
    remote: FlyteRemote,
    logger: logging.Logger,
) -> None:
    """
    Waits for the execution to complete, checking status at regular intervals.
    """
    timeout_duration = timedelta(seconds=3.0)
    synced_execution = None
    try:
        while True:
            try:
                completed_execution = remote.wait(
                    execution, timeout=timeout_duration
                )
                logger.info(f"Execution completed:\n\n{completed_execution}\n")
                if completed_execution.error is None:
                    break
                else:
                    logger.error(
                        f"Execution failed with error:\n\n{completed_execution.error}\n"
                    )
                    sys.exit(1)
            except FlyteTimeout:
                synced_execution = remote.sync(execution)
                logger.info(f"Current status:\n\n{synced_execution}\n")
                time.sleep(timeout_duration.total_seconds())
    except KeyboardInterrupt:
        if synced_execution is not None:
            logger.info(f"Status at KeyboardInterrupt:\n\n{synced_execution}\n")
        else:
            logger.info(
                "KeyboardInterrupt caught before execution status sync."
            )

        input_queue = queue.Queue()
        input_thread = threading.Thread(
            target=get_user_input, args=(input_queue,)
        )
        input_thread.daemon = True
        input_thread.start()

        try:
            response = input_queue.get(timeout=60)
            response = response.strip().lower()
        except queue.Empty:
            response = "n"

        synced_execution = remote.sync(execution)
        if synced_execution.closure.phase in [WorkflowExecutionPhase.RUNNING]:
            try:
                if response in ["y", "yes"]:
                    remote.terminate(
                        execution, "KeyboardInterrupt confirmed termination"
                    )
                    logger.info("Workflow execution terminated.")
                else:
                    logger.warning(
                        f"\nExiting script without terminating workflow execution:\n\n{execution}\n"
                    )
            except FlyteSystemException as e:
                logger.error(
                    f"Error while trying to terminate the execution: {e}"
                )
        else:
            logger.info(
                f"Workflow execution already in terminal state: {synced_execution.closure.phase}"
            )

        sys.exit()


# -----------
# DEPRECATED
# -----------


def generate_workflow_inputs(
    workflow_import_path: str = "dnadiffusion.workflows.lrwine",
    workflow_name: str = "training_workflow",
) -> Dict[str, Any]:
    """
    Deprecated in favor of `generate_entity_inputs`.
    """
    module = importlib.import_module(workflow_import_path)
    workflow = getattr(module, workflow_name)

    if not callable(workflow):
        value_error_message = f"Workflow '{workflow_name}' is not callable"
        raise ValueError(value_error_message)

    inputs = {}

    for name, param in inspect.signature(workflow).parameters.items():
        param_type = param.annotation
        default = param.default

        # check if the type is a built-in type (like int, str, etc.)
        if isinstance(param_type, type) and param_type.__module__ == "builtins":
            inputs[name] = default
        else:
            # dynamically import the type if it's not a built-in type
            type_module = importlib.import_module(param_type.__module__)
            custom_type = getattr(type_module, param_type.__name__)

            inputs[name] = builds(custom_type)

    return inputs
