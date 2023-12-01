import importlib
import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass, field

import pyperclip
import rich.syntax
import rich.tree
from dataclasses_json import dataclass_json
from dotenv import load_dotenv
from flytekit.configuration import Config as FlyteConfig
from flytekit.configuration import (
    FastSerializationSettings,
    ImageConfig,
    SerializationSettings,
)
from flytekit.core.base_task import PythonTask
from flytekit.core.workflow import WorkflowBase
from flytekit.remote import FlyteRemote
from hydra_zen import ZenStore, make_config, make_custom_builds_fn, to_yaml, zen
from omegaconf import DictConfig

from dnadiffusion.hydra.execution_config import (
    ClusterMode,
    ExecutionLocation,
    ExecutionMode,
    LocalMode,
    local_cluster_dev_config,
    local_cluster_prod_config,
    local_shell_config,
    remote_dev_config,
    remote_prod_config,
)
from dnadiffusion.hydra.execution_utils import (
    EntityConfig,
    generate_entity_configs,
    generate_hydra_config,
    git_info_to_workflow_version,
    random_alphanumeric_suffix,
    wait_for_workflow_completion,
)
from dnadiffusion.constants import (
    LOCAL_CLUSTER_CONFIG_FILE_PATH,
    REMOTE_CLUSTER_CONFIG_FILE_PATH,
)
from dnadiffusion.logging import configure_logging

logger = configure_logging("dnadiffusion.hydra.execute")
builds = make_custom_builds_fn(populate_full_signature=True)


@dataclass_json
@dataclass
class ExecutionContext:
    """
    Represents the execution configuration for a workflow.

    This dataclass encapsulates settings related to the execution environment,
    including the mode of execution, container image details, and workflow
    versioning information.

    Attributes:
        name (ExecutionMode): The execution mode, which dictates how and where
        the workflow is executed.
        image (str): The full name of the container image to be used in the
        execution, including the registry path.
        tag (str): The tag appended to the container image, usually git branch
        (DEV) or commit hash (PROD).
        version (str): A string representing the version of the workflow,
        typically including a commit hash or other identifiers.
    """

    mode: ExecutionMode = field(default_factory=ExecutionMode)
    image: str = "ghcr.io/pinellolab/dnadiffusion"
    tag: str = "main"
    version: str = f"dnadiffusion-main-{random_alphanumeric_suffix()}"
    package_path: str = "src"
    import_path: str = "dnadiffusion.workflows"
    project: str = "flytesnacks"
    domain: str = "development"
    wait: bool = True


def handle_local_execution(exec_mode, execution_context, entity, entity_config):
    if exec_mode.local_config.mode == LocalMode.shell:
        # https://github.com/flyteorg/flytekit/blob/dc9d26bfd29d7a3482d1d56d66a806e8fbcba036/flytekit/clis/sdk_in_container/run.py#L477
        output = entity(**entity_config.inputs)
        logger.info(f"Output:\n\n{output}\n")
        return True

    elif exec_mode.local_config.mode == LocalMode.cluster:
        config_file_path = (
            LOCAL_CLUSTER_CONFIG_FILE_PATH
            if exec_mode.local_config.cluster_config.mode == ClusterMode.dev
            else REMOTE_CLUSTER_CONFIG_FILE_PATH
        )
        return handle_cluster_execution(
            exec_mode.local_config.cluster_config.mode,
            execution_context,
            entity,
            entity_config,
            config_file_path,
        )

    return False


def handle_remote_execution(
    exec_mode, execution_context, entity, entity_config
):
    config_file_path = REMOTE_CLUSTER_CONFIG_FILE_PATH
    return handle_cluster_execution(
        exec_mode.remote_config.mode,
        execution_context,
        entity,
        entity_config,
        config_file_path,
    )


def handle_cluster_execution(
    cluster_mode, execution_context, entity, entity_config, config_file_path
):
    remote = FlyteRemote(
        config=FlyteConfig.auto(config_file=config_file_path),
        default_project=execution_context.project,
        default_domain=execution_context.domain,
    )
    logger.debug(f"Remote context:\n\n{remote.context}\n")
    image_config = ImageConfig.from_images(
        default_image=f"{execution_context.image}:{execution_context.tag}",
        m={
            "gpu": f"{execution_context.image}-gpu:{execution_context.tag}",
        },
    )

    serialization_settings = get_serialization_settings(
        cluster_mode, execution_context, entity_config, remote, image_config
    )
    register_and_execute_workflow(
        remote, entity, entity_config, execution_context, serialization_settings
    )
    return True


def get_serialization_settings(
    cluster_mode, execution_context, entity_config, remote, image_config
):
    if cluster_mode == ClusterMode.dev:
        logger.warning(
            "Development mode. Use 'prod' mode for production or CI environments."
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, upload_url = remote.fast_package(
                pathlib.Path(execution_context.package_path), output=tmp_dir
            )
        logger.info(f"Workflow package uploaded to:\n\n{upload_url}\n")
        return SerializationSettings(
            image_config=image_config,
            fast_serialization_settings=FastSerializationSettings(
                enabled=True,
                destination_dir="/root",
                distribution_location=upload_url,
            ),
        )
    elif cluster_mode == ClusterMode.prod:
        logger.info(
            f"Registering workflow: {entity_config.module_name}.{entity_config.entity_name}"
        )
        return SerializationSettings(image_config=image_config)
    else:
        raise_invalid_mode_error(cluster_mode, ClusterMode)


def register_and_execute_workflow(
    remote, entity, entity_config, execution_context, serialization_settings
):
    if isinstance(entity, WorkflowBase):
        remote.register_workflow(
            entity=entity,
            serialization_settings=serialization_settings,
            version=execution_context.version,
        )
    elif isinstance(entity, PythonTask):
        remote.register_task(
            entity=entity,
            serialization_settings=serialization_settings,
            version=execution_context.version,
        )
    execution = remote.execute(
        entity=entity,
        inputs=entity_config.inputs,
        version=execution_context.version,
        execution_name_prefix=execution_context.version,
        wait=False,
    )
    execution_url = remote.generate_console_url(execution)

    try:
        pyperclip.copy(execution_url)
    except Exception as e:
        logger.warning(f"Failed to copy execution URL to clipboard: {e}")
    logger.info(
        f"Execution submitted: {execution}\nExecution url:\n\n{execution_url}\n"
    )

    if execution_context.wait:
        wait_for_workflow_completion(execution, remote, logger)


def raise_invalid_mode_error(mode, valid_modes):
    logger.error(
        f"Invalid mode: {mode}. Please set to one of the following: {', '.join([e.value for e in valid_modes])}."
    )
    sys.exit(1)


def execute_workflow(
    zen_cfg: DictConfig,
    execution_context: ExecutionContext,
    entity_config: EntityConfig,
) -> None:
    """
    Executes the given workflow based on the Hydra configuration. The execution
    mode is controlled by the 'mode' parameter, which is an instance of the
    ExecutionContext dataclass. This dataclass encapsulates execution configuration
    details including the execution environment name (local, dev, prod),
    container image details, and versioning information.

    The 'execution_context.mode' parameter allows for the following execution environments:
    - LOCAL: Attempts to execute the workflow locally without registering it on
      the remote.
    - DEV: Executes a copy of the local workflow on the remote for development
      purposes. This mode allows for testing changes to the workflow code
      remotely without needing to rebuild and push the container image. However,
      rebuilding and pushing the image may be required for significant
      dependency changes. The workflow version is appended with a random
      alphanumeric string. This mode is intended for development purposes only
      and should not be used in production or CI environments.
    - PROD: Registers the workflow on the remote and then executes it, intended
      for production or CI environments. This mode executes the workflow against
      a container image that has been built and pushed to the registry specified
      in the ExecutionContext image. The image used is tagged with the git short
      SHA.

    In all modes, the workflow is registered with Flyte and executed. The
    function logs various informational messages, including the execution URL,
    and optionally waits for workflow completion based on the `wait` flag in the
    workflow configuration.

    Args:
        zen_cfg (DictConfig): Configuration for the execution.
        execution_context (ExecutionContext): An instance of ExecutionContext
        specifying the execution settings.
        entity_config (EntityConfig): Configuration for the workflow entity,
        including the workflow function and its inputs.

        Additional dynamic inputs for the workflow are generated based on the
        entity configuration. These inputs are determined by inspecting the
        signature of the workflow entity and are configured to be compatible
        with dataclass_json and hydra_zen, ensuring proper instantiation and
        configuration of custom types.

    Raises:
        Sets exit status one if an invalid execution mode is specified.
    """
    config_yaml = to_yaml(zen_cfg)
    tree = rich.tree.Tree("execute_workflow", style="dim", guide_style="dim")
    tree.add(rich.syntax.Syntax(config_yaml, "yaml", theme="monokai"))
    rich.print(tree)

    module = importlib.import_module(
        f"{execution_context.import_path}.{entity_config.module_name}"
    )
    entity = getattr(module, entity_config.entity_name)

    exec_mode = execution_context.mode

    if exec_mode.location == ExecutionLocation.local:
        if not handle_local_execution(
            exec_mode, execution_context, entity, entity_config
        ):
            raise_invalid_mode_error(exec_mode.local_config.mode, LocalMode)

    elif exec_mode.location == ExecutionLocation.remote:
        if not handle_remote_execution(
            exec_mode, execution_context, entity, entity_config
        ):
            raise_invalid_mode_error(exec_mode.remote_config.mode, ClusterMode)

    else:
        raise_invalid_mode_error(exec_mode.location, ExecutionLocation)


def main() -> None:
    """
    Main function that executes the workflow in one of the three modes
    determined by the config group mode (local, dev, prod):

    - In 'local' mode, it executes the workflow locally without a remote
    - In 'dev' mode, it uses the container execution_context.imagewith execution_context.tag current
      branch tag for execution. This allows executing a copy of updated local
      workflow on the remote prior to building a new image.
    - In 'prod' mode, it uses the container image with the git short SHA tag
      just after building an image. This is primarily for CI execution.

    See the `execute_workflow` function for more details.

    Note this logic regarding the image tag is independent of setting domain to
    "development", "staging", "production", etc.

    The workflow version is also separately determined based on the current git
    repo name, branch, and commit SHA.
    """

    load_dotenv()

    # equivalent to
    # hydra_zen.wrapper._implementations.store
    # except in name
    store = ZenStore(
        name="dnadiffusion",
        deferred_to_config=True,
        deferred_hydra_store=True,
    )

    store(generate_hydra_config())

    repo_name, git_branch, git_short_sha = git_info_to_workflow_version(logger)

    workflow_image = os.environ.get(
        "WORKFLOW_IMAGE",
        f"localhost:30000/{repo_name}",
    )

    ExecutionContextConf = builds(ExecutionContext)

    # Local Shell
    local_shell_execution_context = ExecutionContextConf(
        mode=local_shell_config,
        image="",
        tag="",
        version=f"{repo_name}-{git_branch}-{git_short_sha}-local-{random_alphanumeric_suffix()}",
    )

    # Local Cluster Dev
    local_cluster_dev_execution_context = ExecutionContextConf(
        mode=local_cluster_dev_config,
        image=f"localhost:30000/{repo_name}",
        tag=git_branch,
        version=f"{repo_name}-{git_branch}-{git_short_sha}-local-{random_alphanumeric_suffix()}",
    )

    # Local Cluster Prod
    local_cluster_prod_execution_context = ExecutionContextConf(
        mode=local_cluster_prod_config,
        image=f"localhost:30000/{repo_name}",
        tag=git_short_sha,
        version=f"{repo_name}-{git_branch}-{git_short_sha}",
    )

    # Remote Dev
    remote_dev_execution_context = ExecutionContextConf(
        mode=remote_dev_config,
        image=workflow_image,
        tag=git_branch,
        version=f"{repo_name}-{git_branch}-{git_short_sha}-dev-{random_alphanumeric_suffix()}",
    )

    # Remote Prod
    remote_prod_execution_context = ExecutionContextConf(
        mode=remote_prod_config,
        image=workflow_image,
        tag=git_short_sha,
        version=f"{repo_name}-{git_branch}-{git_short_sha}",
    )

    # define the execution_context store
    execution_context_store = store(group="execution_context")

    execution_context_store(local_shell_execution_context, name="local_shell")
    execution_context_store(
        local_cluster_dev_execution_context, name="local_cluster_dev"
    )
    execution_context_store(
        local_cluster_prod_execution_context, name="local_cluster_prod"
    )
    execution_context_store(remote_dev_execution_context, name="remote_dev")
    execution_context_store(remote_prod_execution_context, name="remote_prod")

    # define the entity_config store
    entity_config_store = store(group="entity_config")

    # specify the parent module whose submodules will be inspected for workflows
    parent_module_path = os.environ.get(
        "WORKFLOW_PARENT_MODULE_PATH", "dnadiffusion.workflows"
    )
    generate_entity_configs(parent_module_path, entity_config_store, logger)

    hydra_defaults = [
        "_self_",
        # test remote workflow execution
        {"execution_context": "remote_dev"},
        {"entity_config": "lrwine_training_workflow"},
        # # test local cluster task execution
        # # {"execution_context": "local_cluster_dev"},
        # {"execution_context": "local_shell"},
        # {"entity_config": "lrwine_process_data"},
    ]
    logger.debug(f"hydra_defaults: {hydra_defaults}")

    ExecuteWorkflowConf = make_config(
        hydra_defaults=hydra_defaults,
        execution_context=None,
        entity_config=None,
    )

    store(
        ExecuteWorkflowConf,
        name="execute_workflow",
    )

    store.add_to_hydra_store(overwrite_ok=True)

    zen(execute_workflow).hydra_main(
        config_path=None,
        config_name="execute_workflow",
        version_base="1.3",
    )


if __name__ == "__main__":
    """
    This script executes a Flyte workflow configured with hydra-zen.
    > dnadiffusion --help.

    == Configuration groups ==
    First override default group values (group=option)

    entity_config: example_wf, lrwine_training_workflow
    execution_context: local_cluster_dev, local_cluster_prod, local_shell,
    remote_dev, remote_prod


    == Config ==
    Then override any element in the config (foo.bar=value)

    execution_context:
      _target_: dnadiffusion.hydra.execute.ExecutionContext
      mode:
        _target_: dnadiffusion.hydra.execution_config.ExecutionMode
        location: remote
        local_config: null
        remote_config:
          _target_: dnadiffusion.hydra.execution_config.ClusterConfig
          mode: dev
      image: localhost:30000/dnadiffusion
      tag: main
      version: dnadiffusion-main-16323b3-dev-a8x
      name: training_workflow
      package_path: src
      import_path: dnadiffusion.workflows
      project: flytesnacks
      domain: development
      wait: true
    entity_config:
      _target_: dnadiffusion.hydra.execution_utils.EntityConfig
      inputs:
        _target_: builtins.dict
        _convert_: all
        _args_:
        - logistic_regression:
            _target_: dnadiffusion.workflows.lrwine.LogisticRegressionInterface
            penalty: l2
            dual: false
            tol: 0.0001
            C: 1.0
            fit_intercept: true
            intercept_scaling: 1
            class_weight: null
            random_state: null
            solver: lbfgs
            max_iter: 100
            multi_class: auto
            verbose: 0
            warm_start: false
            n_jobs: null
            l1_ratio: null
      module_name: lrwine
      entity_name: training_workflow
      entity_type: PythonFunctionWorkflow

    Example usage:
        > dnadiffusion -h
        > dnadiffusion -c job
        > dnadiffusion
        > dnadiffusion \
            execution_context=remote_dev \
            entity_config=lrwine_training_workflow
        > dnadiffusion \
            entity_config.inputs._args_.0.logistic_regression.C=0.4 \
            entity_config.inputs._args_.0.logistic_regression.max_iter=1200
        # The _args_=[] only works for local_shell execution of tasks
        > dnadiffusion \
            execution_context=local_shell \
            entity_config=lrwine_process_data \
            entity_config.inputs._args_=[]
        # For remote execution of tasks, stub inputs must be provided.
        # This is only meant for testing purposes.
        > dnadiffusion execution_context=local_cluster_dev \
            entity_config=lrwine_process_data \
            entity_config.inputs._args_.0.data.data=[[12.0, 0],[13.0, 1],[9.5, 2]] \
            entity_config.inputs._args_.0.data.columns="[ash, target]"
        # TODO: update to use joblib hydra execution backend
        > dnadiffusion \
            --multirun entity_config.inputs._args_.0.logistic_regression.C=0.2,0.5

        See the the hydra config output in the git-ignored `./outputs` or
        `./multirun` directories. These are also stored as an artifact of
        the CI actions workflow in the `Upload config artifact` step.

    Warning:
        Hydra command-line overrides are only intended to be supported for
        inputs. Do not override workflow-level parameters. This will lead to
        unexpected behavior. You can modify workflow parameters with `.env` or
        environment variables. Note  `version` and `tag` are determined
        automatically in python based on `mode`. The workflow execution
        parameters are stored in the hydra config output for reference.
    """
    main()

# # LOCAL execution
# if exec_mode.location == ExecutionLocation.local:
#     if exec_mode.local_config.mode == LocalMode.shell:
#         # https://github.com/flyteorg/flytekit/blob/dc9d26bfd29d7a3482d1d56d66a806e8fbcba036/flytekit/clis/sdk_in_container/run.py#L477
#         output = entity(**entity_config.inputs)
#         logger.info(f"Workflow output:\n\n{output}\n")
#         return

#     elif exec_mode.local_config.mode == LocalMode.cluster:
#         remote = FlyteRemote(
#             config=FlyteConfig.auto(
#                 config_file=LOCAL_CLUSTER_CONFIG_FILE_PATH
#             ),
#             default_project=execution_context.project,
#             default_domain=execution_context.domain,
#         )
#         image_config = ImageConfig.auto(
#             img_name=f"{execution_context.image}:{execution_context.tag}"
#         )

#         if exec_mode.local_config.cluster_config.mode == ClusterMode.dev:
#             logger.warning(
#                 "This execution_context.mode is intended for development purposes only.\n\n"
#                 "Please use 'prod' execution_context.mode for production or CI environments.\n\n"
#             )
#             with tempfile.TemporaryDirectory() as tmp_dir:
#                 logger.debug(
#                     f"Packaged tarball temporary directory:\n\n\t{tmp_dir}\n"
#                 )
#                 _, upload_url = remote.fast_package(
#                     pathlib.Path(execution_context.package_path),
#                     output=tmp_dir,
#                 )
#             logger.info(
#                 f"Workflow package uploaded to:\n\n  {upload_url}\n"
#             )

#             serialization_settings = SerializationSettings(
#                 image_config=image_config,
#                 fast_serialization_settings=FastSerializationSettings(
#                     enabled=True,
#                     destination_dir="/root",
#                     distribution_location=upload_url,
#                 ),
#             )

#         elif exec_mode.local_config.cluster_config.mode == ClusterMode.prod:
#             logger.info(
#                 f"Registering workflow:\n\n\t{entity_config.module_name}.{entity_config.entity_name}\n"
#             )
#             serialization_settings = SerializationSettings(
#                 image_config=image_config
#             )

#         else:
#             logger.error(
#                 f"Invalid cluster mode: {exec_mode.local_config.cluster_config.mode}. "
#                 "Please set `execution_context.mode.local_config.cluster_config.mode` to one of the following: "
#                 f"{', '.join([e.value for e in ClusterMode])}."
#             )
#             sys.exit(1)

# # REMOTE execution
# elif exec_mode.location == ExecutionLocation.remote:
#     remote = FlyteRemote(
#         config=FlyteConfig.auto(
#             config_file=REMOTE_CLUSTER_CONFIG_FILE_PATH
#         ),
#         default_project=execution_context.project,
#         default_domain=execution_context.domain,
#     )
#     image_config = ImageConfig.auto(
#         img_name=f"{execution_context.image}:{execution_context.tag}"
#     )

#     if exec_mode.remote_config.mode == ClusterMode.dev:
#         logger.warning(
#             "This execution_context.mode is intended for development purposes only.\n\n"
#             "Please use 'prod' execution_context.mode for production or CI environments.\n\n"
#         )
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             logger.debug(
#                 f"Packaged tarball temporary directory:\n\n\t{tmp_dir}\n"
#             )
#             _, upload_url = remote.fast_package(
#                 pathlib.Path(execution_context.package_path),
#                 output=tmp_dir,
#             )
#         logger.info(f"Workflow package uploaded to:\n\n  {upload_url}\n")

#         serialization_settings = SerializationSettings(
#             image_config=image_config,
#             fast_serialization_settings=FastSerializationSettings(
#                 enabled=True,
#                 destination_dir="/root",
#                 distribution_location=upload_url,
#             ),
#         )

#     elif exec_mode.remote_config.mode == ClusterMode.prod:
#         logger.info(
#             f"Registering workflow:\n\n\t{entity_config.module_name}.{entity_config.entity_name}\n"
#         )
#         serialization_settings = SerializationSettings(
#             image_config=image_config
#         )

#     else:
#         logger.error(
#             f"Invalid remote cluster mode: {exec_mode.remote_config.mode}. "
#             "Please set `execution_context.mode.remote_config.mode` to one of the following: "
#             f"{', '.join([e.value for e in ClusterMode])}."
#         )
#         sys.exit(1)

# else:
#     logger.error(
#         f"Invalid execution location: {exec_mode.location}. "
#         "Please set `execution_context.mode.location` to one of the following: "
#         f"{', '.join([e.value for e in ExecutionLocation])}."
#     )
#     sys.exit(1)

# remote.register_workflow(
#     entity=entity,
#     serialization_settings=serialization_settings,
#     version=execution_context.version,
# )
# execution = remote.execute(
#     entity=entity,
#     inputs=entity_config.inputs,
#     version=execution_context.version,
#     execution_name_prefix=execution_context.version,
#     wait=False,
# )
# logger.info(f"Execution submitted:\n\n{execution}\n")
# logger.info(f"Execution url:\n\n{remote.generate_console_url(execution)}\n")

# if execution_context.wait:
#     wait_for_workflow_completion(execution, remote, logger)


# class ExecutionMode(str, Enum):
#     """
#     Enumerates the possible execution modes for a workflow.

#     Attributes:
#         LOCAL: Represents a local execution mode, where the workflow is executed
#         locally without remote registration.
#         DEV: Represents a development execution mode, where the workflow is
#         executed remotely for development purposes. This mode is used for
#         testing workflow code changes remotely without needing to rebuild and
#         push the container image so long as one with the current branch tag
#         already exists.
#         PROD: Represents a production execution mode, where the workflow is
#         registered and executed remotely, intended for production or continuous
#         integration (CI) environments. The image tag is set to the git commit
#         short SHA.

#     TODO: python >=3.11, use StrEnum with auto() values
#     """

#     LOCAL = "LOCAL"
#     DEV = "DEV"
#     PROD = "PROD"

#     def __str__(self):
#         return self.value


# remote = FlyteRemote(
#     config=FlyteConfig.auto(),
#     default_project=execution_context.project,
#     default_domain=execution_context.domain,
# )
# image_config = ImageConfig.auto(
#     img_name=f"{execution_context.image}:{execution_context.tag}"
# )

# if execution_context.mode.remote_config.mode == ClusterMode.DEV:
#     # if execution_context.mode == "dev":
#     logger.warning(
#         "This execution_context.mode is intended for development purposes only.\n\n"
#         "Please use 'prod' execution_context.mode for production or CI environments.\n\n"
#     )
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         logger.debug(
#             f"Packaged tarball temporary directory:\n\n\t{tmp_dir}\n"
#         )
#         _, upload_url = remote.fast_package(
#             pathlib.Path(execution_context.package_path),
#             output=tmp_dir,
#         )
#     logger.info(f"Workflow package uploaded to:\n\n  {upload_url}\n")

#     serialization_settings = SerializationSettings(
#         image_config=image_config,
#         fast_serialization_settings=FastSerializationSettings(
#             enabled=True,
#             destination_dir="/root",
#             distribution_location=upload_url,
#         ),
#     )
# elif execution_context.mode.remote_config.mode == ClusterMode.PROD:
#     # elif execution_context.mode == "prod":
#     logger.info(
#         f"Registering workflow:\n\n\t{entity_config.module_name}.{entity_config.entity_name}\n"
#     )
#     serialization_settings = SerializationSettings(
#         image_config=image_config
#     )
# else:
#     logger.error(
#         f"Invalid workflow registration mode: {execution_context.mode}. "
#         "Please set `execution_context.mode.remote_config.mode` to one of the following: "
#         f"{', '.join([e.value for e in ClusterMode])}."
#     )
#     sys.exit(1)
