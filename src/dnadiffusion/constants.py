import os
import subprocess

from dnadiffusion.logging import configure_logging

logger = configure_logging("dnadiffusion.constants")


def get_git_repo_root(path="."):
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=path
        )
        return git_root.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        git_repo_not_found = (
            "Not inside a Git repository or Git is not installed."
        )
        raise OSError(git_repo_not_found)


repo_root = get_git_repo_root()

if repo_root:
    REMOTE_CLUSTER_CONFIG_FILE_PATH = os.path.join(
        repo_root, ".flyte", "config.yaml"
    )
    LOCAL_CLUSTER_CONFIG_FILE_PATH = os.path.join(
        repo_root, ".flyte", "config-local.yaml"
    )

    if not os.path.isfile(REMOTE_CLUSTER_CONFIG_FILE_PATH):
        remote_cluster_config_file_not_found_message = (
            f"Remote cluster config file not found at path:\n\n"
            f"{REMOTE_CLUSTER_CONFIG_FILE_PATH}\n\n"
            "Verify you have run `make update_config` in the root of the repository,\n"
            "or manually create the file at the path above.\n\n"
        )
        raise FileNotFoundError(remote_cluster_config_file_not_found_message)

    if not os.path.isfile(LOCAL_CLUSTER_CONFIG_FILE_PATH):
        local_cluster_config_file_not_found_message = (
            f"Local cluster config file not found at path:\n\n"
            f"{LOCAL_CLUSTER_CONFIG_FILE_PATH}\n\n"
            f"Check that you have not deleted this file from the repository.\n\n"
        )
        raise FileNotFoundError(local_cluster_config_file_not_found_message)

    logger.debug(
        f"Remote cluster config file path: {REMOTE_CLUSTER_CONFIG_FILE_PATH}"
    )
    logger.debug(
        f"Local cluster config file path: {LOCAL_CLUSTER_CONFIG_FILE_PATH}"
    )
else:
    git_repo_not_found = "Not inside a Git repository or Git is not installed."
    raise OSError(git_repo_not_found)

if __name__ == "__main__":
    from pprint import pprint

    pprint(REMOTE_CLUSTER_CONFIG_FILE_PATH)
    pprint(LOCAL_CLUSTER_CONFIG_FILE_PATH)
