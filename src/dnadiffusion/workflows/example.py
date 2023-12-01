"""
A simple Flyte example.
"""

import typing

from flytekit import task, workflow


@task
def say_hello(name: str = "testing say_hello") -> str:
    """
    A simple Flyte task to say "hello".

    The @task decorator allows Flyte to use this function as a Flyte task, which
    is executed as an isolated, containerized unit of compute.
    """
    return f"hello {name}!"


@task
def greeting_length(greeting: str = "ninechars") -> int:
    """
    A task the counts the length of a greeting.
    """
    return len(greeting)


@workflow
def wf(name: str = "union") -> typing.Tuple[str, int]:
    """
    Declare workflow called `wf`.

    The @workflow decorator defines an execution graph that is composed of tasks
    and potentially sub-workflows. In this simple example, the workflow is
    composed of just one task.

    There are a few important things to note about workflows:
    - Workflows are a domain-specific language (DSL) for creating execution
      graphs and therefore only support a subset of Python's behavior.
    - Tasks must be invoked with keyword arguments
    - The output variables of tasks are Promises, which are placeholders for
      values that are yet to be materialized, not the actual values.
    """
    greeting = say_hello(name=name)
    greeting_len = greeting_length(greeting=greeting)
    return greeting, greeting_len


if __name__ == "__main__":
    # Execute the workflow, simply by invoking it like a function and passing in
    # the necessary parameters
    print(f"Running wf() { wf(name='passengers') }")
