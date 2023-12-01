import dataclasses
import inspect
import sys
from dataclasses import field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    get_type_hints,
)

from mashumaro.mixins.json import DataClassJSONMixin
from sklearn.linear_model import LogisticRegression


def infer_type_from_default(value: Any) -> Type:
    """
    Infers or imputes a type from the default value of a parameter.
    Args:
        value: The default value of the parameter.
    Returns:
        The inferred type.
    """
    if value is None:
        return Optional[Any]
    elif value is inspect.Parameter.empty:
        return Any
    else:
        return type(value)


def create_dataclass_from_callable(
    callable_obj: Callable,
    overrides: Optional[Dict[str, Tuple[Type, Any]]] = None,
) -> List[Tuple[str, Type, Any]]:
    """
    Creates the fields of a dataclass from a `Callable` that includes all
    parameters of the callable as typed fields with default values inferred or
    taken from type hints. The function also accepts a dictionary containing
    parameter names together with a tuple of a type and default to allow
    specification of or override (un)typed defaults from the target callable.

    Args:
        callable_obj (Callable): The callable object to create a dataclass from.
        overrides (Optional[Dict[str, Tuple[Type, Any]]]): Dictionary to
        override inferred types and default values. Each dict value is a tuple
        (Type, default_value).

    Returns:
        Fields that can be used to construct a new dataclass type that
        represents the interface of the callable.

    Examples:
        >>> from pprint import pprint
        >>> custom_types_defaults: Dict[str, Tuple[Type, Any]] = {
        ...     "penalty": (str, "l2"),
        ...     "class_weight": (Optional[dict], None),
        ...     "random_state": (Optional[int], None),
        ...     "max_iter": (int, 2000),
        ...     "n_jobs": (Optional[int], None),
        ...     "l1_ratio": (Optional[float], None),
        ... }
        >>> fields = create_dataclass_from_callable(LogisticRegression, custom_types_defaults)
        >>> LogisticRegressionInterface = dataclasses.make_dataclass(
        ...     "LogisticRegressionInterface", fields, bases=(DataClassJSONMixin,)
        ... )
        >>> lr_instance = LogisticRegressionInterface()
        >>> isinstance(lr_instance, DataClassJSONMixin)
        True
        >>> pprint(lr_instance)
        LogisticRegressionInterface(penalty='l2',
                                    dual=False,
                                    tol=0.0001,
                                    C=1.0,
                                    fit_intercept=True,
                                    intercept_scaling=1,
                                    class_weight=None,
                                    random_state=None,
                                    solver='lbfgs',
                                    max_iter=2000,
                                    multi_class='auto',
                                    verbose=0,
                                    warm_start=False,
                                    n_jobs=None,
                                    l1_ratio=None)
    """
    if inspect.isclass(callable_obj):
        func = callable_obj.__init__
    else:
        func = callable_obj

    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    fields = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue

        if overrides and name in overrides:
            field_type, default_value = overrides[name]
        else:
            inferred_type = infer_type_from_default(param.default)
            field_type = type_hints.get(name, inferred_type)
            default_value = (
                param.default
                if param.default is not inspect.Parameter.empty
                else dataclasses.field(default_factory=lambda: None)
            )

        fields.append((name, field_type, default_value))

    return fields


if __name__ == "__main__":
    # Commented code here is primarily to support CLI or IDE debugger execution.
    # Otherwise, prefer to integrate tests and checks into the docstrings and
    # run pytest with `--xdoc` (default in this project).

    import pprint

    custom_types_defaults: Dict[str, Tuple[Type, Any]] = {
        # "penalty": (str, "l2"),
        # "dual": (bool, False),
        # "tol": (float, 1e-4),
        # "C": (float, 1.0),
        # "fit_intercept": (bool, True),
        # "intercept_scaling": (int, 1),
        # "class_weight": (Optional[dict], None),
        # "random_state": (Optional[int], None),
        # "solver": (str, "lbfgs"),
        "max_iter": (int, 2000),
        # "multi_class": (str, "auto"),
        # "verbose": (int, 0),
        # "warm_start": (bool, False),
        # "n_jobs": (Optional[int], None),
        # "l1_ratio": (Optional[float], None),
    }

    fields = create_dataclass_from_callable(
        LogisticRegression,
        custom_types_defaults,
        # {},
    )
    LogisticRegressionInterface = dataclasses.make_dataclass(
        "LogisticRegressionInterface", fields, bases=(DataClassJSONMixin,)
    )
    pprint.pprint(LogisticRegressionInterface())

    # from dataclasses import dataclass
    # from dataclasses_json import dataclass_json
    # from sklearn.linear_model import LogisticRegression
    # logistic_regression_custom_types = {
    #     "penalty": Optional[str],
    #     "class_weight": Optional[dict],
    #     "random_state": Optional[int],
    #     "n_jobs": Optional[int],
    #     "l1_ratio": Optional[float],
    # }
    # LogisticRegressionInterface = dataclass_json(
    #     dataclass(
    #         create_dataclass_from_callable_json(
    #             LogisticRegression, logistic_regression_custom_types
    #         )
    #     )
    # )
    # print("Annotations:", LogisticRegressionInterface.__annotations__)
    # print("Schema:", LogisticRegressionInterface().schema())
