from __future__ import annotations
"""
Decorators and non-VapourSynth-related functions.
"""
__all__ = [
    # decorators
    'disallow_variable_format', 'disallow_variable_resolution',
    # misc non-vapoursynth related
    'fallback', 'iterate',
]

import inspect
from functools import partial, wraps
from typing import Union, Any, TypeVar, Callable, cast, overload, Optional

import vapoursynth as vs

F = TypeVar('F', bound=Callable)
T = TypeVar('T')
R = TypeVar('R')


def _check_variable(
    function: F, vname: str, only_first: bool, check_func: Callable[[vs.VideoNode], bool]
) -> Any:
    def _check(x: Any) -> bool:
        return isinstance(x, vs.VideoNode) and check_func(x)

    @wraps(function)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        for obj in args[:1] if only_first else [*args, *kwargs.values()]:
            if _check(obj):
                raise ValueError(
                    f"{function.__name__}: 'Variable-{vname} clips not supported.'"
                )

        if not only_first:
            for name, param in inspect.signature(function).parameters.items():
                if param.default is not inspect.Parameter.empty and _check(param.default):
                    raise ValueError(
                        f"{function.__name__}: 'Variable-{vname} clip not allowed in default argument `{name}`.'"
                    )

        return function(*args, **kwargs)

    return cast(F, _wrapper)


@overload
def disallow_variable_format(*, only_first: bool = False) -> Callable[[F], F]:
    ...


@overload
def disallow_variable_format(function: F | None = None, /) -> F:
    ...


def disallow_variable_format(function: F | None = None, /, *, only_first: bool = False) -> Callable[[F], F] | F:
    """Function decorator that raises an exception if input clips have variable format.
        :param function:    Function to wrap.
        :param only_first:  Whether to check only the first argument or not.

        :return:            Wrapped function.
    """

    if function is None:
        return cast(Callable[[F], F], partial(disallow_variable_format, only_first=only_first))

    assert function

    return _check_variable(
        function, 'format', only_first, lambda x: x.format is None
    )


@overload
def disallow_variable_resolution(*, only_first: bool = False) -> Callable[[F], F]:
    ...


@overload
def disallow_variable_resolution(function: F | None = None, /) -> F:
    ...


def disallow_variable_resolution(function: F | None = None, /, *, only_first: bool = False) -> Callable[[F], F] | F:
    """Function decorator that raises an exception if input clips have variable resolution.
        :param function:    Function to wrap.
        :param only_first:  Whether to check only the first argument or not.

        :return:            Wrapped function.
    """

    if function is None:
        return cast(Callable[[F], F], partial(disallow_variable_resolution, only_first=only_first))

    assert function

    return _check_variable(
        function, 'format', only_first, lambda x: not all({x.width, x.height})
    )


def fallback(value: Optional[T], fallback_value: T) -> T:
    """Utility function that returns a value or a fallback if the value is ``None``.

    >>> fallback(5, 6)
    5
    >>> fallback(None, 6)
    6

    :param value:           Argument that can be ``None``.
    :param fallback_value:  Fallback value that is returned if `value` is ``None``.

    :return:                The input `value` or `fallback_value` if `value` is ``None``.
    """
    return fallback_value if value is None else value


def iterate(base: T, function: Callable[[Union[T, R]], R], count: int) -> Union[T, R]:
    """Utility function that executes a given function a given number of times.

    >>> def double(x):
    ...     return x * 2
    ...
    >>> iterate(5, double, 2)
    20

    :param base:      Initial value.
    :param function:  Function to execute.
    :param count:     Number of times to execute `function`.

    :return:          `function`'s output after repeating `count` number of times.
    """
    if count < 0:
        raise ValueError('Count cannot be negative.')

    v: Union[T, R] = base
    for _ in range(count):
        v = function(v)
    return v
