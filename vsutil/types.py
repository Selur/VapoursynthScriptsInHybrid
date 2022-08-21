"""
Enums and related functions.
"""
__all__ = ['Dither', 'Range', 'EXPR_VARS', 'resolve_enum']

# this file only depends on the stdlib and should stay that way
from enum import Enum
from typing import Any, Callable, Optional, Type, TypeVar, Union

E = TypeVar('E', bound=Enum)


class _NoSubmoduleRepr:
    def __repr__(self):
        """Removes submodule name from standard repr, helpful since we re-export everything at the top-level."""
        return '<%s.%s.%s: %r>' % (self.__module__.split('.')[0], self.__class__.__name__, self.name, self.value)


class Dither(_NoSubmoduleRepr, str, Enum):
    """
    Enum for `zimg_dither_type_e`.
    """
    NONE = 'none'
    """Round to nearest."""
    ORDERED = 'ordered'
    """Bayer patterned dither."""
    RANDOM = 'random'
    """Pseudo-random noise of magnitude 0.5."""
    ERROR_DIFFUSION = 'error_diffusion'
    """Floyd-Steinberg error diffusion."""


class Range(_NoSubmoduleRepr, int, Enum):
    """
    Enum for `zimg_pixel_range_e`.
    """
    LIMITED = 0
    """Studio (TV) legal range, 16-235 in 8 bits."""
    FULL = 1
    """Full (PC) dynamic range, 0-255 in 8 bits."""


EXPR_VARS: str = 'xyzabcdefghijklmnopqrstuvw'
"""
This constant contains a list of all variables that can appear inside an expr-string ordered
by assignment. So the first clip will have the name *EXPR_VARS[0]*, the second one will
have the name *EXPR_VARS[1]*, and so on.

This can be used to automatically generate expr-strings.
"""


def _readable_enums(enum: Type[Enum]) -> str:
    """
    Returns a list of all possible values in `enum`.
    Since VapourSynth imported enums don't carry the correct module name, use a special case for them.
    """
    if 'importlib' in enum.__module__:
        return ', '.join([f'<vapoursynth.{i.name}: {i.value}>' for i in enum])
    else:
        return ', '.join([repr(i) for i in enum])


def resolve_enum(enum: Type[E], value: Any, var_name: str, fn: Optional[Callable] = None) -> Union[E, None]:
    """
    Attempts to evaluate `value` in `enum` if value is not ``None``, otherwise returns ``None``.

    >>> def my_fn(family: Optional[int]):
    ...     return resolve_enum(vs.ColorFamily, family, 'family', my_fn)
    >>> my_fn(None)
    None
    >>> my_fn(3)
    <ColorFamily.YUV: 3>
    >>> my_fn(9)
    ValueError: my_fn: family must be in <vapoursynth.UNDEFINED: 0>, <vapoursynth.GRAY: 1>, ...

    :param enum:      Enumeration, i.e. ``vapoursynth.ColorFamily``.
    :param value:     Value to check. Can be ``None``.
    :param var_name:  User-provided parameter name that needs to be checked.
    :param fn:        Function that should be causing the exception (used for error message only).

    :return:          The enum member or ``None``.
    """
    if value is None:
        return None
    try:
        return enum(value)
    except ValueError:
        raise ValueError(f"{fn.__name__ + ': ' if fn else ''}{var_name} must be in {_readable_enums(enum)}.") from None
