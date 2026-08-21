"""
Backwards-compatible entry point for deblendS. (this file will be removed in the future)
======================================================================
deblendS6 used to be the whole script; it only implemented sRestore's
output mode 6.  The implementation now lives in deblendS.py and covers
every sRestore omode (1-6 and pp0-pp3).

This module is kept so that existing .vpy scripts calling
`deblendS6.deblendS6(...)` keep working unchanged.  New code should
import deblendS and pass the omode it wants:

    import deblendS
    clip = deblendS.deblendS(clip, omode=3)

deblendS6(clip, ...) is exactly deblendS(clip, ..., omode=6).
"""

from deblendS import deblendS, deblendS6

__all__ = ['deblendS', 'deblendS6']
