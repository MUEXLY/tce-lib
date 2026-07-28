r"""
This module tells `tce-lib` how to compute adjacency tensors and has various other utilities.
"""

from itertools import permutations
from typing import Optional, Union
import hashlib
import logging

import numpy as np
from numpy.typing import NDArray
import sparse
from ase import Atoms
from ase.neighborlist import neighbor_list


LOGGER = logging.getLogger(__name__)


def symmetrize(tensor: sparse.COO, axes: Optional[tuple[int, ...]] =None) -> sparse.COO:
    r"""
    symmetrize a tensor $T$:

    $$T_{(i_1 i_2 \cdots i_r)} = \frac{1}{r!}\sum_{\sigma\in S_n} T_{\sigma(i_1) \sigma(i_2) \cdots \sigma(i_r)}$$

    Where $S_n$ is the symmetric group on $n$ elements, so we are summing over the permutations of the indices.

    E.g., $T_{(12)} = \frac{T_{12} + T_{21}}{2}$, or equivalently $\text{symmetrize}(T) = \frac{T + T^\intercal}{2}$

    Specify the `axes` argument if you only want to symmetrize over a subset of indices

    Args:
        tensor (sparse.COO):
            The tensor $T$ to symmetrize
        axes (tuple[int]):
            The axes over which to symmetrize. If not provided, symmetrize over all axes. Defaults to `None`.
    """

    if not axes:
        axes = tuple(range(tensor.ndim))

    perms = list(permutations(axes))

    return sum(sparse.moveaxis(tensor, axes, perm) for perm in perms) / len(perms)


def get_adjacency_tensors(
    atoms: Atoms,
    cutoffs: Union[list[float], NDArray[np.floating]],
    tolerance: float = 0.01
) -> sparse.COO:

    r"""
    compute adjacency tensors $A_{ij}^{(n)}$. we first compute the sparse distance matrix as a `sparse.COO` tensor. 
    then we stack the tensors according to neighbor order, i.e., $A_{ij}^{(n)} = 1$ if sites $i$ and $j$ are $n$th order 
    neighbors, and $0$ else.

    Args:
        atoms (ase.Atoms):
            The atoms to compute adjacency tensors from. this structure stores lattice positions as well as lattice
            vectors to encode periodic boundary conditions.
        cutoffs (Union[list[float], NDArray[np.floating]]):
            Distance cutoffs for interatomic distances.
        tolerance (float):
            The tolerance $\varepsilon$ to include when binning interatomic distances. for example, when searching
            for a neighbor at distance $d$, we search in the shell $[(1 - \varepsilon)d, (1 + \varepsilon)d]$. this
            should be a small number. defaults to $0.01$.
    """

    i, j, d = neighbor_list("ijd", atoms, cutoff=(1.0 + tolerance) * cutoffs[-1])

    distances_sp = sparse.COO(
        np.vstack((i, j)),
        d,
        shape=(len(atoms), len(atoms)),
        fill_value=0.0,
        has_duplicates=False
    )

    return sparse.stack([
        sparse.where(
            sparse.logical_and(distances_sp > (1.0 - tolerance) * c, distances_sp < (1.0 + tolerance) * c),
            x=True, y=False
        ) for c in cutoffs
    ])


def hash_numpy_array(v: NDArray) -> str:

    r"""
    method to hash a numpy array so we can cache adjacency tensors when computing features
    Args:
        v (np.ndarray):
            numpy array to be hashed.
    """

    data_bytes = v.tobytes()
    shape_bytes = str(v.shape).encode("utf-8")
    combined = data_bytes + shape_bytes
    return hashlib.sha1(combined).hexdigest()


def hash_topology(atoms: Atoms) -> tuple[str, str]:

    r"""
    method to hash the topology of an Atoms object so we can cache adjacency tensors when computing features
    Args:
        atoms (Atoms):
            Atoms object from which to compute adjacency tensors.
    """

    positions_hash = hash_numpy_array(atoms.positions)
    cell_hash = hash_numpy_array(atoms.cell)

    return positions_hash, cell_hash
