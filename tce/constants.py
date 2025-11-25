r"""
This module defines some useful constants, notably lattice structures (and their corresponding cutoffs, atomic bases,
etc.). These constants define how to compute feature vectors for a solid, since topology is a function of lattice
structure.
"""

from typing import Dict, Optional
from itertools import product, permutations
from dataclasses import dataclass
import logging
import warnings

from aenum import StrEnum, auto, extend_enum
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
import sparse


LOGGER = logging.getLogger(__name__)


class LatticeStructure(StrEnum):

    r"""
    This is an `Enum` type defining typical lattice structures. Importantly, this data type helps define mappings
    between lattice structure and three body labels.

    If you want to inject a custom lattice structure, see `tce.constants.register_new_lattice_structure`.
    """

    SC = auto()
    r"""
    simple cubic lattice structure
    
    <img
        src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/assets/lattice-structures/sc.png"
        width=40%
        alt="SC unit cell"
        title="Simple cubic unit cell"
    />
    """
    BCC = auto()
    r"""
    body-centered cubic lattice structure
    
    <img
        src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/assets/lattice-structures/bcc.png"
        width=40%
        alt="BCC unit cell"
        title="body-centered cubic unit cell"
    />
    """
    FCC = auto()
    r"""
    face-centered cubic lattice structure
    
    <img
        src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/assets/lattice-structures/fcc.png"
        width=40%
        alt="FCC unit cell"
        title="face-centered cubic unit cell"
    />
    """


STRUCTURE_TO_ATOMIC_BASIS: Dict[LatticeStructure, NDArray[np.floating]] = {
    LatticeStructure.SC: np.array([
        [0.0, 0.0, 0.0]
    ]),
    LatticeStructure.BCC: np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ]),
    LatticeStructure.FCC: np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ])
}
r"""Mapping from lattice structure to atomic basis, i.e. positions of atoms within a unit cell. Here, we use the
conventional unit cell"""

STRUCTURE_TO_CUTOFF_LISTS: Dict[LatticeStructure, NDArray[np.floating]] = {
    LatticeStructure.SC: np.array([1.0, np.sqrt(2.0), np.sqrt(3.0), 2.0,
                                   np.sqrt(5.0), np.sqrt(6.0), 2.0 * np.sqrt(2.0), 3.0,
                                   np.sqrt(10.0), np.sqrt(11.0), 2 * np.sqrt(3.0), np.sqrt(13.0),
                                   np.sqrt(14.0), 4.0, np.sqrt(17.0), 3.0 * np.sqrt(2.0)]),
    LatticeStructure.BCC: np.array([0.5 * np.sqrt(3.0), 1.0, np.sqrt(2.0), 0.5 * np.sqrt(11.0),
                                    np.sqrt(3.0), 2.0, 0.5 * np.sqrt(19.0), np.sqrt(5.0), np.sqrt(6.0),
                                    1.5 * np.sqrt(3.0), 2.0 * np.sqrt(2.0), 0.5 * np.sqrt(35.0),
                                    3.0, 0.5 * np.sqrt(43.0), 2.0 * np.sqrt(3.0), 0.5 * np.sqrt(51.0)]),
    LatticeStructure.FCC: np.array([0.5 * np.sqrt(2.0), 1.0, np.sqrt(1.5), np.sqrt(2.0), np.sqrt(2.5), 
                                    np.sqrt(3.0), np.sqrt(3.5), 2.0, 1.5 * np.sqrt(2.0), np.sqrt(5.0),
                                    np.sqrt(0.5 * 11.0), np.sqrt(6.0), np.sqrt(0.5 * 13.0), np.sqrt(0.5 * 15.0),
                                    2.0 * np.sqrt(2.0), np.sqrt(0.5 * 17.0), 3.0, np.sqrt(0.5 * 19.0)])
}
r"""Mapping from lattice structure to neighbor cutoffs, in units of the lattice parameter $a$"""


STRUCTURE_TO_THREE_BODY_LABELS = {
    LatticeStructure.SC: np.array([
        [0, 0, 1], 
        [1, 1, 1], 
        [0, 1, 2], 
        [0, 0, 3], 
        [0, 3, 3], 
        [1, 1, 3], 
        [2, 2, 3], 
        [0, 1, 4], 
        [0, 3, 4], 
        [0, 4, 4], 
        [1, 2, 4], 
        [1, 3, 4], 
        [1, 4, 4], 
        [2, 4, 4], 
        [3, 4, 4], 
        [4, 4, 4], 
        [0, 2, 5], 
        [0, 4, 5], 
        [0, 5, 5], 
        [1, 1, 5], 
        [1, 3, 5], 
        [1, 4, 5], 
        [1, 5, 5], 
        [2, 3, 5], 
        [2, 4, 5], 
        [3, 5, 5], 
        [4, 4, 5], 
        [4, 5, 5], 
        [5, 5, 5], 
        [0, 4, 6], 
        [0, 6, 6], 
        [1, 1, 6], 
        [1, 4, 6], 
        [1, 5, 6], 
        [1, 6, 6], 
        [2, 2, 6], 
        [2, 5, 6], 
        [3, 3, 6], 
        [3, 4, 6], 
        [4, 4, 6], 
        [4, 5, 6], 
        [5, 5, 6], 
        [6, 6, 6], 
        [0, 5, 7], 
        [0, 6, 7], 
        [0, 7, 7], 
        [1, 2, 7], 
        [1, 4, 7], 
        [1, 5, 7], 
        [1, 6, 7], 
        [1, 7, 7], 
        [2, 4, 7], 
        [2, 5, 7], 
        [2, 6, 7], 
        [2, 7, 7], 
        [3, 4, 7], 
        [3, 5, 7], 
        [3, 7, 7], 
        [4, 4, 7], 
        [4, 5, 7], 
        [4, 6, 7], 
        [4, 7, 7], 
        [5, 5, 7], 
        [5, 6, 7], 
        [5, 7, 7], 
        [6, 6, 7], 
        [6, 7, 7], 
        [7, 7, 7], 
        [0, 7, 10],
        [0, 10, 10], 
        [1, 5, 10], 
        [1, 7, 10], 
        [1, 10, 10], 
        [2, 2, 10], 
        [2, 5, 10], 
        [2, 7, 10], 
        [2, 10, 10], 
        [3, 6, 10], 
        [3, 7, 10], 
        [4, 4, 10], 
        [4, 5, 10], 
        [4, 6, 10], 
        [4, 7, 10], 
        [5, 5, 10], 
        [5, 6, 10], 
        [5, 7, 10]
    ]),
    LatticeStructure.BCC: np.array([
        [0, 0, 1], 
        [0, 0, 2], 
        [1, 1, 2], 
        [2, 2, 2], 
        [0, 1, 3], 
        [0, 2, 3], 
        [1, 3, 3], 
        [2, 3, 3], 
        [0, 0, 4], 
        [0, 3, 4], 
        [1, 2, 4], 
        [3, 3, 4], 
        [0, 3, 5], 
        [1, 1, 5], 
        [2, 2, 5], 
        [4, 4, 5], 
        [0, 2, 6], 
        [0, 4, 6], 
        [1, 3, 6], 
        [1, 6, 6], 
        [2, 3, 6], 
        [2, 6, 6], 
        [3, 4, 6], 
        [3, 5, 6], 
        [4, 6, 6], 
        [0, 3, 7], 
        [0, 6, 7], 
        [1, 2, 7], 
        [1, 5, 7], 
        [2, 4, 7], 
        [2, 7, 7], 
        [3, 3, 7], 
        [3, 6, 7], 
        [5, 7, 7], 
        [6, 6, 7], 
        [0, 3, 8], 
        [0, 6, 8], 
        [1, 4, 8], 
        [1, 7, 8], 
        [2, 2, 8], 
        [2, 5, 8], 
        [2, 8, 8], 
        [3, 3, 8], 
        [3, 6, 8], 
        [4, 7, 8], 
        [5, 8, 8], 
        [6, 6, 8], 
        [7, 7, 8], 
        [8, 8, 8], 
        [0, 4, 9], 
        [0, 8, 9], 
        [1, 6, 9], 
        [1, 9, 9], 
        [2, 3, 9], 
        [2, 6, 9], 
        [2, 9, 9], 
        [3, 4, 9], 
        [3, 7, 9], 
        [3, 8, 9], 
        [4, 6, 9], 
        [4, 9, 9], 
        [5, 6, 9], 
        [6, 7, 9], 
        [6, 8, 9], 
        [0, 6, 10], 
        [1, 7, 10], 
        [2, 2, 10], 
        [2, 8, 10], 
        [3, 3, 10], 
        [3, 9, 10], 
        [4, 4, 10], 
        [5, 5, 10], 
        [6, 6, 10], 
        [7, 7, 10], 
        [8, 8, 10], 
        [10, 10, 10], 
        [0, 6, 12], 
        [0, 9, 12], 
        [1, 8, 12], 
        [1, 10, 12], 
        [2, 4, 12], 
        [2, 7, 12], 
        [2, 12, 12], 
        [3, 3, 12], 
        [3, 6, 12], 
        [3, 9, 12], 
        [4, 8, 12], 
        [5, 7, 12], 
        [5, 12, 12], 
        [6, 6, 12], 
        [7, 8, 12], 
        [7, 10, 12], 
        [0, 9, 14], 
        [1, 12, 14], 
        [2, 8, 14], 
        [3, 6, 14], 
        [4, 4, 14], 
        [5, 10, 14], 
        [7, 7, 14]
    ]),
    LatticeStructure.FCC: np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 1, 2],
        [0, 2, 2],
        [1, 2, 2],
        [2, 2, 2],
        [0, 0, 3],
        [0, 2, 3],
        [1, 1, 3],
        [2, 2, 3],
        [3, 3, 3],
        [0, 1, 4],
        [0, 2, 4],
        [0, 3, 4],
        [0, 4, 4],
        [1, 4, 4],
        [2, 2, 4],
        [2, 3, 4],
        [2, 4, 4],
        [3, 4, 4],
        [0, 2, 5],
        [1, 3, 5],
        [2, 4, 5],
        [0, 2, 6],
        [0, 3, 6],
        [0, 4, 6],
        [0, 5, 6],
        [0, 6, 6],
        [1, 2, 6],
        [1, 4, 6],
        [1, 6, 6],
        [2, 2, 6],
        [2, 3, 6],
        [2, 4, 6],
        [2, 6, 6],
        [3, 4, 6],
        [3, 6, 6],
        [4, 4, 6],
        [4, 5, 6],
        [4, 6, 6],
        [6, 6, 6],
        [0, 4, 7],
        [1, 1, 7],
        [2, 2, 7],
        [2, 6, 7],
        [3, 3, 7],
        [5, 5, 7],
        [6, 6, 7],
        [0, 3, 8],
        [0, 4, 8],
        [0, 6, 8],
        [0, 7, 8],
        [1, 2, 8],
        [1, 4, 8],
        [1, 8, 8],
        [2, 2, 8],
        [2, 3, 8],
        [2, 4, 8],
        [2, 5, 8],
        [2, 6, 8],
        [2, 8, 8],
        [3, 4, 8],
        [3, 6, 8],
        [3, 8, 8],
        [4, 4, 8],
        [4, 6, 8],
        [4, 7, 8],
        [5, 6, 8],
        [6, 6, 8],
        [8, 8, 8],
        [0, 4, 9],
        [0, 6, 9],
        [0, 8, 9],
        [1, 3, 9],
        [1, 7, 9],
        [2, 2, 9],
        [2, 4, 9],
        [2, 6, 9],
        [3, 5, 9],
        [3, 9, 9],
        [4, 4, 9],
        [4, 8, 9],
        [6, 6, 9],
        [6, 8, 9],
        [7, 9, 9],
        [0, 5, 10],
        [0, 6, 10],
        [0, 10, 10],
        [1, 6, 10],
        [1, 8, 10],
        [1, 10, 10],
        [2, 2, 10],
        [2, 3, 10],
        [2, 6, 10],
        [2, 8, 10],
        [2, 9, 10],
        [2, 10, 10],
        [3, 4, 10],
        [3, 6, 10],
        [3, 8, 10],
        [3, 10, 10],
        [4, 4, 10],
        [4, 5, 10],
        [4, 6, 10],
        [4, 8, 10],
        [4, 9, 10],
        [4, 10, 10],
        [5, 8, 10],
        [6, 6, 10],
        [6, 7, 10],
        [6, 8, 10],
        [6, 9, 10],
        [6, 10, 10],
        [7, 10, 10],
        [8, 8, 10],
        [8, 10, 10],
        [9, 10, 10],
        [10, 10, 10],
        [0, 6, 11],
        [0, 8, 11],
        [0, 10, 11],
        [1, 5, 11],
        [1, 9, 11],
        [2, 2, 11],
        [2, 4, 11],
        [2, 6, 11],
        [2, 8, 11],
        [2, 10, 11],
        [3, 3, 11],
        [3, 7, 11],
        [3, 11, 11],
        [4, 6, 11],
        [4, 8, 11],
        [4, 10, 11],
        [5, 9, 11],
        [6, 6, 11],
        [6, 8, 11],
        [6, 10, 11],
        [7, 11, 11],
        [8, 8, 11],
        [8, 10, 11],
        [9, 9, 11],
        [10, 10, 11],
        [11, 11, 11],
        [0, 6, 12],
        [0, 8, 12],
        [0, 9, 12],
        [0, 10, 12],
        [0, 11, 12],
        [0, 12, 12],
        [1, 6, 12],
        [1, 8, 12],
        [1, 12, 12],
        [2, 3, 12],
        [2, 4, 12],
        [2, 5, 12],
        [2, 6, 12],
        [2, 8, 12],
        [2, 9, 12],
        [2, 10, 12],
        [2, 11, 12],
        [2, 12, 12],
        [3, 6, 12],
        [3, 8, 12],
        [3, 10, 12],
        [3, 12, 12],
        [4, 4, 12],
        [4, 6, 12],
        [4, 7, 12],
        [4, 8, 12],
        [4, 9, 12],
        [4, 10, 12],
        [4, 11, 12],
        [5, 6, 12],
        [5, 10, 12],
        [6, 6, 12],
        [6, 8, 12],
        [6, 10, 12],
        [6, 11, 12],
        [6, 12, 12],
        [7, 8, 12],
        [8, 9, 12],
        [8, 10, 12],
        [8, 11, 12],
        [8, 12, 12],
        [9, 10, 12],
        [9, 12, 12],
        [10, 11, 12],
        [10, 12, 12],
        [12, 12, 12],
        [0, 8, 14],
        [0, 12, 14],
        [1, 9, 14],
        [2, 6, 14],
        [2, 10, 14],
        [3, 3, 14],
        [3, 11, 14],
        [4, 4, 14],
        [4, 8, 14],
        [5, 5, 14],
        [6, 6, 14],
        [6, 10, 14],
        [7, 7, 14],
        [8, 12, 14],
        [9, 9, 14],
        [11, 11, 14],
        [12, 12, 14],
        [14, 14, 14],
        [0, 10, 15],
        [0, 11, 15],
        [0, 15, 15],
        [1, 10, 15],
        [1, 12, 15],
        [1, 15, 15],
        [2, 5, 15],
        [2, 6, 15],
        [2, 10, 15],
        [2, 11, 15],
        [2, 12, 15],
        [2, 15, 15],
        [3, 6, 15],
        [3, 8, 15],
        [3, 10, 15],
        [3, 12, 15],
        [3, 15, 15],
        [4, 6, 15],
        [4, 8, 15],
        [4, 9, 15],
        [4, 10, 15],
        [4, 11, 15],
        [4, 12, 15],
        [4, 14, 15],
        [5, 6, 15],
        [5, 10, 15],
        [6, 6, 15],
        [6, 8, 15],
        [6, 9, 15],
        [6, 10, 15],
        [6, 11, 15],
        [6, 12, 15],
        [7, 8, 15],
        [7, 12, 15],
        [8, 9, 15],
        [8, 11, 15],
        [8, 12, 15],
        [8, 14, 15],
        [9, 12, 15],
        [0, 10, 16],
        [0, 12, 16],
        [0, 15, 16],
        [1, 11, 16],
        [1, 14, 16],
        [2, 6, 16],
        [2, 8, 16],
        [2, 10, 16],
        [2, 15, 16],
        [3, 5, 16],
        [3, 9, 16],
        [3, 16, 16],
        [4, 6, 16],
        [4, 8, 16],
        [4, 12, 16],
        [4, 15, 16],
        [5, 11, 16],
        [6, 6, 16],
        [6, 10, 16],
        [6, 12, 16],
        [7, 9, 16],
        [7, 16, 16],
        [8, 8, 16],
        [8, 10, 16],
        [8, 12, 16],
        [9, 11, 16],
        [9, 14, 16],
    ])
}
r"""Mapping from lattice structure to set of three body labels"""


def get_three_body_labels(
    lattice_structure: LatticeStructure,
    tolerance: float = 0.01,
    min_num_sites: int = 125
) -> NDArray[np.integer]:
    min_num_unit_cells = min_num_sites // len(STRUCTURE_TO_ATOMIC_BASIS[lattice_structure])
    s = np.ceil(np.cbrt(min_num_unit_cells))
    size = (s, s, s)
    i, j, k = (np.arange(s) for s in size)
    unit_cell_positions = np.array(np.meshgrid(i, j, k, indexing='ij')).reshape(3, -1).T

    cutoffs = STRUCTURE_TO_CUTOFF_LISTS[lattice_structure]
    positions = unit_cell_positions[:, np.newaxis, :] + \
                STRUCTURE_TO_ATOMIC_BASIS[lattice_structure][np.newaxis, :, :]
    positions = positions.reshape(-1, 3)

    tree = KDTree(positions, boxsize=np.array(size))
    distances = tree.sparse_distance_matrix(tree, max_distance=(1.0 + tolerance) * cutoffs[-1]).tocsr()
    distances.eliminate_zeros()
    distances = sparse.COO.from_scipy_sparse(distances)

    adjacency_tensors = sparse.stack([
        sparse.where(
            sparse.logical_and(distances > (1.0 - tolerance) * c, distances < (1.0 + tolerance) * c),
            x=True, y=False
        ) for c in cutoffs
    ])

    max_adj_order = adjacency_tensors.shape[0]
    non_zero_labels = []
    for labels in product(*[range(max_adj_order) for _ in range(3)]):
        if not labels[0] <= labels[1] <= labels[2]:
            continue
        three_body_tensor = sum(
            (sparse.einsum(
                "ij,jk,ki->ijk",
                adjacency_tensors[i],
                adjacency_tensors[j],
                adjacency_tensors[k]
            ) for i, j, k in set(permutations(labels))),
            start=sparse.COO(coords=[], shape=(len(positions), len(positions), len(positions)))
        )
        if not three_body_tensor.nnz:
            continue
        non_zero_labels.append(list(labels))

    non_zero_labels.sort(key=lambda x: (max(x), x))
    return np.array(non_zero_labels)


def register_new_lattice_structure(
    name: str,
    atomic_basis: NDArray[np.floating],
    cutoff_list: NDArray[np.floating],
    three_body_labels: Optional[NDArray[np.integer]] = None,
) -> None:

    """
    Register a new lattice structure to `tce`. For example, if I want to register a cubic diamond lattice:

    ```py
    from tce.constants import register_new_lattice_structure, LatticeStructure
    import numpy as np

    register_new_lattice_structure(
        name="DIAMOND",
        atomic_basis=np.array([
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.5, 0.5],
            [0.25, 0.75, 0.75],
            [0.5, 0.0, 0.5],
            [0.75, 0.25, 0.75],
            [0.5, 0.5, 0.0],
            [0.75, 0.75, 0.25]
        ]),
        cutoff_list=np.array([
            0.25 * np.sqrt(3.0), 0.5 * np.sqrt(2.0), 0.25 * np.sqrt(11.0), 1.0
        ])
    )

    assert LatticeStructure.DIAMOND in LatticeStructure
    ```

    Args:
        name (str):
            The name of the new lattice structure.
        atomic_basis (NDArray[np.floating]):
            The atomic basis of the new lattice structure.
        cutoff_list (NDArray[np.floating]):
            The cutoff list of the new lattice structure.
        three_body_labels (NDArray[np.integer]):
            The three-body labels of the new lattice structure. If not specified, the three body labels will be
            automatically computed. This is potentially expensive, though - it is very recommended to compute them once
            and then store them.
    """

    extend_enum(LatticeStructure, name)
    structure = getattr(LatticeStructure, name)
    STRUCTURE_TO_ATOMIC_BASIS[structure] = atomic_basis
    STRUCTURE_TO_CUTOFF_LISTS[structure] = cutoff_list

    if three_body_labels is None:
        warnings.warn(f"three body labels for {name} not specified. These will be computed, but this is expensive")
        three_body_labels = get_three_body_labels(structure)
    STRUCTURE_TO_THREE_BODY_LABELS[structure] = three_body_labels


@dataclass(frozen=True, eq=True)
class ClusterBasis:

    r"""
    Cluster basis class which defines lattice structure and however many neighbors and triplets to include.

    For example, if I wanted to define a cluster expansion model for an fcc crystal with up to 3rd nearest neighbors
    and the first-order three-body term (which is an equilateral triangle):

    ```py
    from tce.constants import LatticeStructure, ClusterBasis
    from tce.training import train

    basis = ClusterBasis(
        lattice_structure=LatticeStructure.FCC,
        lattice_parameter=...,
        max_adjacency_order=3,
        max_triplet_order=1
    )

    model = train(
        configurations=...,
        basis=basis
    )
    ```
    """

    lattice_structure: LatticeStructure
    r"""lattice structure that the trained model corresponds to"""

    lattice_parameter: float
    r"""lattice parameter that the trained model corresponds to"""

    max_adjacency_order: int
    r"""maximum adjacency order (number of nearest neighbors) that the trained model accounts for"""

    max_triplet_order: int
    r"""maximum triplet order (number of three-body clusters) that the trained model accounts for"""
