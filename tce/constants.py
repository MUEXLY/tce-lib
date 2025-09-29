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

from aenum import Enum, auto, extend_enum
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
import sparse


LOGGER = logging.getLogger(__name__)


class LatticeStructure(Enum):

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
    LatticeStructure.SC: np.array([1.0, np.sqrt(2.0), np.sqrt(3.0), 2.0]),
    LatticeStructure.BCC: np.array([0.5 * np.sqrt(3.0), 1.0, np.sqrt(2.0), 0.5 * np.sqrt(11.0)]),
    LatticeStructure.FCC: np.array([0.5 * np.sqrt(2.0), 1.0, np.sqrt(1.5), np.sqrt(2.0)])
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
        [2, 2, 3]
    ]),
    LatticeStructure.BCC: np.array([
        [0, 0, 1],
        [0, 0, 2],
        [1, 1, 2],
        [2, 2, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 3, 3],
        [2, 3, 3]
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
        [3, 3, 3]
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