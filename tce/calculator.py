r"""
this module provides an `ase.calculator.Calculator` class that wraps `tce-lib`
"""


from dataclasses import dataclass, field
from typing import Optional
from itertools import pairwise, permutations, combinations, repeat
from enum import Enum, auto
import logging
from collections import defaultdict
from string import ascii_lowercase
from math import isqrt
import warnings

from ase.calculators.calculator import Calculator
from ase import Atoms
import numpy as np
from numpy.typing import NDArray
import sparse
from scipy.spatial import KDTree
from opt_einsum import contract
from greek_alphabet import Alphabet

from .training import ClusterExpansion, Model
from .topology import FeatureComputer, topological_feature_vector_factory
from .topology import get_adjacency_tensors


LOGGER = logging.getLogger(__name__)
GREEK_ALPHABET = ''.join(char.lower for char in Alphabet.get_list())


class ASEProperty(Enum):

    r"""
    supported ASE properties to compute
    """

    ENERGY = auto()
    STRESS = auto()


STR_TO_PROPERTY: dict[str, ASEProperty] = {
    "energy": ASEProperty.ENERGY,
    "stress": ASEProperty.STRESS
}
r"""mapping from ase's string to our Enum class for properties"""

INTENSIVE_PROPERTIES: set[ASEProperty] = {
    ASEProperty.STRESS
}
r"""set of intensive properties"""


@dataclass
class TCECalculator(Calculator):

    """
    ASE calculator wrapper for `tce-lib`.
    """

    cluster_expansions: dict[ASEProperty, ClusterExpansion]
    feature_computers: dict[ASEProperty, FeatureComputer] = field(init=False)

    def __post_init__(self):

        for e1, e2 in pairwise(self.cluster_expansions.values()):
            if e1.cluster_basis != e2.cluster_basis:
                raise ValueError(f"cluster bases are different in {self.__class__.__name__}")
            if np.any(e1.type_map != e2.type_map):
                raise ValueError(f"type maps are different in {self.__class__.__name__}")

        self.feature_computers = {}

        expansion_ids = list(self.cluster_expansions.keys())
        extensive_feature_computer = topological_feature_vector_factory(
            basis=self.cluster_expansions[expansion_ids[0]].cluster_basis,
            type_map=self.cluster_expansions[expansion_ids[0]].type_map,
        )

        expansion_ids = list(self.cluster_expansions.keys())
        extensive_feature_computer = topological_feature_vector_factory(
            basis=self.cluster_expansions[expansion_ids[0]].cluster_basis,
            type_map=self.cluster_expansions[expansion_ids[0]].type_map,
        )

        def intensive_feature_computer(atoms: Atoms) -> NDArray:

            return extensive_feature_computer(atoms) / len(atoms)

        for key in expansion_ids:
            if key in INTENSIVE_PROPERTIES:
                self.feature_computers[key] = intensive_feature_computer
                LOGGER.debug(f"intensive feature computer stored for property {key}")
            else:
                self.feature_computers[key] = extensive_feature_computer
                LOGGER.debug(f"extensive feature computer stored for property {key}")

    def get_property(self, name: str, atoms: Optional[Atoms] = None, allow_calculation: bool = True):

        r"""
        compute property from `ase.Atoms` object

        Args:
            name (str): name of property
            atoms (ase.Atoms): atoms object
            allow_calculation (bool): allow calculation
        """

        prop = STR_TO_PROPERTY[name]
        computer = self.feature_computers[prop]

        if atoms is None:
            raise ValueError("please provide Atoms object")

        x = computer(atoms).reshape(1, -1)
        model = self.cluster_expansions[prop].model
        predicted = model.predict(x)

        if isinstance(predicted, np.ndarray):
            predicted = predicted.squeeze()

        self.results = {name: predicted}

        return predicted


@dataclass
class TCECalculatorNew(Calculator):

    models: dict[str, Model]
    neighbor_cutoffs: NDArray[np.floating]
    many_body_features: list[tuple[int, ...]]
    species: list[str]
    topological_tensors: dict[int, sparse.COO] = field(default_factory=dict)
    feature_groups: dict[int, tuple[int, ...]] = field(init=False)
    type_to_idx: dict[str, int] = field(init=False)

    def __post_init__(self):

        self.feature_groups = defaultdict(list)
        for feature in self.many_body_features:

            # each feature motif label is size (m choose 2)
            # where m is the number of atoms in the motif
            # we want to store m -> [list of features of size m]
            # len(feature) here is the size of the label
            # n = (m choose 2) implies that m^2 - m - 2n = 0
            # so m = (1 + sqrt(1 + 8n)) / 2

            discriminant = 1 + 8 * len(feature)
            integer_sqrt = isqrt(discriminant)

            if integer_sqrt * integer_sqrt != discriminant or (1 + integer_sqrt) % 2 != 0:
                raise ValueError(
                    f"feature {feature} is invalid. Every feature label should have length (m choose 2) "
                    "where m is an integer"
                )

            self.feature_groups[(1 + integer_sqrt) // 2].append(feature)

        self.type_to_idx = {sym: a for a, sym in enumerate(self.species)}


    def get_feature_vector(
        self,
        atoms: Atoms
    ) -> NDArray[np.floating]:

        if not self.topological_tensors:

            tree = KDTree(data=atoms.positions, boxsize=np.diag(atoms.cell))

            # these are boolean, so we can sum corresponding to logical or
            adjacency_tensors = get_adjacency_tensors(
                tree=tree,
                cutoffs=self.neighbor_cutoffs
            )
            self.topological_tensors[2] = adjacency_tensors

            # for many body terms, we need to build the einsum string
            # subscripts are edges in K_m graphs
            # for three bodies, 12,23,31->123 in graph K_3
            # for four bodies, 12,23,34,41,13,24->1234 in graph K_4
            # i.e., combinations of string "abcd"

            for body_order, features in self.feature_groups.items():
                
                final_result_str = ascii_lowercase[:body_order]
                input_str = ','.join(
                    f"{i1}{i2}" for i1, i2 in combinations(final_result_str, r=2)
                )
                einsum_str = f"{input_str}->{final_result_str}"

                n_body_tensors = []
                for label in features:
                    n_body_tensor = sum(
                        contract(
                            einsum_str,
                            *(adjacency_tensors[l] for l in permuted_label)
                        ) for permuted_label in set(permutations(label))
                    )

                    if not n_body_tensor.nnz:
                        warnings.warn(f"feature {label} is identically 0")
                    
                    n_body_tensors.append(n_body_tensor)
                
                self.topological_tensors[body_order] = sparse.stack(n_body_tensors)

        symbols = np.array(atoms.get_chemical_symbols())
        indicator_tensor = symbols[:, None] == np.array(self.species)
        indicator_tensor = indicator_tensor.astype(float)

        feature_vec = []
        for body_order, t in self.topological_tensors.items():

            # need to build the cluster count string in terms of the body order here
            # e.g. "nijk,i\alpha,j\beta,k\gamma->i\alpha\beta\gamma"

            spatial_indices = ascii_lowercase[:body_order]
            species_indices = GREEK_ALPHABET[:body_order]

            input_str = f"n{spatial_indices},{','.join(f'{l}{g}' for l, g in zip(spatial_indices, species_indices))}"
            output_str = f"n{species_indices}"
            einsum_str = f"{input_str}->{output_str}"
            cluster_counts = contract(
                einsum_str,
                t,
                *repeat(indicator_tensor, body_order)
            )
            feature_vec.append(cluster_counts.flatten())

        return np.concatenate(feature_vec)


    def get_property(
        self,
        name: str, 
        atoms: Optional[Atoms] = None, 
        allow_calculation: bool = True
    ):

        if atoms is None:
            raise ValueError("Please provide an Atoms object")

        feature_vec = self.get_feature_vector(atoms)
        print(feature_vec)