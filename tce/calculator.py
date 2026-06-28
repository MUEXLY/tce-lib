r"""
this module provides an `ase.calculator.Calculator` class that wraps `tce-lib`
"""


from dataclasses import dataclass, field
from typing import Optional
from itertools import pairwise, permutations, combinations, repeat, product
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

from .training import ClusterExpansion, Model, LimitingRidge
from .topology import FeatureComputer, topological_feature_vector_factory, hash_topology, symmetrize
from .topology import get_adjacency_tensors


LOGGER = logging.getLogger(__name__)
GREEK_ALPHABET = ''.join(char.lower for char in Alphabet.get_list())


class ASEProperty(Enum):

    r"""
    supported ASE properties to compute

    Deprecated: This enum is only used by the legacy `TCECalculator` wrapper.
    """

    ENERGY = auto()
    STRESS = auto()


STR_TO_PROPERTY: dict[str, ASEProperty] = {
    "energy": ASEProperty.ENERGY,
    "stress": ASEProperty.STRESS
}
r"""mapping from ase's string to our Enum class for properties

Deprecated: Only used by the legacy `TCECalculator` wrapper."""

INTENSIVE_PROPERTIES: set[ASEProperty] = {
    ASEProperty.STRESS
}
r"""set of intensive properties

Deprecated: Only used by the legacy `TCECalculator` wrapper."""


@dataclass
class TCECalculatorOld(Calculator):

    """
    ASE calculator wrapper for `tce-lib`.

    Deprecated: This class is deprecated and will be removed in a future release. Use
    `TCECalculator` instead.
    """

    cluster_expansions: dict[ASEProperty, ClusterExpansion]
    feature_computers: dict[ASEProperty, FeatureComputer] = field(init=False)

    def __post_init__(self):

        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed in a future release. "
            "Use TCECalculator or another supported ASE calculator wrapper instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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
class TCECalculator(Calculator):

    neighbor_cutoffs: NDArray[np.floating]
    many_body_features: list[tuple[int, ...]]
    species: list[str]
    models: dict[str, Model] = field(default_factory=dict)
    topological_tensors: dict[tuple[str, str], dict[int, sparse.COO]] = field(default_factory=dict)
    feature_groups: dict[int, tuple[int, ...]] = field(init=False)
    type_to_idx: dict[str, int] = field(init=False)
    einsum_strs: dict[int, str] = field(init=False)
    feature_vector_size: int = field(init=False)

    def __post_init__(self):

        if not self.models:
            self.models = {"energy": LimitingRidge()}

        self.feature_groups = defaultdict(list)
        self.einsum_strs = {2: "nij,iα,jβ->nαβ"}
        seen_features: set[tuple[int, ...]] = set()
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

            if tuple(sorted(feature)) in seen_features:
                warnings.warn(f"feature {feature} is a duplicate feature by symmetry")
            
            seen_features.add(tuple(sorted(feature)))
            self.feature_groups[(1 + integer_sqrt) // 2].append(feature)

        # Pre-compute einsum strings for each body order
        for body_order in self.feature_groups.keys():
            latin_indices = ascii_lowercase[:body_order]
            greek_indices = GREEK_ALPHABET[:body_order]
            input_str = f"n{latin_indices},{','.join(f'{l}{g}' for l, g in zip(latin_indices, greek_indices))}"
            output_str = f"n{greek_indices}"
            self.einsum_strs[body_order] = f"{input_str}->{output_str}"

        # Pre-compute feature vector size
        num_species = len(self.species)
        self.feature_vector_size = len(self.neighbor_cutoffs) * (num_species ** 2)
        
        # For body_order >= 3, the number of features is len(feature_groups[body_order])
        for body_order in self.feature_groups.keys():
            if body_order >= 3:
                num_features = len(self.feature_groups[body_order])
                self.feature_vector_size += num_features * (num_species ** body_order)

        self.type_to_idx = {sym: a for a, sym in enumerate(self.species)}

    def get_feature_label_order(self) -> list[tuple[tuple[int, ...], tuple[str, ...]]]:

        r"""
        Returns the ordered feature labels corresponding to the flattened feature vector.

        Each feature label is returned as a tuple of two multisets:
        - the first multiset is the topological feature label (sorted adjacency indices),
        - the second multiset is the species types in the feature.
        """

        labels: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
        num_species = len(self.species)

        # Two-body features are always present and correspond to the adjacency tensor block.
        for neighbor_order in range(len(self.neighbor_cutoffs)):
            topological_label = (neighbor_order,)
            for species_indices in product(range(num_species), repeat=2):
                species_multiset = tuple(
                        (self.species[idx] for idx in species_indices)
                    )
                labels.append((topological_label, species_multiset))

        # Many-body features follow in the same order as `get_topological_tensors`.
        for body_order, features in self.feature_groups.items():
            if body_order < 3:
                continue
            for feature in features:
                topological_label = tuple(sorted(feature))
                for species_indices in product(range(num_species), repeat=body_order):
                    species_multiset = tuple(
                        (self.species[idx] for idx in species_indices)
                    )
                    labels.append((topological_label, species_multiset))

        return labels

    def get_topological_tensors(self, atoms: Atoms) -> dict[int, sparse.COO]:

        topology_key = hash_topology(atoms)
        topological_tensors = self.topological_tensors.get(topology_key)

        if topological_tensors is None:

            tree = KDTree(data=atoms.positions, boxsize=np.diag(atoms.cell))

            # these are boolean, so we can sum corresponding to logical or
            adjacency_tensors = get_adjacency_tensors(
                tree=tree,
                cutoffs=self.neighbor_cutoffs
            )
            topological_tensors = {2: adjacency_tensors}

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
                
                stacked_tensors = sparse.stack(n_body_tensors)
                difference = symmetrize(stacked_tensors, axes=tuple(range(1, 1 + body_order))) - stacked_tensors
                if difference.nnz and not np.allclose(difference.data, 0):
                    raise ValueError(
                        f"Topological tensors for body order {body_order} are not symmetric in indices 1..{body_order}"
                    )
                topological_tensors[body_order] = stacked_tensors

            self.topological_tensors[topology_key] = topological_tensors

        return topological_tensors


    def get_feature_vector(
        self,
        atoms: Atoms
    ) -> NDArray[np.floating]:

        topological_tensors = self.get_topological_tensors(atoms)

        symbols = np.array(atoms.get_chemical_symbols())
        indicator_tensor = symbols[:, None] == np.array(self.species)
        indicator_tensor = indicator_tensor.astype(float)

        # Pre-allocate feature vector
        feature_vec = np.zeros(self.feature_vector_size, dtype=np.float64)
        pos = 0
        
        for body_order, t in topological_tensors.items():
            einsum_str = self.einsum_strs[body_order]
            cluster_counts = sparse.einsum(
                einsum_str,
                t,
                *repeat(indicator_tensor, body_order)
            )
            flattened = cluster_counts.flatten()
            feature_vec[pos:pos+len(flattened)] = flattened.todense()
            pos += len(flattened)

        return feature_vec


    def get_property(
        self,
        name: str, 
        atoms: Optional[Atoms] = None, 
        allow_calculation: bool = True
    ):

        if atoms is None:
            raise ValueError("Please provide an Atoms object")

        feature_vec = self.get_feature_vector(atoms)
        return self.models[name].predict(feature_vec.reshape(1, -1)).squeeze()

    def train(self, configurations: list[Atoms]):

        feature_matrix = np.array([self.get_feature_vector(atoms) for atoms in configurations])
        for name, model in self.models.items():
            target = np.array([atoms.calc.get_property(name) for atoms in configurations])
            self.models[name].fit(feature_matrix, target)