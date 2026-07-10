r"""
this module provides an `ase.calculator.Calculator` class that wraps `tce-lib`
"""


from dataclasses import dataclass, field
from typing import Optional, Union
from itertools import permutations, combinations, repeat, product
import logging
from collections import defaultdict
from string import ascii_lowercase
from math import isqrt
import warnings
from pathlib import Path
import pickle

from ase.calculators.calculator import Calculator
from ase import Atoms
from ase.data import atomic_numbers
import numpy as np
from numpy.typing import NDArray
import sparse
from scipy.spatial import KDTree
from opt_einsum import contract
from greek_alphabet import Alphabet
from multiset import Multiset

from .training import Model, LimitingRidge
from .topology import hash_topology, symmetrize
from .topology import get_adjacency_tensors


LOGGER = logging.getLogger(__name__)
GREEK_ALPHABET = ''.join(char.lower for char in Alphabet.get_list())


@dataclass
class TCECalculator(Calculator):

    neighbor_cutoffs: NDArray[np.floating]
    many_body_features: list[tuple[int, ...]]
    species: list[str]
    neighbor_tolerance: float = 0.01
    intensive: dict[str, bool] = field(default_factory=dict)
    models: dict[str, Model] = field(default_factory=dict)
    atomic_numbers: NDArray[np.int64] = field(init=False)
    topological_tensors: dict[tuple[str, str], dict[int, sparse.COO]] = field(default_factory=dict)
    feature_groups: dict[int, list[tuple[int, ...]]] = field(init=False)
    type_to_idx: dict[str, int] = field(init=False)
    einsum_strs: dict[int, str] = field(init=False)
    feature_vector_size: int = field(init=False)

    def __post_init__(self):

        if not self.intensive:
            self.intensive = {"energy": False}

        if not self.models:
            self.models = {"energy": LimitingRidge()}

        self.atomic_numbers = np.array([atomic_numbers[sym] for sym in self.species], dtype=np.int64)

        self.feature_groups = defaultdict(list)
        self.einsum_strs = {2: "Lij,iα,jβ->Lαβ"}
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
            input_str = f"L{latin_indices},{','.join(f'{l}{g}' for l, g in zip(latin_indices, greek_indices))}"
            output_str = f"L{greek_indices}"
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
        print(self.feature_groups)

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
            topological_label: tuple[int, ...] = (neighbor_order,)
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
            
            if not np.all(atoms.cell.angles() == 90):
                raise ValueError("supercells must be orthogonal (for now)")

            tree = KDTree(data=atoms.positions, boxsize=np.diag(atoms.cell))

            # these are boolean, so we can sum corresponding to logical or
            adjacency_tensors = get_adjacency_tensors(
                tree=tree,
                cutoffs=self.neighbor_cutoffs,
                tolerance=self.neighbor_tolerance
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
            
            topological_tensors[2] = sparse.COO(
                coords=topological_tensors[2].coords,
                data=topological_tensors[2].data.astype(np.int64),
                shape=topological_tensors[2].shape
            )
            self.topological_tensors[topology_key] = topological_tensors

        return topological_tensors


    def get_feature_vector(
        self,
        atoms: Atoms
    ) -> NDArray[np.floating]:

        topological_tensors = self.get_topological_tensors(atoms)

        #symbols = np.array(atoms.get_chemical_symbols())
        indicator_tensor = atoms.numbers[:, None] == self.atomic_numbers[None, :]
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


    def _get_feature_vector_difference_for_sites(
        self,
        initial: Atoms,
        final: Atoms,
        sites: NDArray[np.int64] | list[int]
    ) -> NDArray[np.floating]:

        if not np.all(np.isclose(initial.positions, final.positions)):
            raise ValueError("positions of the two configurations differ")

        indicator_tensor = initial.numbers[:, None] == self.atomic_numbers[None, :]
        X_init = indicator_tensor.astype(float)

        indicator_tensor = final.numbers[:, None] == self.atomic_numbers[None, :]
        X_final = indicator_tensor.astype(float)

        sites = np.asarray(sites, dtype=np.int64)
        if len(sites) == 0:
            return np.zeros(self.feature_vector_size, dtype=np.float64)

        feature_vec_diff = np.zeros(self.feature_vector_size, dtype=np.float64)
        topological_tensors = self.get_topological_tensors(initial)
        pos = 0

        for body_order, t in topological_tensors.items():
            einsum_str = self.einsum_strs[body_order]
            truncated = sparse.take(t, sites, axis=1)

            initial_truncated = body_order * symmetrize(contract(
                einsum_str,
                truncated,
                X_init[sites, :],
                *repeat(X_init, body_order - 1)
            ), axes=tuple(range(1, 1 + body_order)))
            final_truncated = body_order * symmetrize(contract(
                einsum_str,
                truncated,
                X_final[sites, :],
                *repeat(X_final, body_order - 1)
            ), axes=tuple(range(1, 1 + body_order)))

            flattened = (final_truncated - initial_truncated).flatten()
            if hasattr(flattened, "todense"):
                flattened = flattened.todense()
            feature_vec_diff[pos:pos + len(flattened)] = np.asarray(flattened).reshape(-1)
            pos += len(flattened)

        return feature_vec_diff


    def get_feature_vector_difference_two_cycle(
        self,
        initial: Atoms,
        final: Atoms
    ) -> NDArray[np.floating]:

        sites, _ = np.where(initial.numbers[:, None] != final.numbers[:, None])
        sites = np.unique(sites)
        assert len(sites) == 2

        return self._get_feature_vector_difference_for_sites(initial, final, sites)


    def get_feature_vector_difference_transmutation(
        self,
        initial: Atoms,
        final: Atoms
    ) -> NDArray[np.floating]:

        sites = np.flatnonzero(initial.numbers != final.numbers)
        if len(sites) != 1:
            raise ValueError("transmutation feature differences require exactly one changed site")

        return self._get_feature_vector_difference_for_sites(initial, final, sites)


    def get_feature_vector_difference_nvt(self, initial: Atoms, final: Atoms) -> NDArray[np.floating]:

        # in the NVT ensemble, any move is a permutation
        # and any permutation can be decomposed into two-cycles

        mismatch = initial.numbers != final.numbers
        num_sites_changed = np.sum(mismatch)

        if num_sites_changed == 0:
            return np.zeros(self.feature_vector_size, dtype=np.float64)

        if num_sites_changed == 2:
            return self.get_feature_vector_difference_two_cycle(initial, final)

        # if there's more than 2 sites changed, we can decompose the permutation into two-cycles
        # this is potentially slow for a large permutation
        warnings.warn(
            "More than two sites changed. Decomposing into two-particle swaps. "
            "This is potentially slow, and it may be better to decompose into two-cycles analytically instead of using this function.",
            UserWarning
        )

        swaps = []
        current = initial.numbers.copy()
        target = final.numbers.copy()
        while mismatch.any():

            i = int(np.flatnonzero(mismatch)[0])

            needed = target[i]
            displaced = current[i]

            candidates = np.flatnonzero(
                mismatch & (current == needed)
            )

            reciprocal = candidates[target[candidates] == displaced]

            j = int(reciprocal[0] if len(reciprocal) else candidates[0])
            swaps.append((i, j))

            current[[i, j]] = current[[j, i]]
            mismatch[[i, j]] = current[[i, j]] != target[[i, j]]

        total_feature_diff = np.zeros(self.feature_vector_size, dtype=np.float64)
        for i, j in swaps:
            intermediate = initial.copy()
            intermediate.numbers[[i, j]] = intermediate.numbers[[j, i]]
            total_feature_diff += self.get_feature_vector_difference_two_cycle(initial, intermediate)
            initial = intermediate
        
        return total_feature_diff


    def get_feature_vector_difference(self, initial: Atoms, final: Atoms) -> NDArray[np.floating]:

        initial_counts: Multiset = Multiset(initial.numbers)
        final_counts: Multiset = Multiset(final.numbers)

        count_diff = len(initial_counts - final_counts)

        if count_diff == 0:
            return self.get_feature_vector_difference_nvt(initial, final)

        if count_diff == 1:
            return self.get_feature_vector_difference_transmutation(initial, final)

        raise NotImplementedError


    def get_property(
        self,
        name: str, 
        atoms: Optional[Atoms] = None, 
        allow_calculation: bool = True
    ):

        if atoms is None:
            raise ValueError("Please provide an Atoms object")

        feature_vec = self.get_feature_vector(atoms)
        if self.intensive[name]:
            feature_vec /= len(atoms)

        prop = self.models[name].predict(feature_vec.reshape(1, -1))
        if isinstance(prop, np.ndarray):
            prop = prop.squeeze()
        return prop


    def train(self, configurations: list[Atoms]):

        r"""
        Fit each configured model from a list of atomic configurations.

        Each configuration is expected to expose an ASE calculator with a
        ``get_property`` method for every model key in ``self.models``.
        """

        feature_matrix = np.array([self.get_feature_vector(atoms) for atoms in configurations])
        num_atoms = np.array([len(atoms) for atoms in configurations])

        for name, model in self.models.items():
            target = np.array([
                atoms.calc.get_property(name=name, atoms=atoms)
                for atoms in configurations
            ])
            if self.intensive[name]:        
                self.models[name] = model.fit(
                    feature_matrix / num_atoms[:, None], 
                    target
                )
            else:
                self.models[name] = model.fit(feature_matrix, target)

        return self


    def difference_train(self, configuration_pairs: list[tuple[Atoms, Atoms]]):

        for pair in configuration_pairs:
            assert len(pair[0]) == len(pair[1])

        feature_matrix = np.array([
            self.get_feature_vector(initial) \
            - self.get_feature_vector(final)
            for initial, final in configuration_pairs
        ])
        num_atoms = np.array([len(atoms) for atoms, _ in configuration_pairs])

        for name, model in self.models.items():

            target = np.array([
                initial.calc.get_property(name=name, atoms=initial) \
                - final.calc.get_property(name=name, atoms=final)
                for initial, final in configuration_pairs
            ])
            if self.intensive[name]:
                self.models[name].fit(
                    feature_matrix / num_atoms[:, None],
                    target
                )
            else:
                self.models[name].fit(feature_matrix, target)

        return self

    def save(self, path: Union[Path, str]):

        if isinstance(path, str):
            path = Path(path)

        warnings.warn(
            f"{self.__class__.__name__} uses pickle for now. This is unsecure! TODO write a serialization method"
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    
    @classmethod
    def load(cls, path: Union[Path, str]) -> "TCECalculator":

        warnings.warn(
            f"{cls.__name__} uses pickle for now. This is unsecure! TODO write a serialization method"
        )

        if not isinstance(path, Path):
            path = Path(path)

        with path.open("rb") as file:
            obj = pickle.load(file)

        if not isinstance(obj, cls):
            raise ValueError(f"loaded object is not of type {cls.__name__}")
        return obj
