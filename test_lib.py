from typing import Callable
from types import GeneratorType
from tempfile import TemporaryDirectory
from pathlib import Path
import pickle
from dataclasses import dataclass
from itertools import product

from sklearn.linear_model import RidgeCV, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD

import pytest
import numpy as np
from numpy.typing import NDArray
from ase import build, Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import sparse

from tce.training import LimitingRidge
from tce.topology import symmetrize
from tce.datasets import PresetDataset, Dataset
from tce.calculator import TCECalculator
from tce.monte_carlo import monte_carlo, transform_model
from tce.constants import CUTOFFS


@pytest.fixture
def get_supercell() -> Callable[[], Atoms]:

    def supercell(lattice_structure: str) -> Atoms:

        size = None
        if lattice_structure == "sc":
            size = (5, 5, 5)
        if lattice_structure == "bcc":
            size = (4, 4, 4)
        if lattice_structure == "fcc":
            size = (3, 3, 3)
        if not size:
            raise ValueError("lattice_structure must be sc, bcc, or fcc")

        return build.bulk(
            "X",
            a=1.0,
            crystalstructure=lattice_structure, 
            cubic=True
        ).repeat(size)

    return supercell


@pytest.mark.parametrize(
    "lattice_structure, num_expected_neighbors",
    [
        ("sc", 6),
        ("bcc", 8),
        ("fcc", 12)
    ]
)
def test_num_neighbors(lattice_structure: str, num_expected_neighbors: int, get_supercell):

    supercell = get_supercell(lattice_structure)
    calc = TCECalculator(
        neighbor_cutoffs=CUTOFFS[lattice_structure][:1],
        many_body_features=[],
        species="X"
    )
    calc.get_feature_vector(supercell)
    
    topological_tensors, = calc.topological_tensors.values()
    adjacency_tensor = topological_tensors[2]

    coord_numbers = adjacency_tensor.sum(axis=1)
    assert np.isclose(coord_numbers.std(), 0.0)
    assert coord_numbers.mean() == num_expected_neighbors


@pytest.mark.filterwarnings("ignore:More than two sites changed")
@pytest.mark.parametrize(
    "lattice_structure, permutation_support_size", 
    list(product(
        ["sc", "bcc", "fcc"], 
        range(2, 10)
    ))
)
def test_feature_vector_shortcut(
    lattice_structure: str,
    permutation_support_size: int,
    get_supercell
):

    rng = np.random.default_rng(seed=0)
    calc = TCECalculator(
        neighbor_cutoffs=CUTOFFS[lattice_structure][:1],
        many_body_features=[],
        species=["Pt", "Au", "Po"]
    )

    supercell = get_supercell(lattice_structure)
    supercell.symbols = rng.choice(calc.species, size=len(supercell))

    # Sample a unique set of sites so the reassignment is a true permutation.
    sites_to_permute = rng.choice(
        len(supercell),
        size=permutation_support_size,
        replace=False
    )
    permutation = rng.permutation(sites_to_permute)

    permuted = supercell.copy()
    permuted.symbols[sites_to_permute] = permuted.symbols[permutation]

    feature_diff = calc.get_feature_vector_difference(supercell, permuted)
    naive_diff = calc.get_feature_vector(permuted) - calc.get_feature_vector(supercell)
    
    assert np.linalg.norm(feature_diff - naive_diff, ord=np.inf) == 0


def test_noncubic_cell_raises_value_error():

    configurations = [
        build.bulk("Fe", crystalstructure="bcc", a=2.7, cubic=False).repeat((2, 2, 2)),
        build.bulk("Cr", crystalstructure="bcc", a=2.7, cubic=False).repeat((3, 3, 3))
    ]
    for configuration in configurations:
        configuration.info["energy"] = -1.0

    calc = TCECalculator(
        neighbor_cutoffs=[0.5 * np.sqrt(3.0) * 2.7],
        many_body_features=[],
        species=["Fe", "Cr"]
    )

    with pytest.raises(ValueError):
       calc.train(configurations)


def test_no_energy_computation_raises_attribute_error():

    configurations = [
        build.bulk("Fe", crystalstructure="bcc", a=2.7, cubic=True).repeat((2, 2, 2)),
        build.bulk("Cr", crystalstructure="bcc", a=2.7, cubic=True).repeat((3, 3, 3))
    ]

    calc = TCECalculator(
        neighbor_cutoffs=[0.5 * np.sqrt(3.0) * 2.7],
        many_body_features=[],
        species=["Fe", "Cr"]
    )

    with pytest.raises(AttributeError):
        calc.train(configurations)


def test_ase_calculator_api_initializes_state():

    class DummyModel:
        def predict(self, X):
            return np.zeros(X.shape[0])

    atoms = build.bulk("Fe", crystalstructure="bcc", a=2.7, cubic=True).repeat((2, 2, 2))

    calc = TCECalculator(
        models={"energy": DummyModel()},
        neighbor_cutoffs=[0.5 * np.sqrt(3.0) * 2.7],
        many_body_features=[],
        species=["Fe"]
    )

    energy = calc.get_potential_energy(atoms)

    assert energy == 0.0
    assert calc.atoms is not None
    assert calc.results["energy"] == 0.0


@pytest.mark.parametrize("preset_dataset", PresetDataset)
def test_can_load_and_compute_energies_from_dataset(preset_dataset):

    dataset = Dataset.from_preset(preset_dataset)
    print(dataset)
    for configuration in dataset.configurations:
        _ = configuration.get_potential_energy()


def test_symmetrization_no_axes():

    x = sparse.COO.from_numpy(np.array([
        [1, 1],
        [0, 1]
    ]))
    x_symmetrized = sparse.COO.from_numpy([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    assert np.all(symmetrize(x).todense() == x_symmetrized.todense())


def test_limiting_ridge_throws_error():

    lr = LimitingRidge()
    with pytest.raises(ValueError):
        lr.predict(np.zeros(2))


def test_limiting_ridge_fit():

    X = np.array([1, 2, 3]).reshape((-1, 1))
    y = np.array([2, 4, 6])
    lr = LimitingRidge().fit(X, y)
    _ = lr.score(X, y)
    assert np.all(y == lr.predict(X))


def test_can_write_and_read_model():

    X = np.array([1, 2, 3]).reshape((-1, 1))
    y = np.array([2, 4, 6])
    lr = LimitingRidge().fit(X, y)

    calc = TCECalculator(
        models={"energy": lr},
        neighbor_cutoffs=2.7 * CUTOFFS["bcc"][:3],
        many_body_features=[(0, 0, 1)],
        species=np.array(["Fe", "Cr"])
    )

    with TemporaryDirectory() as directory:
        temp_path = Path(directory) / "model.pkl"
        with pytest.warns(UserWarning):
            calc.save(temp_path)
            calc_new = TCECalculator.load(temp_path)

    assert calc_new.einsum_strs == calc.einsum_strs
    assert np.all(calc_new.models["energy"].coef_ == calc.models["energy"].coef_)


def test_bad_pkl_object():

    with TemporaryDirectory() as directory:
        temp_path = Path(directory) / "obj.pkl"
        with temp_path.open("wb") as f:
            pickle.dump(object(), f)
        with pytest.raises(ValueError), pytest.warns(UserWarning):
            _ = TCECalculator.load(temp_path)


@pytest.mark.filterwarnings(r"ignore:feature (.*, .*, .*) is identically 0")
@pytest.mark.parametrize("preset_dataset", PresetDataset)
def test_can_train_and_attach_calculator(preset_dataset):

    dataset = Dataset.from_preset(preset_dataset)
    configurations = dataset.configurations[:10]

    species = np.unique(
        np.concatenate([
            atoms.symbols for atoms in configurations
        ])
    )

    structure = dataset.lattice_structure
    cutoffs = CUTOFFS[structure][:3]

    calc = TCECalculator(
        neighbor_cutoffs=dataset.lattice_parameter * cutoffs,
        many_body_features=[(0, 0, 0), (0, 0, 1)],
        species=species
    ).train(configurations)

    for configuration in configurations:
        configuration.calc = calc
        assert isinstance(configuration.calc, TCECalculator)
        _ = configuration.get_potential_energy()


@pytest.mark.filterwarnings(r"ignore:feature (.*, .*, .*) is identically 0")
@pytest.mark.parametrize("preset_dataset", PresetDataset)
def test_can_difference_train(preset_dataset):

    dataset = Dataset.from_preset(preset_dataset)
    configurations = dataset.configurations[:10]

    structure = dataset.lattice_structure
    cutoffs = CUTOFFS[structure][:3]
    species = np.unique(
        np.concatenate([
            atoms.symbols for atoms in configurations
        ])
    )

    configuration_pairs = [
        (configurations[0], configurations[1]),
        (configurations[2], configurations[3]),
        (configurations[4], configurations[5])
    ]
    _ = TCECalculator(
        neighbor_cutoffs=dataset.lattice_parameter * cutoffs,
        many_body_features=[(0, 0, 0), (0, 0, 1)],
        species=species
    ).difference_train(configuration_pairs)


# ========================================================================

def test_floating_point_corrected():

    composition = {"Cu": 0.1, "Pd": 0.9}
    lattice_parameter = 3.862
    size = (4, 4, 4)

    rng = np.random.default_rng(seed=0)

    type_map = np.array(list(composition.keys()))
    solution = build.bulk(
        type_map[0],
        crystalstructure="fcc",
        cubic=True,
        a=lattice_parameter
    ).repeat(size)
    solution.symbols = rng.choice(type_map, p=list(composition.values()), size=len(solution))

    @dataclass
    class SurrogateModel:

        coeffs: NDArray[np.floating]

        def train(self, X, y):
            raise NotImplementedError

        def score(self, X, y):
            raise NotImplementedError

        def predict(self, x):
            return np.sum(self.coeffs * x)

    feature_length = len(type_map) ** 2 * 2 + len(type_map) ** 3 * 1
    calc = TCECalculator(
        neighbor_cutoffs=CUTOFFS["fcc"][:2] * lattice_parameter,
        many_body_features=[(0, 0, 0)],
        species=type_map,
        models={
            "energy": SurrogateModel(
                coeffs=rng.normal(
                    loc=0.0, 
                    scale=2.5e-3, 
                    size=(1, feature_length)
                )
            )
        }
    )

    _ = monte_carlo(
        initial_configuration=solution,
        tce_calculator=calc,
        num_steps=1_000,
        beta=11.1,
        generator=rng
    )


def test_sklearn_model_in_mc():

    composition = {"Cu": 0.1, "Pd": 0.9}
    lattice_parameter = 3.862
    size = (4, 4, 4)

    rng = np.random.default_rng(seed=0)

    type_map = np.array(list(composition.keys()))
    solutions = []
    for _ in range(2):
        solution = build.bulk(
            type_map[0],
            crystalstructure="fcc",
            cubic=True,
            a=lattice_parameter
        ).repeat(size)
        solution.symbols = rng.choice(type_map, p=list(composition.values()), size=len(solution))
        solution.calc = SinglePointCalculator(solution, energy=rng.normal())
        solutions.append(solution)

    calc = TCECalculator(
        neighbor_cutoffs=CUTOFFS["fcc"][:2] * lattice_parameter,
        many_body_features=[(0, 0, 0)],
        species=type_map,
        models={"energy": RidgeCV(fit_intercept=False)}
    ).train(solutions)

    _ = monte_carlo(
        initial_configuration=solutions[0],
        tce_calculator=calc,
        num_steps=10,
        beta=11.1
    )


def test_sklearn_model_with_intercept_warns_in_mc():

    composition = {"Cu": 0.1, "Pd": 0.9}
    lattice_parameter = 3.862
    size = (4, 4, 4)

    rng = np.random.default_rng(seed=0)

    type_map = np.array(list(composition.keys()))
    solutions = []
    for _ in range(2):
        solution = build.bulk(
            type_map[0],
            crystalstructure="fcc",
            cubic=True,
            a=lattice_parameter
        ).repeat(size)
        solution.symbols = rng.choice(type_map, p=list(composition.values()), size=len(solution))
        solution.calc = SinglePointCalculator(solution, energy=rng.normal())
        solutions.append(solution)

    calc = TCECalculator(
        neighbor_cutoffs=CUTOFFS["fcc"][:2] * lattice_parameter,
        many_body_features=[(0, 0, 0)],
        species=type_map,
        models={"energy": RidgeCV(fit_intercept=True)}
    ).train(solutions)

    with pytest.warns(UserWarning):
        _ = monte_carlo(
            initial_configuration=solutions[0],
            tce_calculator=calc,
            num_steps=10,
            beta=11.1
        )


def test_sklearn_pipeline_in_mc():

    composition = {"Cu": 0.1, "Pd": 0.9}
    lattice_parameter = 3.862
    size = (4, 4, 4)

    rng = np.random.default_rng(seed=0)

    type_map = np.array(list(composition.keys()))
    solutions = []
    for _ in range(2):
        solution = build.bulk(
            type_map[0],
            crystalstructure="fcc",
            cubic=True,
            a=lattice_parameter
        ).repeat(size)
        solution.symbols = rng.choice(type_map, p=list(composition.values()), size=len(solution))
        solution.calc = SinglePointCalculator(solution, energy=rng.normal())
        solutions.append(solution)

    calc = TCECalculator(
        neighbor_cutoffs=CUTOFFS["fcc"][:2] * lattice_parameter,
        many_body_features=[(0, 0, 0)],
        species=type_map,
        models={
            "energy": Pipeline([
                ("scale", StandardScaler()),
                ("fit", RidgeCV())
            ])
        }
    ).train(solutions)

    with pytest.warns(UserWarning):
        _ = monte_carlo(
            initial_configuration=solutions[0],
            tce_calculator=calc,
            num_steps=10,
            beta=11.1
        )


@pytest.mark.parametrize("beta", [11.1, [11.1]*10, np.linspace(10.0, 12.0, 10), [11.1, 11.1]])
def test_annealing_mc(beta):

    composition = {"Cu": 0.1, "Pd": 0.9}
    lattice_structure = "fcc"
    lattice_parameter = 3.862
    size = (4, 4, 4)

    rng = np.random.default_rng(seed=0)

    type_map = np.array(list(composition.keys()))
    solutions = []
    for _ in range(2):
        solution = build.bulk(
            type_map[0],
            crystalstructure=lattice_structure,
            cubic=True,
            a=lattice_parameter
        ).repeat(size)
        solution.symbols = rng.choice(type_map, p=list(composition.values()), size=len(solution))
        solution.calc = SinglePointCalculator(solution, energy=rng.normal())
        solutions.append(solution)

    calc = TCECalculator(
        neighbor_cutoffs=CUTOFFS["fcc"][:2] * lattice_parameter,
        many_body_features=[(0, 0, 0)],
        species=type_map,
        models={"energy": RidgeCV(fit_intercept=False)}
    ).train(solutions)
    
    if isinstance(beta, list):
        if len(beta) == 2:
            # invalid length. should raise error
            with pytest.raises(AssertionError):
                _ = monte_carlo(
                    initial_configuration=solutions[0],
                    tce_calculator=calc,
                    num_steps=10,
                    beta=beta
                )
            return

    # for now, only tests that no error is raised. TODO: check that correct betas are used
    _ = monte_carlo(
        initial_configuration=solutions[0],
        tce_calculator=calc,
        num_steps=10,
        beta=beta
    )


@pytest.mark.parametrize(
    "model",
    [
        LinearRegression(),
        LinearRegression(fit_intercept=False),
        Ridge(),
        Ridge(fit_intercept=False),
        Lasso(),
        Lasso(fit_intercept=False),
        ElasticNet(),
        ElasticNet(fit_intercept=False),
        Pipeline([("reduce", PCA(n_components=2)), ("fit", LinearRegression())]),
        Pipeline([("reduce", PCA(n_components=2)), ("fit", LinearRegression(fit_intercept=False))]),
        Pipeline([("scale", StandardScaler()), ("reduce", PCA(n_components=2)), ("fit", LinearRegression())]),
        Pipeline([("scale", StandardScaler()), ("reduce", PCA(n_components=2)), ("fit", Ridge())]),
        Pipeline([("scale", StandardScaler()), ("reduce", PCA(n_components=2)), ("fit", Lasso())]),
        Pipeline([("scale", StandardScaler()), ("reduce", TruncatedSVD(n_components=2)), ("fit", LinearRegression())]),
        Pipeline([("scale", StandardScaler()), ("reduce", TruncatedSVD(n_components=2)), ("fit", Ridge())]),
        Pipeline([("scale", StandardScaler()), ("reduce", TruncatedSVD(n_components=2)), ("fit", Lasso())])
    ]
)
def test_energy_diff_transform(model):

    rng = np.random.default_rng(seed=0)
    fe_cohesive_energy = 4.0
    cr_cohesive_energy = 3.0

    # fit some dummy CE using pure Fe, pure Cr, and a 50/50
    pure_fe = build.bulk("Fe", crystalstructure="bcc", a=3.0, cubic=True).repeat((5, 5, 5))
    pure_fe.calc = SinglePointCalculator(pure_fe, energy=len(pure_fe) * -fe_cohesive_energy)
    pure_cr = build.bulk("Cr", crystalstructure="bcc", a=3.0, cubic=True).repeat((5, 5, 5))
    pure_cr.calc = SinglePointCalculator(pure_cr, energy=len(pure_cr) * -cr_cohesive_energy)
    mixture = pure_fe.copy()
    mixture.symbols = rng.choice(["Fe", "Cr"], size=len(mixture))
    mixture.calc = SinglePointCalculator(
        mixture,
        energy=len(mixture) * -0.5 * (fe_cohesive_energy + cr_cohesive_energy)
    )

    calc = TCECalculator(
        neighbor_cutoffs=CUTOFFS["bcc"][:2] * 3.0,
        many_body_features=[(0, 0, 1)],
        species=["Fe", "Cr"],
        models={"energy": model}
    ).train([pure_fe, pure_cr, mixture])

    new_mixture = pure_fe.copy()
    new_mixture.symbols = rng.choice(["Fe", "Cr"], size=len(new_mixture))
    new_mixture.calc = calc

    attempt = new_mixture.copy()
    assert attempt.symbols[5] != attempt.symbols[10]
    attempt.symbols[5] = new_mixture.symbols[10]
    attempt.symbols[10] = new_mixture.symbols[5]
    attempt.calc = calc

    energy_diff = attempt.get_potential_energy() - new_mixture.get_potential_energy()
    assert np.abs(energy_diff) > 1.0e-6

    # from feature vector diff
    calc.models["energy"] = transform_model(calc.models["energy"])
    first_feature_vector = calc.get_feature_vector(new_mixture)
    second_feature_vector = calc.get_feature_vector(attempt)
    feature_diff = second_feature_vector - first_feature_vector
    energy_diff_from_delta = calc.models["energy"].predict(feature_diff.reshape(1, -1)).squeeze()
    assert np.isclose(energy_diff, energy_diff_from_delta), f"{energy_diff}, {energy_diff_from_delta}"

def test_monte_carlo_generator(): 
    """Test if the list and generator return the same results"""

    rng = np.random.default_rng(seed=0)
    fe_cohesive_energy = 4.0
    cr_cohesive_energy = 3.0

    pure_fe = build.bulk("Fe", crystalstructure="bcc", a=3.0, cubic=True).repeat((5, 5, 5))
    pure_fe.calc = SinglePointCalculator(pure_fe, energy=len(pure_fe) * -fe_cohesive_energy)
    pure_cr = build.bulk("Cr", crystalstructure="bcc", a=3.0, cubic=True).repeat((5, 5, 5))
    pure_cr.calc = SinglePointCalculator(pure_cr, energy=len(pure_cr) * -cr_cohesive_energy)
    mixture = pure_fe.copy()
    mixture.symbols = rng.choice(["Fe", "Cr"], size=len(mixture))
    mixture.calc = SinglePointCalculator(
        mixture,
        energy=len(mixture) * -0.5 * (fe_cohesive_energy + cr_cohesive_energy)
    )

    calc = TCECalculator(
        neighbor_cutoffs=CUTOFFS["bcc"][:2] * 3.0,
        many_body_features=[(0, 0, 1)],
        species=["Fe", "Cr"],
        models={"energy": RidgeCV(fit_intercept=False)}
    ).train([pure_fe, pure_cr, mixture])

    new_mixture = pure_fe.copy()
    new_mixture.symbols = rng.choice(["Fe", "Cr"], size=len(new_mixture))
    new_mixture.calc = calc

    num_steps = 10
    beta = 11.1

    list_results = monte_carlo(
        initial_configuration=new_mixture,
        tce_calculator=calc,
        num_steps=num_steps,
        beta=beta,
        return_generator=False
    )

    generator_results = monte_carlo(
        initial_configuration=new_mixture,
        tce_calculator=calc,
        num_steps=num_steps,
        beta=beta,
        return_generator=True
    )

    assert isinstance(list_results, list)
    assert isinstance(generator_results, GeneratorType)  # check if generator
    assert list_results == list(generator_results)