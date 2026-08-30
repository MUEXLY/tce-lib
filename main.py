# ref from max_ent - test file edited for confirming the max_ent function works as expected

from ase.build import bulk
import numpy as np
from tce.calculator import TCECalculator, maximum_entropy_subset_up_to_size_k #changed to import from the calculator
from tce.constants import CUTOFFS

RNG: np.random.Generator = np.random.default_rng(seed=0)
SUPERSET_SIZE: int = 100
COPPER_LATTICE_PARAMETER: float = 3.61


def main():

    pure_cu = bulk("Cu", a=COPPER_LATTICE_PARAMETER, cubic=True).repeat((4, 4, 4))
    
    alloys = []
    for sample in range(SUPERSET_SIZE):

        alloy = pure_cu.copy()
        nickel_fraction = RNG.uniform(low=0.0, high=1.0)
        alloy.symbols = RNG.choice(
            a=["Cu", "Ni"], 
            p=[1.0 - nickel_fraction, nickel_fraction],
            size=len(alloy)
        )
        alloys.append(alloy)

    calc = TCECalculator(
        neighbor_cutoffs=COPPER_LATTICE_PARAMETER * CUTOFFS["fcc"][:4],
        many_body_features=[(0, 0, 0)],
        species=["Cu", "Ni"]
    )

    # the two types of lists generated:

    #1: generation through max_ent method 
    subsets_method_1 = list(calc.select_maximum_entropy_subsets(alloys, k=30))

    #2: normalized method used prior to max_ent
    X = calc.get_batched_feature_vectors(alloys)
    normalizer = calc.get_normalizer(alloys[0])
    X /= normalizer

    subsets_method_2 = []
    for index_subset in maximum_entropy_subset_up_to_size_k(X, k=30):
        alloy_indices = sorted(index_subset)
        subsets_method_2.append([alloys[i] for i in alloy_indices])

    # compare and identify defects, if none, print success message
    assert len(subsets_method_1) == len(subsets_method_2)
    for s1, s2 in zip(subsets_method_1, subsets_method_2):
        assert len(s1) == len(s2)
        for a1, a2 in zip(s1, s2):
            np.testing.assert_array_equal(a1.numbers, a2.numbers)
            np.testing.assert_allclose(a1.positions, a2.positions)

    print("Test passed: Method 1 and Method 2 yield identical results!")


if __name__ == "__main__":
    main()