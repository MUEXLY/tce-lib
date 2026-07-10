r"""
This module defines some convenience wrappers for running a Monte Carlo simulation from a fitted cluster expansion
model.
"""


from typing import Optional, Callable, TypeAlias, Sequence
import logging
from functools import wraps
import warnings

import numpy as np
from numpy.typing import NDArray
from ase import Atoms

from tce.training import Model, LimitingRidge
from tce.calculator import TCECalculator


LOGGER = logging.getLogger(__name__)
"""@private"""

MCStep: TypeAlias = Callable[[NDArray[np.floating]], NDArray[np.floating]]
r"""
Type alias defining what a step in a monte carlo simulation looks like. In general, a step should look like a function 
that takes in a state matrix $\mathbf{X}$, and returns a new one.
"""


def two_particle_swap_factory(generator: np.random.Generator) -> MCStep:

    r"""
    Factory to create a sensible MC step, which is to swap two particles.

    Args:
        generator (np.random.Generator): Random number generator to be used to sample a new MC step
    """

    @wraps(two_particle_swap_factory)
    def wrapper(state_matrix: NDArray) -> NDArray[np.floating]:

        new_state_matrix = state_matrix.copy()
        i, j = generator.integers(len(state_matrix), size=2)
        new_state_matrix[i], new_state_matrix[j] = state_matrix[j], state_matrix[i]
        return new_state_matrix

    return wrapper


EnergyModifier: TypeAlias = Callable[[NDArray[np.floating], NDArray[np.floating]], float]
r"""
Type alias defining what an energy modifier should look like. In general, a modifier should look like a function that 
takes in two state matrices $\mathbf{X}$ and $\mathbf{X}'$ and returns the term to be added to the energy difference 
$\Delta E$. For example, if you want to simulate a grand canonical ensemble, the Metropolis acceptance criterion is:

$$ \exp\left(-\beta\left(\Delta E - \sum_\alpha \mu_\alpha \Delta N_\alpha\right)\right) = \exp\left(-\beta\left(\Delta E - \boldsymbol{\mu}\cdot\Delta\mathbf{N}\right)\right) > u $$

for a random number $u$ from $\text{Uniform}(0, 1)$. You can implement this strategy by defining an energy modifier:

```py
from typing import Callable
from functools import wraps

import numpy as np

def energy_modifier_factory(
    chemical_potentials: NDArray[np.floating]
) -> Callable[[NDArray[np.floating], NDArray[np.floating]], float]:

    @wraps(energy_modifier_factory)
    def wrapper(
        state_matrix: NDArray[np.floating],
        new_state_matrix: NDArray[np.floating]
    ) -> float:
        change_in_num_types = new_state_matrix.sum(axis=0) - state_matrix.sum(axis=0)
        return -chemical_potentials @ change_in_num_types

    return wrapper
```

You can see a concrete example of the above energy modifier [here](https://github.com/MUEXLY/tce-lib#training-monte-carlo).
"""


class SurrogateModel:

    r"""
    Surrogate model class to help compute $\beta_\text{eff}$ for a complex pipeline class
    """

    def __init__(self, coeffs):

        self.coeffs = coeffs

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        return X @ self.coeffs

    def score(self, X, y):
        raise NotImplementedError


def transform_model(model: Model) -> Model:

    r"""
    Model transformation function that takes in a Model instance and returns a Model instance transformed
    to predict from $\Delta$'s for the `monte_carlo` function. Namely, if the model is $f$:

    $$
    f(\mathbf{t}) = \beta^\intercal\mathbf{t} + \beta_0
    $$

    then this function returns a model $g$ such that:
    $$
    g(\mathbf{t}_2 - \mathbf{t}_1) = \beta_\text{eff}^\intercal(\mathbf{t}_2 - \mathbf{t}_1) = f(\mathbf{t}_2) - f(\mathbf{t}_1)
    $$

    which is important in the Monte Carlo simulation, where we want energy differences $\Delta E = \mathbf{j}^\intercal\Delta\mathbf{t}$
    from models trained on an energetic model $E\approx f(\mathbf{t}) = \beta^\intercal\mathbf{t} + \beta_0$.

    For simple models, this is as easy as setting $\beta_0 = 0$. However, for complex pipelines using things like
    `sklearn.preprocessing.StandardScaler`
    ([here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)), or
    `sklearn.decomposition.PCA` ([here](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)),
    zero'ing out the mean introduces an artificial intercept.

    For a cluster expansion object that you wish to use in an MC simulation, you can use this transformation:

    ```py
    from tce.training import ClusterExpansion
    from tce.monte_carlo import transform_model, monte_carlo

    ce: ClusterExpansion = ...
    ce.model = transform(ce.model)
    trajectory = monte_carlo(cluster_expansion=ce, ...)
    ```

    **IMPORTANT**: This transformation is not possible to write in the general case. We have implemented this for a
    relatively large set of cases of `sklearn`-derived models, including `sklearn.pipeline.Pipeline`. Do not expect
    this to work for:

    - custom `tce.training.Model` instances written without `sklearn`
    - nonlinear models

    In these cases, you will have to likely write a custom transformation function to suit your specific needs.
    """

    if isinstance(model, LimitingRidge):
        return model

    from_sklearn = model.__class__.__module__.startswith('sklearn')
    if not from_sklearn:
        return model

    from sklearn.linear_model._base import LinearModel
    from sklearn.pipeline import Pipeline
    if isinstance(model, LinearModel):
        if hasattr(model, 'intercept_'):
            model.intercept_ = 0.0
        else:
            raise ValueError("Linear model does not have intercept attribute, likely unfitted")
        return model

    if isinstance(model, Pipeline):

        # most complicated case, need to calculate an effective β

        # find the final dimension d
        def _infer_input_dim(pipeline):
            # scan from the *front*
            for _, step in pipeline.steps:
                if hasattr(step, "n_features_in_"):
                    return step.n_features_in_

            # fallback: try final estimator ONLY if nothing else exists
            final = pipeline.steps[-1][1]
            if hasattr(final, "n_features_in_"):
                return final.n_features_in_

            raise ValueError("Could not infer input dimension.")

        d = _infer_input_dim(model)

        # find effective by plugging in basis vectors

        def _eval_pipeline(pipeline, X):
            y = pipeline.predict(X)
            return float(np.asarray(y).reshape(-1)[0])

        x0 = np.zeros((1, d))
        f0 = _eval_pipeline(model, x0)

        beta = np.zeros(d)

        # probe standard basis
        for i in range(d):
            xi = np.zeros((1, d))
            xi[0, i] = 1.0

            fi = _eval_pipeline(model, xi)
            beta[i] = fi - f0

        return SurrogateModel(beta)

    raise NotImplementedError


def null_energy_modifier(
    state_matrix: NDArray[np.floating],
    new_state_matrix: NDArray[np.floating]
) -> float:

    r"""
    Default energy modifier, which does nothing to the total energy
    """

    return 0.0


def monte_carlo_new(
    initial_configuration: Atoms,
    tce_calculator: TCECalculator,
    num_steps: int,
    beta: float | Sequence[float] | NDArray[np.floating],
    save_every: int = 1,
    generator: Optional[np.random.Generator] = None,
    mc_step: Optional[Callable[[Atoms], Atoms]] = None,
    energy_modifier: Optional[Callable[[Atoms, Atoms], float]] = None,
    callback: Optional[Callable[[int, int], None]] = None
) -> list[Atoms]:

    r"""
    New Monte Carlo simulation function that uses the `transform_model` function to transform the model to predict
    from $\Delta$'s. This is a more robust implementation that should work for most models, including pipelines.
    """

    if not generator:
        generator = np.random.default_rng(seed=0)

    if not callback:
        def callback(step_: int, num_steps_: int):
            LOGGER.info(f"MC step {step_:.0f}/{num_steps_:.0f}")

    if not mc_step:
        def mc_step(atoms: Atoms) -> Atoms:
            new_atoms = atoms.copy()
            i, j = generator.integers(len(atoms), size=2)
            new_atoms[i].symbol, new_atoms[j].symbol = new_atoms[j].symbol, new_atoms[i].symbol
            return new_atoms

    if not energy_modifier:
        def energy_modifier(initial: Atoms, final: Atoms) -> float:
            return 0.0

    if isinstance(beta, (Sequence, np.ndarray)):
        assert len(beta) == num_steps, "if beta is a sequence, it must be the same length as num_steps"
        beta_values = np.array(beta)
    elif isinstance(beta, float):
        beta_values = np.full(num_steps, beta)
    else:
        raise TypeError("beta must be either a float or a sequence of floats")


    # try to pass zeros into the model
    zero_feature = np.zeros(tce_calculator.feature_vector_size).reshape(1, -1)
    predicted = tce_calculator.models["energy"].predict(zero_feature).item()
    if predicted != 0.0:
        warnings.warn(
            "Input model has an intercept, which will mess with energy difference calculations. "
            "The monte carlo run will automatically zero-out this intercept, transforming your model.",
            UserWarning
        )
        
    transformed_model = transform_model(tce_calculator.models["energy"])

    energy = transformed_model.predict(
        tce_calculator.get_feature_vector(initial_configuration).reshape(1, -1)
    )
    
    trajectory = []
    for step in range(num_steps):
        callback(step, num_steps)

        if not step % save_every:
            to_save = initial_configuration.copy()
            to_save.info["energy"] = energy
            trajectory.append(to_save)
            LOGGER.info(f"saved configuration at step {step:.0f}/{num_steps:.0f}")

        new_configuration = mc_step(initial_configuration)
        feature_diff = tce_calculator.get_feature_vector_difference(
            initial_configuration, new_configuration
        ).reshape(1, -1)
        energy_diff = transformed_model.predict(feature_diff)
        energy_diff += energy_modifier(initial_configuration, new_configuration)

        if not isinstance(energy_diff, float):
            energy_diff = energy_diff.item()
        if np.exp(-beta_values[step] * energy_diff) > 1.0 - generator.random():
            LOGGER.debug(f"move accepted with energy difference {energy_diff}")
            initial_configuration = new_configuration
            energy += energy_diff

    return trajectory
