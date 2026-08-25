r"""
.. include:: ../README.md

# Examples

Below is a set of examples to help you quickstart your work 🙂

The basic workflow is:

- Load in your atomic configurations with energies attached
- Create a `TCECalculator`, which wraps the Atomic Simulation Environment (ase) package
- Train the calculator
- ...and deploy it! 🧪

Most of these examples include external packages. If you want to set up an appropriate environment to run these 
examples, run:

```
pip install tce-lib[examples]
```

## ⚛ ASE integration

The core object of this library is the `tce.calculator.TCECalculator` object. This object inherits from the
`ase.Calculator` parent class from the [Atomic Simulation Environment](https://docs.ase-lib.org/) (ASE), which serves
as a unifying framework for computing atomic properties from a large class of simulation software, including
[LAMMPS](https://en.wikipedia.org/wiki/LAMMPS), [VASP](https://vasp.at/), [MACE](https://mace-docs.readthedocs.io/),
and more.

The `tce.calculator.TCECalculator` object can then compute feature vectors $\mathbf{t}$, consisting of $m$-body cluster
counts:

$$ N_{\alpha_1\cdots\alpha_m}^{[\ell]} = T_{i_1\cdots i_m}^{[\ell]} \prod_{n=1}^m X_{i_n \alpha_n}$$

where $T_{i_1\cdots i_m}^{[\ell]}$ is the adjacency tensor for an order $m$ hyper-graph, i.e.:

$$ T_{i_1\cdots i_m}^{[\ell]} = [\text{sites $i_1$, $\cdots$, $i_m$ are in a cluster $[\ell]$}] $$

and $X_{i\alpha}$ denotes the occupation tensor:

$$ X_{i\alpha} = [\text{site $i$ is occupied by type $\alpha$}] $$

and $[\cdot]$ denotes the [Iverson bracket](https://en.wikipedia.org/wiki/Iverson_bracket).

You can compute these cluster counts directly integrated within ASE:

```py
.. include:: ../examples/using-ase.py
```

This creates a calculator that will compute cluster counts for the Fe-Cr system, which is bcc, up to 2rd nearest
neighbors, and including two types of three-body clusters, assuming a lattice parameter $a = 2.9\;\text{Å}$.

The grammar of the many-body features follows edge-labeling of $k$-cliques in the graph of the full solid. In plainer
English, the `(0, 0, 1)` cluster is an isosceles triangle with two first-nearest neighbor bonds and one
second-nearest neighbor bond.

You can use the same grammar to define $k$-body clusters as well. We will see later how to visualize these clusters
using OVITO!

## 🏋️‍♀️ Training + Monte Carlo

Below is a template for a standard workflow, using an Embedded Atom Method potential for the Cu-Ni alloy. This trains a
model using the cluster counts above:

$$ \mathcal{H}_\text{eff}(\mathbf{X}) = \sum_m \frac{1}{m!}\varepsilon_{\alpha_1\cdots\alpha_m}^{[\ell]} T_{i_1\cdots i_m}^{[\ell]}\prod_{n=1}^m X_{i_n\alpha_n} $$

where $\varepsilon_{\alpha_1\cdots\alpha_m}^{[\ell]}$ are fitting coefficients, corresponding to the energy of each
cluster type.

The first script is training a CuNi model using an EAM potential from Fischer et al.
(paper [here](https://doi.org/10.1016/j.actamat.2019.06.027)). In this script, we generate a bunch of random CuNi
solid solutions, attach an `ase.calculators.eam.EAM` calculator to each configuration, compute their energies, and
then train a `TCECalculator` instance. The calculator is then saved to be used for later.

We can do a very standard benchmark, which is a parity plot of the test set and the training set:

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/parity-plot.png"
    width=100%
    alt="CuNi SRO parameter from CE"
    title="SRO parameter"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/parity-plot.png)

**IMPORTANT**: These are unrelaxed energies! A real production environment should optimize the structure - see the
prior example on how to do this within a LAMMPS calculator.

```py
.. include:: ../examples/0-copper-nickel-training.py
```

For the sake of simplicity, we stuck with ase's EAM implementation, which is implemented in pure Python. This example
serves as a good template for using other methods to compute energies. For example, one could define a a `Calculator`
instance that wraps VASP:

```py
from ase.calculators.vasp import Vasp

calculator_constructor = lambda: Vasp(
    prec="Accurate",
    encut=500,
    istart=0,
    ismear=1,
    sigma=0.1,
    nsw=400,
    nelmin=5,
    nelm=100,
    ibrion=1,
    potim=0.5,
    isif=3,
    isym=2,
    ediff=1e-5,
    ediffg=-5e-4,
    lreal=False,
    lwave=False,
    lcharg=False
)
```

See ASE's documentation [here](https://ase-lib.org/ase/calculators/vasp.html) for how to properly set this up!

This pattern also works for more complex lattice structures - simply provide the neighbor cutoffs, and you're good to
go!

The next script uses the saved calculator to run a canonical Monte Carlo simulation on a $15\times 15\times 15$
supercell, storing the configuration (saved in an `ase.Atoms` object) every 1000 frames. We also set up a `logging`
configuration here, which will tell you how far-along the simulation is. Note that `trajectory` looks complicated, but
is just a list of `ase.Atoms` objects, so you have a lot of freedom to do what you wish with this trajectory later.

```py
.. include:: ../examples/1-copper-nickel-mc.py
```

You can also change the logging level here! For example, changing to `logging.DEBUG` will display more things:

```py
logging.getLogger("numba").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
```

`sparse` uses `numba` so you will also get a lot of `numba` logging messages if you don't manually set `numba`'s
logger to a higher level. You can also do much more advanced things - see this
[video by mCoding](https://www.youtube.com/watch?v=9L77QExPmI0) on the `logging` library.

The configurations generated by the MC run are then visualizable with a number of softwares, including
[OVITO](https://www.ovito.org/). An example of such a rendering is below:

<div style="padding:50% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/1117980384?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media; web-share" referrerpolicy="strict-origin-when-cross-origin" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="animation"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

Just from the animation, it doesn't look like much is happening at all. The animation is not the whole story, though -
you can also use the trajectory to do some analysis. We can use OVITO's Python library [here](https://pypi.org/p/ovito)
and any of its plugins to do some analysis, as if our files are from any other atomistic simulation software. Below
we'll compute the Cowley short range order parameter using the
`cowley-sro-parameters` plugin [here](https://pypi.org/p/cowley-sro-parameters) (shameless plug... I'm the author 🙂).

```py
.. include:: ../examples/2-copper-nickel-sro.py
```

This generates the plot below. A negative value indicates attraction between two atom types. So, the solution is
clearly not fully random! We might need a more than 100,000 steps too - this curve should bottom out once we
reach steady state. Note we can also just grab the potential energy from the `ase.Atoms` instances - the Monte Carlo
run stores this information using `ase.calculators.singlepoint.SinglePointCalculator` instances.

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cu-ni-sro.png"
    width=100%
    alt="CuNi SRO parameter from CE"
    title="SRO parameter"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cu-ni-sro.png)

We can also use the model to sample a different ensemble. For the prototypical canonical ensemble, the acceptance rule
for a swap with energy difference $\Delta E$ is $\exp(-\beta\Delta E) > u$, where $u$ is a random number drawn from
$[0, 1]$. For the grand canonical ensemble, the acceptance rule is instead:

$$ \exp\left(-\beta\left(\Delta E - \sum_\alpha \mu_\alpha \Delta N_\alpha\right)\right) = \exp\left(-\beta\left(\Delta E - \boldsymbol{\mu}\cdot\Delta \mathbf{N}\right)\right) > u $$

where $\mu_\alpha$ is the chemical potential of type $\alpha$ and $\Delta N_\alpha$ is the change in the number of
$\alpha$ atoms in the swap. You can inject this into `tce.monte_carlo.monte_carlo` by defining an `energy_modifier`,
which adds a term to $\Delta E$:

```py
.. include:: ../examples/1-copper-nickel-mc2.py
```

We also can specify our own Monte Carlo step, which is done above. This plot generates a curve which is useful for
computing phase diagrams:

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cu-ni-sgcmc.png"
    width=100%
    alt="CuNi SGCMC curve"
    title="CuNi curve"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/cu-ni-sgcmc.png)

Note that the curve is continuous, which denotes no phase transitions at the temperature. This matches experimental
phase diagrams - CuNi forms a solid solution along the whole composition range below the melting point.

<div style="text-align: center;">
  <a href="https://www.doitpoms.ac.uk/miclib/phase_diagrams/imagesPD/CuNi2.png">
    <img
      src="https://www.doitpoms.ac.uk/miclib/phase_diagrams/imagesPD/CuNi2.png"
      width="50%"
      alt="CuNi phase diagram"
      title="CuNi"
    />
  </a>
</div>

Image credit: DoITPoMS @ University of Cambridge ([url](https://www.doitpoms.ac.uk/miclib/phase_diagrams.php?id=11))

## 💻 Custom Training (Advanced)

Below is an example of using a custom training method to train the CE model. There are many reasons one might want to do
this. The example below is a very typical one - using [lasso](https://en.wikipedia.org/wiki/Lasso_(statistics)). This
regularization technique minimizes the loss:

$$ L(\beta\; |\;\lambda) = \|X\beta - y\|_2^2 + \lambda \|\beta\|_1 $$

which better-supports sparse best-fit parameters $\hat{\beta}$, which may be useful if you only want to exclude
non-important clusters. We'll use `scikit-learn`'s interface for providing a model. You can really use any linear
model here (without an intercept...), see `scikit-learn`'s docs
[here](https://scikit-learn.org/stable/modules/linear_model.html) for more examples of these.

```py
.. include:: ../examples/3-sklearn-fitting.py
```

This script (it will be quite slow...) will calculate the number of nonzero cluster interaction coefficients as a
function of the regularization parameter. For larger regularization parameters, the number of nonzero coefficients
should decrease. This is a useful technique if you explicitly want to exclude clusters that are relatively unimportant,
but are not sure which clusters should be included.

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/regularization.png"
    width=100%
    alt="Lasso regularization"
    title="Lasso"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/regularization.png)


## 🧲 Learning tensorial vs. scalar property

In general, one might also want to learn tensorial properties. This can be done by vectorizing the property in some
way, like [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation):

$$ \sigma = (\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{yz}, \sigma_{xz}, \sigma_{xy}) $$

Below is an example of adding a new target property, namely stress. It also showcases an important point
about `tce-lib`: our feature vectors are **extensive**, not intensive like other CE libraries. This matters when
training on intensive properties, like stress. Here, we can inject custom behavior, i.e. train on intensive features.
Of course, it is also fine to use this same pattern to train a CE model on other scalar properties. Using this new
target property, we can predict things like enthalpy.

```py
.. include:: ../examples/4-tensorial-property.py
```

## 🔔 Callback functionality

The `tce.monte_carlo.monte_carlo` routine also has a `callback` argument that lets you inject a notification system
into the Monte Carlo run. This argument needs to be a function with signature:

```py
def callback(step: int, num_steps: int) -> None:
    ...
```

If it is not provided, it defaults to calling the `logging` library:

```py
import logging

LOGGER = logging.getLogger(__name__)

def callback(step_: int, num_steps_: int):
    LOGGER.info(f"MC step {step_:.0f}/{num_steps_:.0f}")
```

But, you can do very cool things with this. It's a bit of a cute example that might not be practical, but you can
send notifications to third party systems like [Discord](https://en.wikipedia.org/wiki/Discord) using webhooks.

```py
.. include:: ../examples/5-callbacking.py
```

which will send a notification in whatever Discord channel once the MC run is finished. See
[Discord's documentation on webhooks](https://support.discord.com/hc/en-us/articles/228383668-Intro-to-Webhooks) for
a tutorial on how to set up your own webhook URL. You can get really creative here too, like Slack's similar
functionality [here](https://docs.slack.dev/messaging/sending-messages-using-incoming-webhooks/), or the
[Gmail API](https://developers.google.com/workspace/gmail/api/guides). None of these are particularly useful for what I
have done above (sending a single email once the run is finished), but really shine for long runs where you want to
be periodicially notified.

## 🕵️ Loading and Visualizing Datasets

Below is an example of using one of our pre-set training datasets using `tce.datasets`. When you install `tce-lib`, you
automatically install some toy datasets that are mostly of pedagogical benefit, i.e. you can look at one of these datasets 
and see examples of what you can train on. Since `ovito` has an `ase` interface, you can also use `ovito` to visualize the 
dataset, which might be of interest.

```py
.. include:: ../examples/load-dataset-and-visualize.py
```

which generates the figure below:

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/visualized.png"
    width=100%
    alt="TaW dataset visualization from genetic algorithm"
    title="TaW"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/visualized.png)

Each dataset has a metadata object as well, which is stored in `json` format:

```json
.. include:: ../tce/datasets/tungsten_tantalum_genetic/metadata.json
```

which tells you some info, and (should) additionally give you contact information to inquire about the dataset. Note
that `tce.datasets.Dataset.configurations` is of type `list[ase.Atoms]`, so you can directly plug this into a training
routine. Please contact me directly (email above) if you have datasets you would like to be added 😊

## 💎 Exotic lattice structures

ASE has quite a large suite for initializing different lattice structures. Here, the pattern does not change: simply
provide your `ase.Atoms` instance and neighbor cutoffs, as well as any many-body terms you're interested in. See an
example below for an Si-Ge alloy within a cubic diamond structure:

```py
.. include:: ../examples/exotic-lattice.py
```

or for a fluorite structure with multiple sublattices:

```py
.. include:: ../examples/exotic-lattice2.py
```

You'll notice that a lot of the features are $0$. This is not uncommon for exotic lattice types, especially when not
all lattice sites are equivalent. This is not a problem - we just likely need to feature reduce later using something
like PCA, which is relatively easy with an sklearn.pipeline.Pipeline object.

## 🔬 OVITO Plugin

We've implemented a plugin for OVITO Pro that allows you to visualize clusters in your system. This is a very useful tool for 
debugging your cluster expansion model, and for visualizing the clusters that are being used in your model. You can find 
the plugin [here](https://github.com/jwjeffr/tce-modifier). This plugin can used within the OVITO Pro software, or within 
a stand-alone Python script using the OVITO Python API. See the README within that repository for more instructions! The stand-alone 
Python script is useful for generating images of clusters for use in publications, such as the figure below:

[<img
    src="https://raw.githubusercontent.com/jwjeffr/tce-modifier/refs/heads/main/examples/ws2-grid/grid.png"
    width=100%
    alt="WS2 feature grid"
    title="WS2"
/>](https://raw.githubusercontent.com/jwjeffr/tce-modifier/refs/heads/main/examples/ws2-grid/grid.png)

See the example [here](https://github.com/jwjeffr/tce-modifier/tree/main/examples/ws2-grid) for the full script that creates this grid.

# Sharp Edges

`tce-lib` has a couple of sharp edges (or gotcha's) that one needs to look out for.

## Extensivity of features

In traditional cluster expansion packages, correlation functions are usually intensive (i.e. independent of size). This
is not the case for `tce-lib`. We can showcase this by creating some feature vectors for an FCC lattice of varying
sizes:

```py
.. include:: ../examples/size-dependence.py
```

[<img
    src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/size-dependence.png"
    width=100%
    alt="Size dependence of features"
    title="Size dependence"
/>](https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/examples/size-dependence.png)

So, be careful if you are training an intensive property! By default, you will be training intensive properties on
extensive features, which does not make sense. You can fix this by training on an equivalent extensive feature, and
then make the property intensive later, or more preferably defining features with a strategy that makes them intensive,
as done in [the stress example above](https://muexly.github.io/tce-lib/tce.html#learning-a-tensorial-property).

## Loading your own data

You've probably noticed that the input to the typical training routines is a list of configurations, rather than a 
list of configurations and a list of energies like one might expect. This is because we can store energy inside of 
the `ase.Atoms` object that represents a configuration.

This makes it easier to generate data - but what if you already have data sitting around? Most atomistic software
(all that I am familiar with) stores the energy separate from the configuration. You can load these in by attaching a
`ase.calculators.singlepoint.SinglePointCalculator` to a configuration. For example, if I have a directory with data 
that looks like:

```
data/
├── run1/
│   ├── configuration.xyz
│   └── energy.txt
├── run2/
│   ├── configuration.xyz
│   └── energy.txt
├── run3/
│   ├── configuration.xyz
│   └── energy.txt
├── run4/
│   ├── configuration.xyz
│   └── energy.txt
├── run5/
│   ├── configuration.xyz
│   └── energy.txt
```

you can load in this dataset using `ase` entirely:

```py
from pathlib import Path

from ase import io, Atoms
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

configurations: list[Atoms] = []
for path in Path("data").iterdir():
    configuration: Atoms = io.read(path / "configuration.xyz", format="extxyz")
    energy: float = np.loadtxt(path / "energy.txt")
    configuration.calc = SinglePointCalculator(configuration, energy=energy)
    configurations.append(configuration)

# do whatever with tce here
...
```

where I have assumed that all configurations are of [Extended XYZ format](https://ase-lib.org/ase/io/formatoptions.html#extxyz). 
It is very easy to, however, load in a different format. For example, replace `"extxyz"` with `"vasp"` if you have `POSCAR` files 
generated by VASP, or with `"lammps-data"` if you have LAMMPS data files. There is quite a large set of supported formats, 
which you can find [here](https://ase-lib.org/ase/io/io.html#ase.io.read).

## Fitting with an intercept → MC

If you choose to fit using `scikit-learn`, models will include an intercept by default. This is not a problem if you
just want to compute energies, but you likely want to feed this into `tce.monte_carlo.monte_carlo` later, which expects a linear 
transformation, rather than an affine transformation.

If your model is affine, i.e. includes an intercept:

$$ f(\mathbf{t}) = \alpha + \boldsymbol{\beta}^\intercal\mathbf{t} $$

Then, the resulting energy difference will be computed as:

$$ f(\Delta\mathbf{t}) = \alpha + \boldsymbol{\beta}^\intercal\Delta\mathbf{t} $$

which is invalid unless $\alpha = 0$. To address this, the Monte Carlo function will remove this intercept by probing
basis vectors, i.e. by evaluating the intercept and subtracting out that intercept:

$$ f_{\text{new}}(\Delta\mathbf{t}) = f(\Delta\mathbf{t}) - f(\mathbf{0}) $$

where $\mathbf{I}$ is the identity matrix.

"""

__version__ = "1.0.3"
__authors__ = ["Jacob Jeffries"]

__url__ = "https://github.com/MUEXLY/tce-lib"

import warnings

from . import calculator as calculator
from . import constants as constants
from . import datasets as datasets
from . import monte_carlo as monte_carlo
from . import topology as topology
from . import training as training
from . import citations as citations


if __version__.startswith("0."):
    warnings.simplefilter("once", UserWarning)

    warnings.warn(
        f"{__name__} is in alpha. APIs are unstable and may change without notice. "
        f"Please report any problems at {__url__}/issues",
        UserWarning,
        stacklevel=2,
    )
