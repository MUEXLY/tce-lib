# tce-lib

<p align="center">
  <a href="https://muexly.github.io/tce-lib">
    <img src="https://img.shields.io/badge/docs-orange?logo=github&logoColor=green&label=gh-pages" style="height: 48px;" alt="Documentation Website">
  </a>
</p>

<hr>

<p align="left">
  <a href="https://pypi.org/project/tce-lib/"><img src="https://img.shields.io/pypi/v/tce-lib?color=blue" style="height: 22px;" alt="Stable Version"></a>
  <a href="https://en.wikipedia.org/wiki/MIT_License"><img src="https://img.shields.io/badge/License-MIT-8A2BE2" style="height: 22px;" alt="Static Badge"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" style="height: 22px;" alt="Linting: Ruff"></a>
  <a href="https://mypy-lang.org/"><img src="https://www.mypy-lang.org/static/mypy_badge.svg" style="height: 22px;" alt="Checked with mypy"></a>
  <a href="https://docs.pytest.org/en/stable/"><img src="https://img.shields.io/badge/pytest-tested-blue?logo=pytest" style="height: 22px;" alt="Tested with pytest"></a>
  <a href="https://deepwiki.com/MUEXLY/tce-lib"><img src="https://deepwiki.com/badge.svg" style="height: 22px;" alt="Ask DeepWiki"></a>
</p>

<img src="https://raw.githubusercontent.com/MUEXLY/tce-lib/refs/heads/main/assets/logo.png" alt="tce-lib logo" style="width:50%;height:auto;">


## 🔎 What is tce-lib?

`tce-lib` is a library for creating and deploying tensor cluster expansion models of concentrated alloys following
our work published [here](https://doi.org/10.1016/j.commatsci.2025.114338). The core philosophy of `tce-lib` is to respect the 
[strategy pattern](https://en.wikipedia.org/wiki/Strategy_pattern) as core to the library's functionality. This design
pattern stages workflows as sequences of strategies, of which the user can override each. This allows for the majority 
of users to plug-and-play for an ordinary workflow, while still supporting fine-grained autonomy for more advanced 
users. 

## 📩 Installation

`tce-lib` is installable via the Python Package Index:

```shell
pip install tce-lib
```

or, from source:

```shell
git clone https://github.com/MUEXLY/tce-lib
pip install -e tce-lib/
```

## 📌 Citation

Please cite our work [here](https://doi.org/10.1016/j.commatsci.2025.114338) if you use `tce-lib` in your work.

## 💙 Acknowledgements

Authors acknowledge support from the U.S. Department of Energy, Office of Basic Energy Sciences, Materials Science and Engineering Division under Award No. DE-SC0022980.

## 🐝 Found a bug?

Please open an issue [here](https://github.com/MUEXLY/tce/issues), with a description of the issue and a [minimal, reproducible example](https://stackoverflow.com/help/minimal-reproducible-example) of the issue.

## 📑 License

`tce-lib` is released under the MIT license.
