import logging
from typing import Callable, Optional
from functools import wraps
from string import Template


LOGGER = logging.getLogger(__name__)
ORIGINAL_PAPER: str = "https://doi.org/10.1016/j.commatsci.2025.114338"
KMC_PAPER: str = "https://arxiv.org/abs/2605.23612"


def cite(
    paper_link: str,
    msg_template: Optional[Template] = None
) -> Callable[[Callable], Callable]:

    r"""
    Function decorator to log a citation message. Example usage:

    ```py
    from tce.citations import cite

    @cite(paper_url="https://google.com")
    def add(x, y):
        return x + y
    ```

    Then, the first call to `add` will log a citation message to the user.

    Args:
        paper_link (str):
            The url to cite
        msg_template (Optional[Template]):
            The template for the message. This template must have the following keys:
            
            - `${fn_name}` representing the name of the function it wraps
            - `${url}` representing the url within the message
            
            If not supplied, defaults to:
            ```py
            from string import Template
            msg_template = Template(
                "Function ${fn_name} uses the work at ${url}. Please cite it."
            )
            ```
    """

    if not msg_template:
        msg_template = Template(
            "Function ${fn_name} uses the work at ${url}. Please cite it."
        )

    name_and_urls: set[tuple[str, str]] = set()

    def decorator(fn: Callable) -> Callable:

        @wraps(fn)
        def wrapper(*args, **kwargs):
            
            key = (fn.__name__, paper_link)
            if key not in name_and_urls:
                name, url = key
                msg = msg_template.substitute({"fn_name": name, "url": url})
                LOGGER.info(msg)
                name_and_urls.add(key)

            return fn(*args, **kwargs)

        return wrapper

    return decorator
