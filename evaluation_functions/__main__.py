"""Allow ``python -m evaluation_functions`` as scorer shorthand."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
