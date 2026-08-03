Contributing
============

We welcome contributions to StatQA! This guide will help you get started.

Development Setup
-----------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/gojiplus/statqa.git
   cd statqa

If you do not have push access, fork the repository on GitHub first and clone
your fork instead.

2. Install development dependencies:

.. code-block:: bash

   uv sync --all-extras

3. Install pre-commit hooks:

.. code-block:: bash

   pre-commit install

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=statqa --cov-report=html

   # Run specific test markers
   pytest -m "not slow"  # Exclude slow tests
   pytest -m "not llm"   # Exclude LLM API tests

Code Quality
------------

We use several tools to maintain code quality:

.. code-block:: bash

   # Format code
   ruff format statqa tests

   # Lint code  
   ruff check statqa tests

   # Type checking
   pyright

   # Docstring linting
   pydoclint statqa

Pull Request Guidelines
-----------------------

1. **Create a feature branch** from main
2. **Add tests** for new functionality  
3. **Update documentation** if needed
4. **Ensure all checks pass** (tests, linting, type checking)
5. **Write clear commit messages**
6. **Submit pull request** with description of changes

Documentation
-------------

Documentation is built with Sphinx:

.. code-block:: bash

   cd docs
   make html

The documentation is automatically deployed on pushes to main.

Release Process
---------------

1. Update version in ``pyproject.toml``
2. Update ``CHANGELOG.md`` 
3. Create release tag
4. GitHub Actions handles PyPI publishing