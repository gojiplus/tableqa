Installation
============

Basic Installation
------------------

.. code-block:: bash

   pip install statqa

With Optional Features
----------------------

.. code-block:: bash

   # Include LLM support (OpenAI/Anthropic)
   pip install statqa[llm]

   # Include PDF parsing
   pip install statqa[pdf]

   # Include statistical formats (SPSS/Stata/SAS)
   pip install statqa[statistical-formats]

   # Every optional feature
   pip install statqa[all]

From Source
-----------

Development tooling lives in PEP 735 dependency groups rather than extras, so
``uv sync`` installs it:

.. code-block:: bash

   git clone https://github.com/gojiplus/statqa.git
   cd statqa
   uv sync --all-groups --all-extras

Development Environment
-----------------------

For development, we recommend using ``uv`` for faster dependency management:

.. code-block:: bash

   # Install uv first
   pip install uv

   # Clone and setup
   git clone https://github.com/gojiplus/statqa.git
   cd statqa
   uv sync --all-extras
