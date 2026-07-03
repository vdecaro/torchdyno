.. toctree::
   :maxdepth: 6
   :hidden:

   Overview <source/overview/index.rst>
   Reference <source/api/index.rst>



.. image:: _static/images/logo_no_name.png
   :align: left
   :width: 25%
   :class: only-light

.. image:: _static/images/logo_no_name.png
   :align: left
   :width: 25%
   :class: only-dark


======================
**Welcome to TorchDyno**
======================
TorchDyno is a PyTorch-based library for the implementation of dynamical systems. It provides a simple and flexible way to build and train non-standard recurrent models, such as Reservoir Computing models. TorchDyno is designed to be easy to use, efficient, and flexible. It is built on top of PyTorch, which makes it easy to integrate with other PyTorch-based libraries.

.. grid:: 12

   .. grid-item::
      :columns: auto

      .. button-ref:: source/overview/index
         :ref-type: myst
         :outline:
         :class: start-button

         :octicon:`desktop-download;1em;info` Install

   .. grid-item::
      :columns: auto

      .. button-ref:: source/api/index
         :ref-type: myst
         :outline:
         :class: start-button

         :octicon:`book;1em;info` References


================
**Installation**
================
First, install the PyTorch build for your platform by following the
`PyTorch install guide <https://pytorch.org/get-started/locally/>`_. Then install
TorchDyno with pip:

.. code-block:: bash

   pip install torchdyno

Some bundled example datasets need extra dependencies (SequentialMNIST needs
``torchvision``, HHAR needs ``pandas``); install them via the ``datasets`` extra:

.. code-block:: bash

   pip install "torchdyno[datasets]"

To install the latest version from source, clone the repository and use uv (via the Makefile):

.. code-block:: bash

   git clone https://github.com/vdecaro/torchdyno
   cd torchdyno
   make setup

===========
**Credits**
===========
We thank `eclypse-org <https://github.com/eclypse-org>`_ and `Jacopo Massa <https://github.com/jacopo-massa>`_ for the structure and the template of the documentation!