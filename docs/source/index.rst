.. RLMeta documentation master file, created by
   sphinx-quickstart on Wed Dec 29 14:57:09 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RLMeta's documentation!
==================================
**RLMeta** is a light-weight flexible framework for Distributed Reinforcement 
Learning Research based on `PyTorch <https://pytorch.org/>`_ and 
`moolib <https://github.com/facebookresearch/moolib>`_.

.. note::
   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Installation
============

To install RLMeta, please install `PyTorch`_ first. It is highly recommanded to
install PyTorch with CUDA to achieve best performance.

To build from souce run:

.. code-block:: bash

   $ git clone https://github.com/facebookresearch/rlmeta
   $ cd rlmeta
   $ git submodule sync && git submodule update --init --recursive
   $ pip install -e .


Getting Started
===============

.. toctree::
   getting_started.rst


Tutorials
=========

Comming soon...


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
