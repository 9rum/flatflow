Chronica
========

.. image:: https://pkg.go.dev/badge/github.com/9rum/chronica.svg
   :target: https://pkg.go.dev/github.com/9rum/chronica

.. image:: badge.fury.io/py/~.svg
   :target: badge.fury.io/py/~

.. image:: zenodo.org/~.svg
   :target: zenodo.org/~

.. inclusion-marker-start-do-not-remove

|

Chronica is a data-imbalance-aware scheduler and associated scheduling framework for distributed deep learning.
Chronica accelerates data-parallel training with imbalanced data such as video, audio, and text by reducing unnecessary operations and memory usage while improving scalability and resource utilization.
The goal of Chronica is to make a fast, efficient and easy-to-use data scheduling framework, compatible with existing deep learning systems such as TensorFlow, PyTorch, JAX and DistDGL.

|

.. contents::

|

Features
--------
Data-imbalance-aware scheduling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chronica provides several schedule clauses such as **static** and **dynamic**, each of which can be useful for environments with different characteristics.

Partitioned data set
^^^^^^^^^^^^^^^^^^^^

In addition to the traditional fully sharded data sets, Chronica supports a partitioned data set where the data is split into multiple data partitions across nodes in the cluster.

Minimal code modifications
^^^^^^^^^^^^^^^^^^^^^^^^^^

While benefiting from the above features, users only need to modify a few lines in the existing code base, and the rests are transparent.

Installation
------------
Prerequisites
^^^^^^^^^^^^^

You need to have the `Go <https://go.dev/>`_ compiler, version 1.20 or higher to be installed on the master node (rank 0) and add the ``GOBIN`` environment variable to ``PATH``, *i.e.*, ``PATH=$GOBIN:$PATH``.
Once the Go compiler is installed, you can install the ``chronica`` pip package.

.. code-block:: bash

    $ pip install chronica

Usage
^^^^^



Publications
------------

#. Sanha Maeng, Gordon Euhyun Moon and Sungyong Park, `Chronica: A Data-Imbalance-Aware Scheduler for Distributed Deep Learning <https://ieeexplore.ieee.org/document/10171495>`_, In Proceedings of the `2023 IEEE/ACM 23rd International Symposium on Cluster, Cloud and Internet Computing (CCGrid) <https://ccgrid2023.iisc.ac.in/>`_.

Citation
--------
If you use Chronica in your publications, we would appreciate citations to the following paper:

::

    @inproceedings{maeng2023chronica,
        author={Sanha Maeng, Gordon Euhyun Moon and Sungyong Park},
        booktitle={2023 IEEE/ACM 23rd International Symposium on Cluster, Cloud and Internet Computing (CCGrid)}, 
        title={Chronica: A Data-Imbalance-Aware Scheduler for Distributed Deep Learning}, 
        year={2023},
        pages={262-272},
        doi={10.1109/CCGrid57682.2023.00033}
    }
