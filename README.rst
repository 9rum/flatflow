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

Highlighted Features
--------------------
Data-imbalance-aware scheduling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Heterogeneous data partitioning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Minimal code modifications
^^^^^^^^^^^^^^^^^^^^^^^^^^



Installation
------------
Prerequisites
^^^^^^^^^^^^^

You need to have the `Go compiler <https://go.dev/>`_, version 1.20 or higher to be installed on the master node (rank 0) and include the `GOBIN` environment variable to `PATH`.
i.e., ``PATH=$GOBIN:$PATH``

You can easily install the ``chronica`` pip package by running ``pip install chronica``.

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
