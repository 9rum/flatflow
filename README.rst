Chronica
--------

.. image:: https://pkg.go.dev/badge/github.com/9rum/chronica.svg
   :target: https://pkg.go.dev/github.com/9rum/chronica

.. image:: badge.fury.io/py/~.svg
   :target: badge.fury.io/py/~

.. image:: zenodo.org/~.svg
   :target: zenodo.org/~

.. inclusion-marker-start-do-not-remove

|

Chronica is a data-imbalance-aware scheduler for distributed deep learning.
Chronica accelerates data-parallel training with imbalanced data such as video, audio, and text by reducing unnecessary operations and memory usage while improving scalability and resource utilization.
This is an artifact and open source implementation of our paper `Chronica: A Data-Imbalance-Aware Scheduler for Distributed Deep Learning <https://ieeexplore.ieee.org/document/10171495>`_, which has been proposed in the `23rd IEEE/ACM International Symposium on Cluster, Cloud and Internet Computing <https://ccgrid2023.iisc.ac.in/>`_ (CCGrid'23).

Highlighted Features
^^^^^^^^^^^^^^^^^^^^

Data-imbalance-aware scheduling

 - static/dynamic

Heterogeneous data partitioning

Minimal code modification

Get Started
^^^^^^^^^^^

Prerequisites
^^^^^^^^^^^^^

1.20 <= Go, GOPATH, GOBIN on master (rank 0)

Install
^^^^^^^

> pip install chronica

Cite
^^^^

If you ...

@INPROCEEDINGS{10171495,
  author={Maeng, Sanha and Moon, Gordon Euhyun and Park, Sungyong},
  booktitle={2023 IEEE/ACM 23rd International Symposium on Cluster, Cloud and Internet Computing (CCGrid)}, 
  title={Chronica: A Data-Imbalance-Aware Scheduler for Distributed Deep Learning}, 
  year={2023},
  volume={},
  number={},
  pages={262-272},
  doi={10.1109/CCGrid57682.2023.00033}}

License
^^^^^^^

Chronica is distributed under the terms of the Apache-2.0 license.
