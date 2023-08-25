Chronica
========

.. image:: https://img.shields.io/pypi/v/chronica?logo=pypi&logoColor=white
   :target: https://pypi.org/project/chronica/

.. image:: https://pkg.go.dev/badge/github.com/9rum/chronica.svg
   :target: https://pkg.go.dev/github.com/9rum/chronica

.. image:: https://img.shields.io/pypi/l/chronica?logo=apache
   :target: https://github.com/9rum/chronica/blob/master/LICENSE

.. inclusion-marker-start-do-not-remove

|

Chronica is a data-imbalance-aware scheduler and associated scheduling framework for distributed deep learning.
Chronica accelerates data-parallel training with imbalanced data such as video, audio, and text by reducing unnecessary operations and memory usage while improving scalability and resource utilization.
The goal of Chronica is to make a fast, efficient and easy-to-use data scheduling framework, compatible with existing deep learning systems such as TensorFlow, PyTorch, JAX and DGL.

|

.. contents::

|

Highlighted features
--------------------
Data-imbalance-aware scheduling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chronica provides several schedule kinds such as **static** and **dynamic**, each of which can be useful for environments with different characteristics.

Partitioned data set
^^^^^^^^^^^^^^^^^^^^

In addition to the traditional fully sharded data sets, Chronica supports a **partitioned data set** where the data is split into multiple data partitions across nodes in the cluster.

Minimal code modifications
^^^^^^^^^^^^^^^^^^^^^^^^^^

While benefiting from the above features, users only need to modify a few lines in the existing code base, and the rests are transparent.

Supported frameworks
--------------------

Chronica currently supports PyTorch only.

Quick start
-----------
Prerequisites
^^^^^^^^^^^^^

You need to have the latest version of the `Go <https://go.dev/>`_ compiler installed on the master node (rank 0) and add the **GOBIN** environment variable to **PATH**, *i.e.*, ``PATH=$GOBIN:$PATH``.
Once the Go compiler is installed, you can install the **chronica** pip package.

.. code-block:: bash

    $ pip install chronica

How to use
^^^^^^^^^^

To use Chronica, make the following modifications to your program:

#. Make your data set inherit from Chronica's data set class instead of existing data set class:
   *e.g.*, for PyTorch, use ``chronica.torch.utils.data.Dataset`` instead of ``torch.utils.data.Dataset``.

#. Overwrite ``__sizeof__`` in your data set, which represents the relative size of each data sample:
   *e.g.*, for video data sets, the relative size of each data sample is determined by the number of frames.
   Thus you can overwrite ``__sizeof__`` using OpenCV as follows.

   .. code-block:: python

       def __sizeof__(self, index: int) -> int:
           return int(cv2.VideoCapture(self.videos[index]).get(cv2.CAP_PROP_FRAME_COUNT))

#. Use Chronica's data sampler instead of existing data sampler:
   *e.g.*, for PyTorch, use ``chronica.torch.utils.data.DistributedSampler`` instead of ``torch.utils.data.DistributedSampler``.

#. Pass additional parameters to the data sampler:
   *e.g.*, set ``kind="dynamic"`` if you need dynamic scheduling.

In most cases, the above modifications can be done by adding ``chronica.`` to import statements and overwriting ``__sizeof__``, as shown below:

.. code-block:: diff

    -from torch.utils.data import Dataset
    +from chronica.torch.utils.data import Dataset

    +    def __sizeof__(self, index: int) -> int:
    +        return int(cv2.VideoCapture(self.videos[index]).get(cv2.CAP_PROP_FRAME_COUNT))

    -from torch.utils.data import DistributedSampler
    +from chronica.torch.utils.data import DistributedSampler

Publications
------------

#. Sanha Maeng, Gordon Euhyun Moon and Sungyong Park, `Chronica: A Data-Imbalance-Aware Scheduler for Distributed Deep Learning <https://ieeexplore.ieee.org/document/10171495>`_, in Proceedings of the `2023 IEEE/ACM 23rd International Symposium on Cluster, Cloud and Internet Computing (CCGrid) <https://ccgrid2023.iisc.ac.in/>`_.

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
