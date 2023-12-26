<h1><p align="center">Chronica</p></h1>

<p align="center">
  <a href="https://pypi.org/project/chronica/">
    <img src="https://img.shields.io/pypi/v/chronica?logo=pypi&logoColor=white">
  </a>
  <a href="https://pkg.go.dev/github.com/9rum/chronica">
    <img src="https://pkg.go.dev/badge/github.com/9rum/chronica.svg">
  </a>
  <a href="https://github.com/9rum/chronica/blob/master/LICENSE">
    <img src="https://img.shields.io/pypi/l/chronica?logo=apache">
  </a>
</p>

Chronica is a data-imbalance-aware scheduler and associated scheduling framework for distributed deep learning.
Chronica accelerates data-parallel training with imbalanced data such as video, audio, and text by reducing unnecessary operations and memory usage while improving scalability and resource utilization.
The goal of Chronica is to make a fast, efficient and easy-to-use data scheduling framework, compatible with existing deep learning systems such as TensorFlow, PyTorch, JAX and DGL.

## Highlighted features

### Data-imbalance-aware scheduling

Chronica provides several schedule kinds such as **static** and **dynamic**, each of which can be useful for workloads with different characteristics.

### Partitioned data set

In addition to the traditional fully sharded data sets, Chronica supports a **partitioned data set** where the data is split into multiple data partitions across nodes in the cluster.

### Minimal code modifications

While benefiting from the above features, users only need to modify a few lines in the existing code base, and the rests are transparent.

## Supported frameworks

Chronica currently supports PyTorch only.

## Quick start

### Prerequisites

You need to have the latest version of the [Go](https://go.dev/) compiler installed on the master node (rank 0) and add the **GOBIN** environment variable to **PATH**, *i.e.*, ``PATH=$GOBIN:$PATH``.
Once the Go compiler is installed, you can install the **chronica** pip package.

```sh
pip install chronica
```

### How to use

To use Chronica, make the following modifications to your program:

1. Make your data set inherit from Chronica's data set class instead of existing data set class:
   *e.g.*, for PyTorch, use ``chronica.torch.utils.data.Dataset`` instead of ``torch.utils.data.Dataset``.

1. Overwrite ``__sizeof__`` in your data set, which represents the relative size of each data sample:
   *e.g.*, for video data sets, the relative size of each data sample is determined by the number of frames.
   Thus you can overwrite ``__sizeof__`` using OpenCV as follows.

   ```python
   def __sizeof__(self, index: int) -> int:
       return int(cv2.VideoCapture(self.videos[index]).get(cv2.CAP_PROP_FRAME_COUNT))
   ```

1. Use Chronica's data sampler instead of existing data sampler:
   *e.g.*, for PyTorch, use ``chronica.torch.utils.data.DistributedSampler`` instead of ``torch.utils.data.DistributedSampler``.

1. Pass additional parameters to the data sampler:
   *e.g.*, set ``kind="dynamic"`` if you need dynamic scheduling.

> [!TIP]
> In most cases, the above modifications can be done by adding ``chronica.`` to import statements and overwriting ``__sizeof__``, as shown below:

```diff
-from torch.utils.data import Dataset
+from chronica.torch.utils.data import Dataset

+    def __sizeof__(self, index: int) -> int:
+        return int(cv2.VideoCapture(self.videos[index]).get(cv2.CAP_PROP_FRAME_COUNT))

-from torch.utils.data import DistributedSampler
+from chronica.torch.utils.data import DistributedSampler
```

## Publications

1. Sanha Maeng, Gordon Euhyun Moon and Sungyong Park, [Chronica: A Data-Imbalance-Aware Scheduler for Distributed Deep Learning](https://ieeexplore.ieee.org/document/10171495), in Proceedings of the [2023 IEEE/ACM 23rd International Symposium on Cluster, Cloud and Internet Computing (CCGrid)](https://ccgrid2023.iisc.ac.in/)

## Citation

If you use Chronica in your work, we would appreciate citations using the following metadata:

```bibtex
@inproceedings{maeng2023chronica,
  author={Sanha Maeng, Gordon Euhyun Moon and Sungyong Park},
  booktitle={2023 IEEE/ACM 23rd International Symposium on Cluster, Cloud and Internet Computing (CCGrid)},
  title={Chronica: A Data-Imbalance-Aware Scheduler for Distributed Deep Learning},
  year={2023},
  pages={262--272},
  doi={10.1109/CCGrid57682.2023.00033}
}
```
