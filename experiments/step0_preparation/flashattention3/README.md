## How to use FlashAttention-3

Run the following commands to install FlashAttention-3:

```bash
$ pip install "git+https://github.com/Dao-AILab/flash-attention.git#egg=flashattn-hopper&subdirectory=hopper"
$ python_path=`python -c "import site; print(site.getsitepackages()[0])"`
$ mkdir -p $python_path/flash_attn_3
$ wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/hopper/flash_attn_interface.py
```

> [!NOTE]
> This cp logic assumes that you are running nemo:24.09 so if Python version is different, please modify to your right version.

```bash
$ cp experiments/step0_preparation/flashattention3/attention.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention.py
```
