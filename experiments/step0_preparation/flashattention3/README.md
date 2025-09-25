## How to setup of FlashAttention3

```bash
To use flash-attn v3, please use the following commands to install:
(1) pip install "git+https://github.com/Dao-AILab/flash-attention.git#egg=flashattn-hopper&subdirectory=hopper"
(2) python_path=`python -c "import site; print(site.getsitepackages()[0])"`
(3) mkdir -p $python_path/flash_attn_3
(4) wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/hopper/flash_attn_interface.py

# Advise : This cp logic is based on the condition that you are running nemo:24.09 so if python version is different please modify to your right version.
cp step0_preparation/flashattention3/megatron_core_attention.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention.py
```