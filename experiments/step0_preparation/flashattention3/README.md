## How to setup of FlashAttention3

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install

# Advise : This cp logic is based on the condition that you are running nemo:24.09 so if python version is different please modify to your right version.
cp step0_preparation/flashattention3/megatron_core_attention.py /usr/local/lib/python3.10/dist-packages/transformer_engine/pytorch/attention.py
```