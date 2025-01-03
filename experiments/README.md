# FlatFlow Experiment Scripts

- The script and related modules were written based on `nvcr.io/nvidia/nemo:24.05` image, then slightly updated to reflect changes of `nemo:24.09`.

## Step 0: Dataset & Model Preparation
- Dolly
- Alpaca
- QuALITY (for long context)

## Step 1: Multi-node Training
- See the script for training arguments.
- Run `bash step1_training/run_train_multi_nodes_2405.sh`.

## Step 2: Evaluation
- IFEval
- ARC
- (QuALITY)
