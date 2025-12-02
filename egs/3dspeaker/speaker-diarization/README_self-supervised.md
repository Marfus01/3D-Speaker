# Self-Supervised Fine-tuning for Speaker Diarization

This document describes the self-supervised learning pipeline for improving speaker diarization accuracy through iterative model fine-tuning.

## Overview

The self-supervised learning pipeline refines speaker embeddings by iteratively:
1. Using HMM-corrected clustering results as pseudo-labels
2. Fine-tuning the speaker embedding model
3. Re-extracting embeddings and re-clustering
4. Evaluating and iterating until convergence

## Pipeline Stages

### Epoch 0 (Part 1): Initial Clustering
- Performs multi-modal speaker clustering (audio + visual)
- Applies HMM smoothing for label correction
- Saves initial clustering results
- **This is the existing Stage 5 in run_video.sh**

### Epoch 0 (Part 2): Supervised Fine-tuning
- Uses clustering labels as pseudo-labels for supervised training
- **Two-phase training:**
  1. **Warmup**: Freeze embedding model, train only MLP classifier head
  2. **Fine-tuning**: Unfreeze last few layers, jointly train embedding model and classifier
- Saves fine-tuned model checkpoint

### Epoch 0 (Part 3): Evaluation
- Extracts embeddings using fine-tuned model
- Performs clustering with new embeddings
- Evaluates speaker recognition accuracy
- Saves model checkpoint (epoch0, best)

### Epoch 1+: Iterative Refinement
- **Part 1**: Load previous epoch's clustering results
- **Part 2**: Fine-tune model with pseudo-labels
- **Part 3**: Extract embeddings, cluster, evaluate
- **Repeat** until convergence or max epochs reached

## Usage

### Basic Usage

Run the full pipeline from scratch:

```bash
bash run_video.sh --stage 1 --stop_stage 7 --mode self_supervised
```

### Run Only Self-supervised Learning

If you already have initial clustering results:

```bash
bash run_video.sh --stage 7 --stop_stage 7 --mode self_supervised
```

### Configuration Parameters

Edit the following parameters in `run_video.sh`:

```bash
# Self-supervised learning parameters
max_epochs=10              # Maximum iteration epochs
early_stop_patience=5      # Early stopping patience
use_hmm_smoothing=true    # Use HMM smoothing in iterations
warmup_epochs=2           # Classifier warmup epochs
finetune_lr=0.001         # Fine-tuning learning rate
finetune_batch_size=64    # Fine-tuning batch size
num_finetune_epochs=5     # Epochs per fine-tuning stage
freeze_layers=2           # Number of bottom layers to freeze
```

### Advanced Usage

Fine-tune with custom parameters:

```bash
torchrun --nproc_per_node=1 local/self_supervised_finetune.py \
    --conf conf/diar_video.yaml \
    --wavs /path/to/wav.list \
    --audio_embs_dir /path/to/initial/embeddings \
    --visual_embs_dir /path/to/visual/embeddings \
    --result_dir /path/to/results \
    --speaker_anno_file /path/to/annotation.xlsx \
    --subseg_json /path/to/subseg.json \
    --pretrained_model /path/to/pretrained/model.bin \
    --max_epochs 10 \
    --early_stop_patience 5 \
    --use_hmm_smoothing \
    --warmup_epochs 2 \
    --finetune_lr 0.001 \
    --finetune_batch_size 64 \
    --num_finetune_epochs 5 \
    --freeze_layers 2 \
    --use_gpu --gpu 0 \
    --seed 1234
```

## Output Structure

Results are saved in `$result_dir/self_supervised/`:

```
self_supervised/
├── epoch0_part1/                    # Initial clustering
│   ├── cluster_results_*.json
│   └── *_accuracy.txt
├── epoch0/                          # First fine-tuning iteration
│   ├── finetuned_model.pth         # Fine-tuned model
│   ├── embeddings/                 # New embeddings
│   │   └── *.pkl
│   └── cluster_results/            # New clustering results
│       ├── cluster_results_*.json
│       └── *_accuracy.txt
├── epoch1/                          # Second iteration
│   └── ...
├── best_model.pth                   # Best model across all epochs
├── accuracy_history.json            # Accuracy tracking
└── self_supervised_train.log        # Training log
```

## Key Features

### 1. Pseudo-label Quality
- Uses HMM-corrected clustering results as pseudo-labels
- Filters out noisy samples (cluster label = -1)
- Maps cluster IDs to continuous class indices

### 2. Two-phase Fine-tuning
- **Phase 1 (Warmup)**: Train classifier head only
  - Prevents overfitting to noisy pseudo-labels
  - Stabilizes training
- **Phase 2 (Fine-tuning)**: Update embedding model
  - Only fine-tune last few layers
  - Uses lower learning rate for stability

### 3. Early Stopping
- Monitors accuracy on annotated data
- Stops if no improvement for N epochs (default: 5)
- Saves best model automatically

### 4. Memory Management
- Deletes previous checkpoints after each epoch
- Keeps only: current checkpoint, best checkpoint, epoch0 checkpoint
- Reduces storage requirements

## Expected Results

The pipeline should show:
- **Initial accuracy**: From Epoch 0 Part 1 (baseline)
- **Gradual improvement**: Accuracy increases over epochs
- **Convergence**: Stops when accuracy plateaus

Example:
```
Epoch 0 Part 1: Initial accuracy = 0.75
Epoch 0 Part 3: Accuracy = 0.78 (+0.03)
Epoch 1 Part 3: Accuracy = 0.80 (+0.02)
Epoch 2 Part 3: Accuracy = 0.81 (+0.01)
...
Best accuracy: 0.82 at epoch 3
```

## Troubleshooting

### Issue: No valid training samples
**Cause**: All clustering labels are -1 (noise)
**Solution**: Adjust clustering parameters to reduce noise classification

### Issue: Accuracy decreases
**Cause**: Overfitting to noisy pseudo-labels
**Solution**: 
- Increase `freeze_layers` (freeze more layers)
- Decrease `finetune_lr` (use smaller learning rate)
- Increase `warmup_epochs` (longer warmup)

### Issue: Out of memory
**Cause**: Large batch size or model
**Solution**:
- Decrease `finetune_batch_size`
- Use gradient accumulation (modify code)

### Issue: Pretrained model not found
**Cause**: Model path incorrect
**Solution**: Check and update `pretrained_model_path` in run_video.sh

## Notes

1. **Annotations Required**: Self-supervised learning requires speaker annotations for evaluation and early stopping
2. **Initial Quality Matters**: Better initial clustering → better fine-tuning results
3. **Computational Cost**: Each epoch requires:
   - Model fine-tuning (~5-10 epochs)
   - Embedding extraction
   - Clustering with HMM
4. **Hyperparameter Tuning**: May need adjustment based on your dataset

## References

- Initial clustering: `cluster_and_postprocess.py`
- Accuracy computation: `compute_acc_spk.py`
- Embedding extraction: `extract_diar_embeddings.py`
- Training utilities: `speakerlab/bin/train.py`

## Citation

If you use this self-supervised learning pipeline, please cite:

```
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```
