"""
Training utilities and trainer class.
"""
import os
import torch
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from metrics import compute_multi_task_metrics, MetricsTracker
from model import MultiTaskOrdinalClassifier


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_class_weights(
    dataset,
    score_dimensions: List[str],
    num_classes: int = 5
) -> Dict[str, torch.Tensor]:
    """
    Compute class weights for handling class imbalance.

    Args:
        dataset: PyTorch Dataset
        score_dimensions: List of score dimension names
        num_classes: Number of classes

    Returns:
        Dictionary of {dimension: weight tensor [num_classes]}
    """
    # Count class occurrences
    class_counts = {dim: np.zeros(num_classes) for dim in score_dimensions}

    for i in range(len(dataset)):
        sample = dataset[i]
        for dim in score_dimensions:
            if dim in sample['labels']:
                label = sample['labels'][dim]
                if label >= 0:  # Valid label
                    class_counts[dim][label] += 1

    # Compute weights (inverse frequency)
    class_weights = {}
    for dim in score_dimensions:
        counts = class_counts[dim]
        # Avoid division by zero
        counts = np.maximum(counts, 1)

        # Inverse frequency
        weights = 1.0 / counts
        # Normalize
        weights = weights / weights.sum() * num_classes

        class_weights[dim] = torch.FloatTensor(weights)

    return class_weights


class Trainer:
    """Trainer for multi-task ordinal classification/regression."""

    def __init__(
        self,
        model: MultiTaskOrdinalClassifier,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        config,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        logger=None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.class_weights = class_weights
        self.logger = logger

        # AMP (Automatic Mixed Precision) for memory optimization
        self.use_amp = config.fp16 and torch.cuda.is_available()
        self.scaler = GradScaler(device="cuda") if self.use_amp else None

        if self.use_amp:
            print("[OK] Using Automatic Mixed Precision (AMP) for faster training")

        # Move class weights to device
        if self.class_weights:
            self.class_weights = {
                dim: w.to(device) for dim, w in self.class_weights.items()
            }

        # Metrics tracker
        self.metrics_tracker = MetricsTracker(model.score_dimensions)

        # Early stopping — use composite score so we don't stall at QWK=0
        self.best_score = float('-inf')
        self.patience_counter = 0
        self.best_model_state = None

        # Backbone freezing
        self.freeze_epochs = getattr(config, 'freeze_backbone_epochs', 0)
        if self.freeze_epochs > 0:
            print(f"[OK] Will freeze backbone for first {self.freeze_epochs} epochs")

    def freeze_backbone(self):
        """Freeze the encoder backbone."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        print("[FROZEN] Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze the encoder backbone."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        print("[UNFROZEN] Backbone unfrozen")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()

        # Handle backbone freezing
        if self.freeze_epochs > 0:
            if epoch < self.freeze_epochs:
                self.freeze_backbone()
            elif epoch == self.freeze_epochs:
                self.unfreeze_backbone()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=True)

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = {
                dim: batch['labels'][dim].to(self.device)
                for dim in self.model.score_dimensions
            }
            # label_mask is optional (new dataset provides it, old collate may not)
            label_mask = {
                dim: batch['label_mask'][dim].to(self.device)
                for dim in self.model.score_dimensions
            } if 'label_mask' in batch else None

            # Forward pass with AMP
            if self.use_amp:
                with autocast(device_type="cuda"):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        class_weights=self.class_weights
                    )
                    loss = outputs['loss']

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping (unscale first)
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                # Regular training without AMP
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    class_weights=self.class_weights
                )
                loss = outputs['loss']

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # Optimizer step
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar with total + RECOMMENDATION loss
            per_task = outputs.get('per_task_loss', {})
            rec_loss = per_task.get('RECOMMENDATION', None)
            postfix = {'loss': f"{loss.item():.4f}"}
            if rec_loss is not None:
                postfix['rec'] = f"{rec_loss.item():.4f}"
            progress_bar.set_postfix(postfix)

            # Logging
            if self.logger and (batch_idx + 1) % self.config.logging_steps == 0:
                step = epoch * len(self.train_dataloader) + batch_idx
                self.logger.add_scalar('train/loss', loss.item(), step)
                if self.scheduler:
                    self.logger.add_scalar('train/lr', self.scheduler.get_last_lr()[0], step)

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate on a dataset."""
        self.model.eval()

        all_predictions = {dim: [] for dim in self.model.score_dimensions}
        all_labels = {dim: [] for dim in self.model.score_dimensions}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                # Forward pass (with AMP if enabled)
                if self.use_amp:
                    with autocast(device_type="cuda"):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                predictions = outputs['predictions']

                # Collect predictions and labels  (skip NaN / missing labels)
                for dim in self.model.score_dimensions:
                    dim_labels = labels[dim].numpy()  # may contain NaN

                    if self.model.use_regression:
                        preds = predictions[dim].cpu().numpy()
                    else:
                        preds = torch.argmax(predictions[dim], dim=-1).cpu().numpy()

                    # Only keep samples with a valid (non-NaN, >= 1) label
                    valid_mask = (~np.isnan(dim_labels)) & (dim_labels >= 1)
                    all_predictions[dim].extend(preds[valid_mask])
                    all_labels[dim].extend(dim_labels[valid_mask])

        # Convert to numpy arrays
        all_predictions = {dim: np.array(preds) for dim, preds in all_predictions.items()}
        all_labels = {dim: np.array(labs) for dim, labs in all_labels.items()}

        # Compute metrics
        metrics = compute_multi_task_metrics(
            all_predictions,
            all_labels,
            self.model.score_dimensions,
            is_regression=self.model.use_regression
        )

        return metrics

    def train(self, num_epochs: int):
        """Full training loop."""
        print("Starting training...")
        print(f"Number of epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_dataloader)}")
        print(f"Dev batches: {len(self.dev_dataloader)}")

        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}")

            # Train
            train_loss = self.train_epoch(epoch)
            print(f"\nTrain Loss: {train_loss:.4f}")

            # Evaluate on dev set
            print("\nEvaluating on dev set...")
            dev_metrics = self.evaluate(self.dev_dataloader)

            # Log metrics
            print(self.metrics_tracker.format_metrics(dev_metrics, "Dev Set Metrics"))

            # Update tracker
            self.metrics_tracker.update(epoch, train_loss, dev_metrics)

            # TensorBoard logging
            if self.logger:
                self.logger.add_scalar('train/epoch_loss', train_loss, epoch)
                # Recommendation-specific (primary target)
                self.logger.add_scalar('dev/recommendation_spearman', dev_metrics.get('recommendation_spearman', 0.0), epoch)
                self.logger.add_scalar('dev/recommendation_qwk',      dev_metrics.get('recommendation_qwk', 0.0),      epoch)
                self.logger.add_scalar('dev/recommendation_mae',       dev_metrics.get('recommendation_mae', 5.0),       epoch)
                # All-dimension averages (secondary, for context)
                self.logger.add_scalar('dev/avg_qwk', dev_metrics['avg_qwk'], epoch)
                for metric_name, value in dev_metrics['macro_avg'].items():
                    self.logger.add_scalar(f'dev/macro_{metric_name}', value, epoch)

            # ------------------------------------------------------------------
            # Early stopping — track RECOMMENDATION Spearman ONLY.
            # The model is judged solely on how well it ranks papers by their
            # recommendation score; the 7 auxiliary dimensions help train but
            # do not influence the stopping criterion.
            # ------------------------------------------------------------------
            rec_spearman = dev_metrics.get('recommendation_spearman', 0.0)
            rec_qwk      = dev_metrics.get('recommendation_qwk', 0.0)
            rec_mae      = dev_metrics.get('recommendation_mae', 5.0)

            # Use Spearman as the primary signal (works even before QWK improves)
            current_score = rec_spearman

            if current_score > self.best_score:
                self.best_score = current_score
                self.patience_counter = 0
                # Save best model
                self.best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'recommendation_spearman': rec_spearman,
                    'recommendation_qwk': rec_qwk,
                    'recommendation_mae': rec_mae,
                    'metrics': dev_metrics
                }
                print(f"\n[BEST] New best model!  "
                      f"RECOMMENDATION → Spearman={rec_spearman:.4f}  "
                      f"QWK={rec_qwk:.4f}  MAE={rec_mae:.4f}")

                # Save checkpoint
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                print(f"\nNo improvement in RECOMMENDATION Spearman. "
                      f"Patience: {self.patience_counter}/{self.config.early_stopping_patience}"
                      f"  (Spearman={rec_spearman:.4f}, best={self.best_score:.4f})")

                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\n[STOP] Early stopping triggered after {epoch + 1} epochs")
                    break

            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(epoch, is_best=False)

        print("\n" + "="*80)
        print("Training completed!")
        best_epoch, best_qwk = self.metrics_tracker.get_best_metrics()
        print(f"Best model: Epoch {best_epoch + 1}  "
              f"(RECOMMENDATION Spearman: {self.best_score:.4f}  |  Avg QWK: {best_qwk:.4f})")
        print("="*80)

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            print("\n[OK] Best model loaded")

        return self.best_model_state

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }

        if is_best:
            path = os.path.join(self.config.output_dir, 'best_model.pt')
            torch.save(checkpoint, path)
            print(f"[OK] Best model saved to {path}")
        else:
            path = os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, path)
            print(f"[OK] Checkpoint saved to {path}")


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    config
):
    """Create optimizer with differential learning rates and scheduler."""

    # Separate parameters for backbone and heads
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'encoder' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Create parameter groups with differential learning rates
    # During freezing, only heads will train. After unfreezing, use different LRs.
    optimizer_grouped_parameters = [
        {
            'params': backbone_params,
            'lr': config.backbone_lr,  # Lower LR for pretrained encoder
            'weight_decay': config.weight_decay
        },
        {
            'params': head_params,
            'lr': config.head_lr,  # Higher LR for regression heads
            'weight_decay': config.weight_decay
        }
    ]

    # Optimizer with parameter groups
    optimizer = AdamW(
        optimizer_grouped_parameters,
        eps=config.adam_epsilon
    )

    # Learning rate scheduler
    num_training_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps

    # Use warmup_steps if provided, otherwise use warmup_ratio
    if hasattr(config, 'warmup_steps') and config.warmup_steps is not None:
        num_warmup_steps = config.warmup_steps
    else:
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print(f"[OK] Optimizer: AdamW with differential LRs")
    print(f"  - Backbone LR: {config.backbone_lr:.2e}")
    print(f"  - Head LR: {config.head_lr:.2e}")
    print(f"  - Weight decay: {config.weight_decay}")
    print(f"[OK] Scheduler: Linear warmup ({num_warmup_steps} steps / {config.warmup_ratio*100:.0f}%) + decay ({num_training_steps} total steps)")

    return optimizer, scheduler

