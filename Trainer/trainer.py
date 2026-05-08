"""
Training utilities and Trainer class for the CORN ordinal model.
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Class-distribution diagnostics (used at startup, not during training)
# ---------------------------------------------------------------------------

def count_classes(
    dataset,
    score_dimensions: List[str],
    num_classes: int = 5,
) -> Dict[str, np.ndarray]:
    """Count per-class occurrences across the dataset (skipping NaN/missing)."""
    class_counts = {dim: np.zeros(num_classes, dtype=np.int64) for dim in score_dimensions}
    for i in range(len(dataset)):
        sample = dataset[i]
        for dim in score_dimensions:
            if dim not in sample['labels']:
                continue
            label = float(sample['labels'][dim])
            if np.isnan(label) or label < 1:
                continue
            label_int = int(np.clip(np.round(label), 1, num_classes)) - 1
            class_counts[dim][label_int] += 1
    return class_counts


def print_class_distribution(
    counts: Dict[str, np.ndarray],
    num_classes: int = 5,
    title: str = "Class distribution",
) -> None:
    print(f"\n[{title}]")
    for dim, c in counts.items():
        total = int(c.sum())
        if total == 0:
            print(f"  {dim}: (no labeled samples)")
            continue
        pct = (c / total) * 100.0
        cells = [f"{i+1}: {int(c[i])} ({pct[i]:.1f}%)" for i in range(num_classes)]
        print(f"  {dim} [n={total}]  " + "  |  ".join(cells))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Trainer for the CORN ordinal model."""

    def __init__(
        self,
        model: MultiTaskOrdinalClassifier,
        train_dataloader: DataLoader,
        dev_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        config,
        logger=None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger

        self.use_amp = config.fp16 and torch.cuda.is_available()
        self.scaler = GradScaler(device="cuda") if self.use_amp else None
        if self.use_amp:
            print("[OK] Using Automatic Mixed Precision (AMP)")

        self.metrics_tracker = MetricsTracker(model.score_dimensions)

        # Early stopping
        self.best_score = float('-inf')
        self.patience_counter = 0
        self.best_model_state = None

        # Backbone freezing
        self.freeze_epochs = getattr(config, 'freeze_backbone_epochs', 0)
        if self.freeze_epochs > 0:
            print(f"[OK] Will freeze backbone for first {self.freeze_epochs} epochs")

    # ---- backbone control ------------------------------------------------

    def freeze_backbone(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        print("[FROZEN] Backbone frozen")

    def unfreeze_backbone(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = True
        print("[UNFROZEN] Backbone unfrozen")

    # ---- training --------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        self.model.train()

        if self.freeze_epochs > 0:
            if epoch < self.freeze_epochs:
                self.freeze_backbone()
            elif epoch == self.freeze_epochs:
                self.unfreeze_backbone()

        total_loss = 0.0
        num_batches = 0
        progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=True)

        for batch_idx, batch in enumerate(progress):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = {
                dim: batch['labels'][dim].to(self.device)
                for dim in self.model.score_dimensions
            }

            sample_weights = None
            if getattr(self.config, 'use_confidence_weighting', False) and 'confidence_weight' in batch:
                sample_weights = batch['confidence_weight'].to(self.device)

            if self.use_amp:
                with autocast(device_type="cuda"):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        sample_weights=sample_weights,
                    )
                    loss = outputs['loss']
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    sample_weights=sample_weights,
                )
                loss = outputs['loss']
                loss.backward()
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

            per_task = outputs.get('per_task_loss', {})
            rec_loss = per_task.get('RECOMMENDATION', None)
            postfix = {'loss': f"{loss.item():.4f}"}
            if rec_loss is not None:
                postfix['rec'] = f"{rec_loss.item():.4f}"
            progress.set_postfix(postfix)

            if self.logger and (batch_idx + 1) % self.config.logging_steps == 0:
                step = epoch * len(self.train_dataloader) + batch_idx
                self.logger.add_scalar('train/loss', loss.item(), step)
                if self.scheduler:
                    self.logger.add_scalar('train/lr', self.scheduler.get_last_lr()[0], step)

        return total_loss / max(num_batches, 1)

    # ---- evaluation ------------------------------------------------------

    def evaluate(self, dataloader: DataLoader) -> Dict:
        self.model.eval()
        all_predictions = {dim: [] for dim in self.model.score_dimensions}
        all_labels      = {dim: [] for dim in self.model.score_dimensions}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                if self.use_amp:
                    with autocast(device_type="cuda"):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                preds = outputs['predictions']
                reg_preds = outputs.get('regression_predictions')

                for dim in self.model.score_dimensions:
                    dim_labels = labels[dim].numpy()
                    rp = reg_preds[dim] if reg_preds is not None else None
                    pred_classes = self.model.resolve_class_predictions(preds[dim], rp).cpu().numpy()
                    labels_valid = np.clip(np.round(dim_labels), 1, self.model.num_classes)

                    valid_mask = (~np.isnan(dim_labels)) & (dim_labels >= 1)
                    all_predictions[dim].extend(pred_classes[valid_mask])
                    all_labels[dim].extend(labels_valid[valid_mask])

        all_predictions = {dim: np.array(p) for dim, p in all_predictions.items()}
        all_labels      = {dim: np.array(l) for dim, l in all_labels.items()}

        return compute_multi_task_metrics(
            all_predictions,
            all_labels,
            self.model.score_dimensions,
            is_regression=False,
            num_classes=self.model.num_classes,
        )

    # ---- training loop ---------------------------------------------------

    def train(self, num_epochs: int):
        print("Starting training...")
        print(f"Number of epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_dataloader)}")
        if self.dev_dataloader is not None:
            print(f"Dev batches: {len(self.dev_dataloader)}")
        else:
            print("Dev batches: 0 (dev disabled; using test only)")

        for epoch in range(num_epochs):
            print(f"\n{'='*80}\nEpoch {epoch + 1}/{num_epochs}\n{'='*80}")

            train_loss = self.train_epoch(epoch)
            print(f"\nTrain Loss: {train_loss:.4f}")

            if self.dev_dataloader is not None:
                print("\nEvaluating on dev set...")
                dev_metrics = self.evaluate(self.dev_dataloader)
                print(self.metrics_tracker.format_metrics(dev_metrics, "Dev Set Metrics"))
                self.metrics_tracker.update(epoch, train_loss, dev_metrics)

                if self.logger:
                    self.logger.add_scalar('train/epoch_loss', train_loss, epoch)
                    for k in ('spearman', 'qwk', 'mae', 'accuracy', 'macro_f1', 'rmse'):
                        v = dev_metrics.get(f'recommendation_{k}')
                        if v is not None:
                            self.logger.add_scalar(f'dev/recommendation_{k}', v, epoch)
                    self.logger.add_scalar('dev/avg_qwk', dev_metrics.get('avg_qwk', 0.0), epoch)
                    for name, value in dev_metrics.get('macro_avg', {}).items():
                        self.logger.add_scalar(f'dev/macro_{name}', value, epoch)

                metric_key = getattr(self.config, 'early_stopping_metric', 'recommendation_qwk')
                current_score = dev_metrics.get(metric_key, 0.0)

                if current_score > self.best_score:
                    self.best_score = current_score
                    self.patience_counter = 0
                    self.best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_metric': metric_key,
                        'best_score': current_score,
                        'metrics': dev_metrics,
                    }
                    print(f"\n[BEST] New best model!  {metric_key}={current_score:.4f}")
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    print(f"\nNo improvement in {metric_key}. "
                          f"Patience: {self.patience_counter}/{self.config.early_stopping_patience}  "
                          f"({metric_key}={current_score:.4f}, best={self.best_score:.4f})")
                    if self.patience_counter >= self.config.early_stopping_patience:
                        print(f"\n[STOP] Early stopping triggered after {epoch + 1} epochs")
                        break
            else:
                if self.logger:
                    self.logger.add_scalar('train/epoch_loss', train_loss, epoch)

            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(epoch, is_best=False)

        print("\n" + "=" * 80)
        print("Training completed!")
        if self.dev_dataloader is not None:
            best_epoch, best_qwk = self.metrics_tracker.get_best_metrics()
            metric_key = getattr(self.config, 'early_stopping_metric', 'recommendation_qwk')
            print(f"Best model: Epoch {best_epoch + 1}  "
                  f"({metric_key}: {self.best_score:.4f}  |  Avg QWK: {best_qwk:.4f})")
        else:
            print("Best model: dev disabled; keeping last epoch weights")
        print("=" * 80)

        # Load best weights
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            print("\n[OK] Best model loaded")
        elif self.dev_dataloader is None:
            self.best_model_state = {
                'epoch': num_epochs - 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_metric': None,
                'best_score': None,
                'metrics': None,
            }

        return self.best_model_state

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        os.makedirs(self.config.output_dir, exist_ok=True)
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        if is_best:
            path = os.path.join(self.config.output_dir, 'best_model.pt')
            torch.save(ckpt, path)
            print(f"[OK] Best model saved to {path}")
        else:
            path = os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(ckpt, path)
            print(f"[OK] Checkpoint saved to {path}")


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    config,
):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    groups = [
        {'params': backbone_params, 'lr': config.backbone_lr, 'weight_decay': config.weight_decay},
        {'params': head_params,     'lr': config.head_lr,     'weight_decay': config.weight_decay},
    ]
    optimizer = AdamW(groups, eps=config.adam_epsilon)

    num_training_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    if getattr(config, 'warmup_steps', None) is not None:
        num_warmup_steps = config.warmup_steps
    else:
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print(f"[OK] AdamW with differential LRs: backbone={config.backbone_lr:.2e}, head={config.head_lr:.2e}")
    print(f"[OK] Linear warmup ({num_warmup_steps} steps / {config.warmup_ratio*100:.0f}%) + decay ({num_training_steps} total)")
    return optimizer, scheduler
