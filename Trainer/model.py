"""
Ordinal CORN model for scientific-paper review scoring.

The model is CORN-only:
- Primary head : OrdinalCORNHead with K-1 binary outputs.
- Aux head    : RegressionHead — continuous score in [1, num_classes],
                 trained with Huber loss, used as a smooth regularizer
                 *and* as a tie-breaker / strong-override at inference.
- Encoder     : AutoModel (e.g. SciBERT) optionally wrapped in a
                 HierarchicalEncoder (chunked input + attention pooling).

Removed in the simplification pass:
- Pure-regression-as-primary path (`use_regression=True`)
- Softmax classification head (`head_type="softmax"`, ClassificationHead)
- Focal CE loss + ordinal label smoothing — CORN's structural ordering
  combined with the aux Huber head already provides "close-is-better".
- Class-weight wiring in the loss — CORN handles imbalance natively.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, List

# Per-dimension loss weights:
#   RECOMMENDATION is the PRIMARY target (weight = 1.0)
#   Auxiliary dimensions (none today) contribute at _AUXILIARY_WEIGHT each.
_PRIMARY_DIMENSION = "RECOMMENDATION"
_AUXILIARY_WEIGHT  = 0.3


# ---------------------------------------------------------------------------
# CORN loss + decoder
# ---------------------------------------------------------------------------

def corn_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CORN ordinal loss with optional per-sample weights.

    `logits`         : [B, K-1] — head outputs P(y > k) for k = 0..K-2
    `targets`        : [B]      — long, 0..K-1
    `sample_weights` : [B]      — optional, multiplies each sample's BCE term.

    Each binary head k trains on the subset of samples whose target >= k.
    """
    K = num_classes
    losses = []
    weight_sum = 0.0
    for k in range(K - 1):
        mask = targets >= k
        if mask.sum() == 0:
            continue
        sub_logits  = logits[mask, k]
        sub_targets = (targets[mask] > k).float()
        bce = F.binary_cross_entropy_with_logits(
            sub_logits, sub_targets, reduction='none'
        )
        if sample_weights is not None:
            w = sample_weights[mask]
            bce = bce * w
            weight_sum += float(w.sum().item())
        else:
            weight_sum += float(mask.sum().item())
        losses.append(bce.sum())

    if not losses or weight_sum <= 0:
        return logits.new_zeros(())
    return torch.stack(losses).sum() / weight_sum


def corn_logits_to_class(
    logits: torch.Tensor,
    thresholds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode CORN [B, K-1] logits to predicted class index in [1, K].

    `thresholds` (optional, shape [K-1]) lets each binary head fire at a
    custom probability threshold. Default 0.5 everywhere.
    """
    probs = torch.sigmoid(logits)
    if thresholds is None:
        return 1 + (probs > 0.5).long().sum(dim=-1)
    th = thresholds.to(probs.device).view(1, -1)
    return 1 + (probs > th).long().sum(dim=-1)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class RegressionHead(nn.Module):
    """Aux regression head — outputs continuous score in [1, num_classes]."""

    def __init__(self, hidden_size: int, dropout: float = 0.1, num_classes: int = 5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)
        self.num_classes = int(num_classes)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_output)
        raw = self.regressor(x)
        score = 1.0 + float(self.num_classes - 1) * torch.sigmoid(raw)
        return score.squeeze(-1)


class OrdinalCORNHead(nn.Module):
    """K-1 binary outputs for CORN ordinal regression.

    Each output k corresponds to P(y > k). At inference, the predicted class
    index is 1 + sum_k 1[sigmoid(logit_k) > threshold_k].
    """

    def __init__(self, hidden_size: int, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        assert num_classes >= 2, "CORN head needs at least 2 classes"
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes - 1)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_output)
        return self.classifier(x)  # [B, K-1]


# ---------------------------------------------------------------------------
# Hierarchical encoder
# ---------------------------------------------------------------------------

class HierarchicalEncoder(nn.Module):
    """Chunk a long document, encode each chunk, then pool across chunks.

    Pooling modes: "mean", "max", or "attention" (default).
    """

    def __init__(
        self,
        base_model_name: str,
        chunk_size: int = 512,
        aggregation: str = 'attention'
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.aggregation = aggregation

        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)

        if self.aggregation == 'attention':
            self.chunk_attention = nn.Linear(self.config.hidden_size, 1)
        else:
            self.chunk_attention = None
        # Diagnostic buffer; populated during forward when aggregation='attention'.
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Single-chunk fast path
        if seq_len <= self.chunk_size:
            outputs = self.encoder(input_ids, attention_mask, return_dict=True)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            hidden = outputs.last_hidden_state
            return (hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

        # Multi-chunk path
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        chunk_embeddings = []
        for i in range(num_chunks):
            s = i * self.chunk_size
            e = min((i + 1) * self.chunk_size, seq_len)
            ck_ids = input_ids[:, s:e]
            ck_mask = attention_mask[:, s:e]
            outputs = self.encoder(ck_ids, ck_mask, return_dict=True)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                hidden = outputs.last_hidden_state
                emb = (hidden * ck_mask.unsqueeze(-1)).sum(1) / ck_mask.sum(1, keepdim=True)
            chunk_embeddings.append(emb)

        chunk_embeddings = torch.stack(chunk_embeddings, dim=1)  # [B, num_chunks, H]

        if self.aggregation == 'mean':
            return chunk_embeddings.mean(dim=1)
        if self.aggregation == 'max':
            return chunk_embeddings.max(dim=1)[0]
        if self.aggregation == 'attention':
            scores = self.chunk_attention(chunk_embeddings).squeeze(-1)  # [B, nc]
            weights = torch.softmax(scores, dim=-1)
            self.last_attention_weights = weights.detach()
            return (chunk_embeddings * weights.unsqueeze(-1)).sum(dim=1)
        raise ValueError(f"Unknown aggregation: {self.aggregation}")


# ---------------------------------------------------------------------------
# Multi-task CORN classifier
# ---------------------------------------------------------------------------

class MultiTaskOrdinalClassifier(nn.Module):
    """CORN-only multi-dim ordinal classifier with optional aux regression."""

    def __init__(
        self,
        base_model_name: str,
        score_dimensions: List[str],
        num_classes: int = 5,
        dropout: float = 0.1,
        # Aux regression — strongly recommended ON; provides Huber smooth
        # signal + acts as tie-breaker / strong-override at inference.
        use_aux_regression: bool = True,
        aux_regression_weight: float = 0.5,
        aux_regression_loss: str = "huber",     # "huber" | "mse"
        # Hierarchical document encoding
        use_hierarchical: bool = True,
        chunk_size: int = 512,
        chunk_aggregation: str = "attention",   # "attention" | "mean" | "max"
        # Regression-vs-CORN inference policy
        regression_decider_enabled: bool = True,
        regression_strong_override_distance: float = 1.5,
        # CORN per-head thresholds (length K-1) — None means 0.5 everywhere
        corn_thresholds: Optional[List[float]] = None,
        # Memory
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.score_dimensions = score_dimensions
        self.num_classes = num_classes
        self.use_aux_regression = use_aux_regression
        self.aux_regression_weight = float(aux_regression_weight)
        self.aux_regression_loss = aux_regression_loss
        self.use_hierarchical = use_hierarchical
        self.chunk_aggregation = chunk_aggregation
        self.regression_decider_enabled = regression_decider_enabled
        self.regression_strong_override_distance = float(regression_strong_override_distance)

        # CORN thresholds buffer — None or length-K-1
        if corn_thresholds is not None:
            assert len(corn_thresholds) == num_classes - 1, (
                f"corn_thresholds must have length {num_classes - 1}, got {len(corn_thresholds)}"
            )
            self.register_buffer(
                'corn_thresholds_buf',
                torch.tensor(list(corn_thresholds), dtype=torch.float32),
                persistent=False,
            )
        else:
            self.corn_thresholds_buf = None

        # Encoder
        self.config = AutoConfig.from_pretrained(base_model_name)
        if self.use_hierarchical:
            self.encoder = HierarchicalEncoder(
                base_model_name,
                chunk_size=chunk_size,
                aggregation=chunk_aggregation,
            )
        else:
            self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)

        if gradient_checkpointing:
            inner = getattr(self.encoder, 'encoder', self.encoder)
            if hasattr(inner, 'gradient_checkpointing_enable'):
                inner.gradient_checkpointing_enable()
                if hasattr(inner, 'config'):
                    inner.config.use_cache = False
                print("[OK] Gradient checkpointing enabled")

        hidden_size = self.config.hidden_size

        # Heads — CORN primary + optional regression aux (per dimension)
        self.heads = nn.ModuleDict({
            dim: OrdinalCORNHead(hidden_size, num_classes, dropout)
            for dim in score_dimensions
        })
        if self.use_aux_regression:
            self.regression_heads = nn.ModuleDict({
                dim: RegressionHead(hidden_size, dropout, num_classes=num_classes)
                for dim in score_dimensions
            })
        else:
            self.regression_heads = None

    # ---- inference helpers ------------------------------------------------

    def resolve_class_predictions(
        self,
        corn_logits: torch.Tensor,
        regression_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CORN class index in [1, K], optionally overridden by regression
        when |regression - top1| >= regression_strong_override_distance.
        """
        top1 = corn_logits_to_class(corn_logits, thresholds=self.corn_thresholds_buf)

        if (not self.regression_decider_enabled) or (regression_scores is None):
            return top1

        reg = regression_scores.clamp(1.0, float(self.num_classes))
        out = top1.clone()
        d = self.regression_strong_override_distance
        if d > 0:
            reg_class = torch.round(reg).clamp(1, self.num_classes).long()
            strong_mask = torch.abs(reg - top1.float()) >= d
            out = torch.where(strong_mask, reg_class, out)
        return out

    # ---- forward + loss ---------------------------------------------------

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.use_hierarchical:
            return self.encoder(input_ids, attention_mask)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        h = outputs.last_hidden_state
        return (h * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        pooled = self._encode(input_ids, attention_mask)

        predictions = {dim: self.heads[dim](pooled) for dim in self.score_dimensions}
        regression_predictions = None
        if self.regression_heads is not None:
            regression_predictions = {
                dim: self.regression_heads[dim](pooled) for dim in self.score_dimensions
            }

        output = {
            'predictions': predictions,
            'logits': predictions,  # back-compat
        }
        if regression_predictions is not None:
            output['regression_predictions'] = regression_predictions

        if labels is None:
            return output

        # ---- loss --------------------------------------------------------
        if self.aux_regression_loss.lower() == 'mse':
            reg_fn = nn.MSELoss(reduction='none')
        else:
            reg_fn = nn.HuberLoss(reduction='none', delta=1.0)

        losses: Dict[str, torch.Tensor] = {}
        primary_loss: Optional[torch.Tensor] = None
        aux_losses: List[torch.Tensor] = []

        for dim in self.score_dimensions:
            if dim not in labels:
                continue
            dim_labels = labels[dim].float()
            dim_logits = predictions[dim]

            valid = (dim_labels >= 0) & (~torch.isnan(dim_labels))
            if valid.sum() == 0:
                continue

            sw = sample_weights[valid].to(dim_logits.device) if sample_weights is not None else None
            dim_labels_int = torch.round(dim_labels).clamp(1, self.num_classes).long() - 1

            ce = corn_loss(
                dim_logits[valid],
                dim_labels_int[valid],
                num_classes=self.num_classes,
                sample_weights=sw,
            )
            dim_loss = ce

            # Aux regression
            if regression_predictions is not None and self.aux_regression_weight > 0:
                rp = regression_predictions[dim]
                rt = dim_labels[valid].clamp(1.0, float(self.num_classes))
                per = reg_fn(rp[valid], rt)
                if sw is not None:
                    reg_loss = (per * sw).sum() / sw.sum().clamp(min=1e-8)
                else:
                    reg_loss = per.mean()
                dim_loss = ce + self.aux_regression_weight * reg_loss

            losses[dim] = dim_loss
            if dim == _PRIMARY_DIMENSION:
                primary_loss = dim_loss
            else:
                aux_losses.append(dim_loss)

        if primary_loss is not None:
            total = primary_loss
            if aux_losses:
                total = primary_loss + _AUXILIARY_WEIGHT * torch.stack(aux_losses).mean()
            output['loss'] = total
            output['per_task_loss'] = losses
        elif aux_losses:
            output['loss'] = torch.stack(aux_losses).mean()
            output['per_task_loss'] = losses

        return output

    # ---- single-paper inference -----------------------------------------

    def predict_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            out = self.forward(input_ids, attention_mask)
            preds = out['predictions']
            reg_preds = out.get('regression_predictions')
            scores = {}
            for dim, p in preds.items():
                rp = reg_preds[dim] if reg_preds is not None else None
                scores[dim] = float(self.resolve_class_predictions(p, rp).item())
        return scores
