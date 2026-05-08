"""
Multi-task ordinal classification model for scientific paper review scoring.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, List


def ordinal_soft_labels(
    targets: torch.Tensor,
    num_classes: int,
    smoothing: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Build Laplace-style soft labels that put residual mass on neighbors.

    targets : LongTensor [B] in [0, num_classes-1]
    returns : FloatTensor [B, num_classes]
    """
    idx = torch.arange(num_classes, device=targets.device, dtype=torch.float32)
    dist = torch.abs(idx.unsqueeze(0) - targets.float().unsqueeze(1))  # [B, C]
    neighbor = torch.exp(-dist / max(temperature, 1e-6))
    neighbor = neighbor / neighbor.sum(dim=1, keepdim=True)
    one_hot = F.one_hot(targets, num_classes=num_classes).float()
    return (1.0 - smoothing) * one_hot + smoothing * neighbor


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
        mask = targets >= k                             # [B]
        if mask.sum() == 0:
            continue
        sub_logits  = logits[mask, k]                   # [n_k]
        sub_targets = (targets[mask] > k).float()       # [n_k] (0 or 1)
        # Per-element BCE so we can weight individually.
        bce = F.binary_cross_entropy_with_logits(
            sub_logits, sub_targets, reduction='none'
        )                                               # [n_k]
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
    custom probability threshold. Lowering thresholds[0] biases toward
    predicting class 0 less often (i.e., predicting class 1+ more often);
    raising it biases the other way. Default 0.5 everywhere.
    """
    probs = torch.sigmoid(logits)                       # [B, K-1]
    if thresholds is None:
        return 1 + (probs > 0.5).long().sum(dim=-1)
    th = thresholds.to(probs.device).view(1, -1)        # [1, K-1]
    return 1 + (probs > th).long().sum(dim=-1)


def focal_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    smoothing: float = 0.0,
    smoothing_temperature: float = 1.0,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal cross-entropy with optional ordinal label smoothing, class weights,
    and per-sample weights. Reduces to weighted CE when gamma=0 and smoothing=0.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    if smoothing > 0.0:
        soft_targets = ordinal_soft_labels(targets, num_classes, smoothing, smoothing_temperature)
        ce_per_sample = -(soft_targets * log_probs).sum(dim=-1)
    else:
        ce_per_sample = F.nll_loss(log_probs, targets, reduction='none')

    one_hot = F.one_hot(targets, num_classes=num_classes).float()
    pt = (probs * one_hot).sum(dim=-1).clamp(min=1e-8)
    focal_weight = (1.0 - pt).pow(gamma) if gamma > 0 else torch.ones_like(pt)

    loss = focal_weight * ce_per_sample
    if alpha is not None:
        loss = loss * alpha[targets]
    if sample_weights is not None:
        loss = loss * sample_weights
        denom = sample_weights.sum().clamp(min=1e-8)
        return loss.sum() / denom
    return loss.mean()

# Per-dimension loss weights:
#   RECOMMENDATION is the PRIMARY target  (weight = 1.0, evaluated alone)
#   The other 7 are AUXILIARY helpers     (weight = 0.3 each)
#   Final loss = primary_loss + auxiliary_weight * mean(aux_losses)
_PRIMARY_DIMENSION     = "RECOMMENDATION"
_AUXILIARY_WEIGHT      = 0.3   # How much the 7 aux dimensions contribute vs. RECOMMENDATION


class RegressionHead(nn.Module):
    """Regression head outputs continuous score in [1, num_classes]."""

    def __init__(self, hidden_size: int, dropout: float = 0.1, num_classes: int = 5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)
        self.num_classes = int(num_classes)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_output)
        raw_score = self.regressor(x)
        # Map to [1, num_classes] using sigmoid: 1 + (K-1) * sigmoid(x)
        score = 1.0 + float(self.num_classes - 1) * torch.sigmoid(raw_score)
        return score.squeeze(-1)  # [batch_size]


class ClassificationHead(nn.Module):
    """Classification head for a single score dimension (kept for backward compatibility)."""

    def __init__(self, hidden_size: int, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_output: [batch_size, hidden_size]
        Returns:
            logits: [batch_size, num_classes]
        """
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits


class OrdinalCORNHead(nn.Module):
    """K-1 binary outputs for CORN ordinal regression.

    Each output k corresponds to P(y > k). At inference, the predicted class
    index is 1 + sum_k 1[sigmoid(logit_k) > 0.5].
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


class MultiTaskOrdinalClassifier(nn.Module):
    """
    Multi-task ordinal regression model.

    Uses a shared transformer encoder with separate regression heads
    for each score dimension. Outputs continuous scores in [1, 5] range.
    """

    def __init__(
        self,
        base_model_name: str,
        score_dimensions: List[str],
        num_classes: int = 5,  # Kept for compatibility, not used in regression
        dropout: float = 0.1,
        use_regression: bool = False,
        use_aux_regression: bool = True,
        aux_regression_weight: float = 0.0,
        aux_regression_loss: str = "huber",
        use_hierarchical: bool = False,
        chunk_size: int = 512,
        regression_decider_enabled: bool = True,
        regression_decider_margin: float = 0.15,
        regression_strong_override_distance: float = 0.0,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        use_ordinal_smoothing: bool = False,
        ordinal_smoothing: float = 0.1,
        ordinal_smoothing_temperature: float = 1.0,
        gradient_checkpointing: bool = False,
        chunk_aggregation: str = "attention",
        head_type: str = "softmax",
        corn_thresholds: Optional[List[float]] = None,
    ):
        super().__init__()

        self.score_dimensions = score_dimensions
        self.num_classes = num_classes
        self.use_regression = use_regression
        self.use_aux_regression = use_aux_regression
        self.aux_regression_weight = float(aux_regression_weight)
        self.aux_regression_loss = aux_regression_loss
        self.use_hierarchical = use_hierarchical
        self.regression_decider_enabled = regression_decider_enabled
        self.regression_decider_margin = float(regression_decider_margin)
        self.regression_strong_override_distance = float(regression_strong_override_distance)
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = float(focal_gamma)
        self.use_ordinal_smoothing = use_ordinal_smoothing
        self.ordinal_smoothing = float(ordinal_smoothing)
        self.ordinal_smoothing_temperature = float(ordinal_smoothing_temperature)
        self.head_type = head_type
        self.chunk_aggregation = chunk_aggregation
        # Buffer (not parameter): per-binary-head thresholds for CORN.
        if corn_thresholds is not None:
            assert len(corn_thresholds) == num_classes - 1, (
                f"corn_thresholds must have length num_classes-1 = {num_classes-1}"
            )
            self.register_buffer(
                'corn_thresholds_buf',
                torch.tensor(list(corn_thresholds), dtype=torch.float32),
                persistent=False,
            )
        else:
            self.corn_thresholds_buf = None

        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(base_model_name)
        if self.use_hierarchical:
            self.encoder = HierarchicalEncoder(
                base_model_name,
                chunk_size=chunk_size,
                aggregation=chunk_aggregation,
            )
        else:
            self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)

        # Optional gradient checkpointing on the underlying transformer (saves VRAM)
        if gradient_checkpointing:
            inner = getattr(self.encoder, 'encoder', self.encoder)
            if hasattr(inner, 'gradient_checkpointing_enable'):
                inner.gradient_checkpointing_enable()
                if hasattr(inner, 'config'):
                    inner.config.use_cache = False
                print("[OK] Gradient checkpointing enabled")

        hidden_size = self.config.hidden_size

        # Create regression heads for each dimension
        if use_regression:
            self.heads = nn.ModuleDict({
                dim: RegressionHead(hidden_size, dropout, num_classes=num_classes)
                for dim in score_dimensions
            })
            self.regression_heads = None
        else:
            # Classification heads — softmax (default) or CORN ordinal
            if self.head_type == "corn":
                self.heads = nn.ModuleDict({
                    dim: OrdinalCORNHead(hidden_size, num_classes, dropout)
                    for dim in score_dimensions
                })
            elif self.head_type == "softmax":
                self.heads = nn.ModuleDict({
                    dim: ClassificationHead(hidden_size, num_classes, dropout)
                    for dim in score_dimensions
                })
            else:
                raise ValueError(f"Unknown head_type: {self.head_type!r}")
            # Optional regression heads to guide classification decisions
            if self.use_aux_regression:
                self.regression_heads = nn.ModuleDict({
                    dim: RegressionHead(hidden_size, dropout, num_classes=num_classes)
                    for dim in score_dimensions
                })
            else:
                self.regression_heads = None

    def _expected_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute expected score in [1, num_classes] from class logits."""
        probs = torch.softmax(logits, dim=-1)
        idx = torch.arange(1, self.num_classes + 1, device=logits.device, dtype=probs.dtype)
        return torch.sum(probs * idx, dim=-1)

    def resolve_class_predictions(
        self,
        logits: torch.Tensor,
        regression_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Resolve class predictions, using regression to break ties and to
        override classification on strong disagreements.

        Handles both:
          - Softmax logits  : shape [B, K]      (top-1 = argmax + 1)
          - CORN logits     : shape [B, K-1]    (class = 1 + sum sigmoid > 0.5)
        """
        if logits.size(-1) == self.num_classes - 1:
            # CORN head — top-1 from monotone binary outputs (with optional
            # per-head thresholds for calibration).
            top1 = corn_logits_to_class(logits, thresholds=self.corn_thresholds_buf)
            # The top-2 / softmax tie-break logic doesn't apply to CORN, but
            # the strong-override (regression vs top-1) still does.
            if (not self.regression_decider_enabled) or (regression_scores is None):
                return top1
            reg = regression_scores.clamp(1.0, float(self.num_classes))
            out = top1.clone()
            strong_dist = self.regression_strong_override_distance
            if strong_dist > 0:
                reg_class_full = torch.round(reg).clamp(1, self.num_classes).long()
                strong_mask = torch.abs(reg - top1.float()) >= strong_dist
                out = torch.where(strong_mask, reg_class_full, out)
            return out

        top1 = torch.argmax(logits, dim=-1) + 1
        if (not self.regression_decider_enabled) or (regression_scores is None):
            return top1

        reg = regression_scores.clamp(1.0, float(self.num_classes))

        # Stage 1 — strong-override: if regression disagrees with top-1 by more
        # than `regression_strong_override_distance`, trust the regression's
        # rounded value (clamped to valid class range).
        out = top1.clone()
        strong_dist = self.regression_strong_override_distance
        if strong_dist > 0:
            reg_class_full = torch.round(reg).clamp(1, self.num_classes).long()
            strong_mask = torch.abs(reg - top1.float()) >= strong_dist
            out = torch.where(strong_mask, reg_class_full, out)

        # Stage 2 — close-tie tie-break between top-1 and top-2.
        probs = torch.softmax(logits, dim=-1)
        top2_probs, top2_idx = torch.topk(probs, k=2, dim=-1)
        margin_mask = (top2_probs[:, 0] - top2_probs[:, 1]) <= self.regression_decider_margin
        top2_scores = top2_idx + 1
        dist = torch.abs(top2_scores.float() - reg.unsqueeze(-1))
        reg_choice = top2_scores.gather(1, torch.argmin(dist, dim=-1).unsqueeze(-1)).squeeze(-1)

        # Apply tie-break only where the strong override didn't already fire.
        not_overridden = (out == top1)
        out = torch.where(margin_mask & not_overridden, reg_choice, out)
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: Dict of {dimension: [batch_size]} with continuous values [1, 5] or -1 for missing
            class_weights: Optional dict (not used in regression mode)

        Returns:
            Dictionary containing:
                - predictions: Dict of {dimension: [batch_size]} continuous scores
                - logits: Same as predictions (for backward compatibility)
                - loss: Scalar tensor (if labels provided)
                - per_task_loss: Dict of {dimension: scalar} (if labels provided)
        """
        # Encode text
        if self.use_hierarchical:
            pooled_output = self.encoder(input_ids, attention_mask)
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            # Get pooled representation (CLS token or mean pooling)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            else:
                # Mean pooling over sequence
                hidden_states = outputs.last_hidden_state
                pooled_output = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

        # Compute predictions for each dimension
        predictions = {}
        for dim in self.score_dimensions:
            predictions[dim] = self.heads[dim](pooled_output)

        regression_predictions = None
        if self.regression_heads is not None:
            regression_predictions = {
                dim: self.regression_heads[dim](pooled_output)
                for dim in self.score_dimensions
            }

        output = {
            'predictions': predictions,
            'logits': predictions  # For backward compatibility
        }
        if regression_predictions is not None:
            output['regression_predictions'] = regression_predictions

        # Compute loss if labels provided
        if labels is not None:
            losses = {}
            primary_loss = None
            aux_losses = []

            if self.use_regression:
                loss_fn = nn.HuberLoss(reduction='none', delta=1.0)
            else:
                loss_fn = None

            reg_aux_fn = None
            if (not self.use_regression) and self.use_aux_regression and self.aux_regression_weight > 0:
                if self.aux_regression_loss.lower() == "mse":
                    reg_aux_fn = nn.MSELoss(reduction='none')
                else:
                    reg_aux_fn = nn.HuberLoss(reduction='none', delta=1.0)

            for dim in self.score_dimensions:
                if dim not in labels:
                    continue

                dim_labels = labels[dim].float()
                dim_preds = predictions[dim]

                # Skip samples with missing labels (-1 or NaN)
                valid_mask = (dim_labels >= 0) & (~torch.isnan(dim_labels))

                if valid_mask.sum() == 0:
                    continue

                # Per-sample weights (e.g. reviewer confidence). May be None.
                sw = None
                if sample_weights is not None:
                    sw = sample_weights[valid_mask].to(dim_preds.device)

                if self.use_regression:
                    per = loss_fn(dim_preds[valid_mask], dim_labels[valid_mask])
                    if sw is not None:
                        dim_loss = (per * sw).sum() / sw.sum().clamp(min=1e-8)
                    else:
                        dim_loss = per.mean()
                else:
                    dim_labels_int = torch.round(dim_labels).clamp(1, self.num_classes).long() - 1

                    if self.head_type == "corn":
                        # CORN ordinal loss; class weights are intentionally
                        # ignored — the ordinal structure handles imbalance.
                        ce_loss = corn_loss(
                            dim_preds[valid_mask],
                            dim_labels_int[valid_mask],
                            num_classes=self.num_classes,
                            sample_weights=sw,
                        )
                    else:
                        cls_weight = class_weights.get(dim) if class_weights else None
                        if cls_weight is not None:
                            cls_weight = cls_weight.to(dim_preds.device)

                        gamma = self.focal_gamma if self.use_focal_loss else 0.0
                        smoothing = self.ordinal_smoothing if self.use_ordinal_smoothing else 0.0
                        ce_loss = focal_ce_loss(
                            dim_preds[valid_mask],
                            dim_labels_int[valid_mask],
                            num_classes=self.num_classes,
                            alpha=cls_weight,
                            gamma=gamma,
                            smoothing=smoothing,
                            smoothing_temperature=self.ordinal_smoothing_temperature,
                            sample_weights=sw,
                        )

                    dim_loss = ce_loss

                    # Aux regression — applies for both softmax and CORN heads.
                    if reg_aux_fn is not None and regression_predictions is not None:
                        reg_preds = regression_predictions[dim]
                        reg_targets = dim_labels[valid_mask].clamp(1.0, float(self.num_classes))
                        reg_per = reg_aux_fn(reg_preds[valid_mask], reg_targets)
                        if sw is not None:
                            reg_loss = (reg_per * sw).sum() / sw.sum().clamp(min=1e-8)
                        else:
                            reg_loss = reg_per.mean()
                        dim_loss = ce_loss + (self.aux_regression_weight * reg_loss)

                losses[dim] = dim_loss

                if dim == _PRIMARY_DIMENSION:
                    primary_loss = dim_loss
                else:
                    aux_losses.append(dim_loss)

            # Combine: primary + weighted mean of auxiliaries
            if primary_loss is not None:
                if aux_losses:
                    aux_mean = torch.stack(aux_losses).mean()
                    total_loss = primary_loss + _AUXILIARY_WEIGHT * aux_mean
                else:
                    total_loss = primary_loss
                output['loss'] = total_loss
                output['per_task_loss'] = losses
            elif aux_losses:
                # Fallback: no primary label available in this batch
                total_loss = torch.stack(aux_losses).mean()
                output['loss'] = total_loss
                output['per_task_loss'] = losses

        return output

    def predict_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, float]:
        """
        Predict scores for a single paper.

        Args:
            input_ids: [seq_len] or [1, seq_len]
            attention_mask: [seq_len] or [1, seq_len]

        Returns:
            Dictionary of {dimension: predicted_score} (continuous value in [1, 5])
        """
        self.eval()

        with torch.no_grad():
            # Ensure batch dimension
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            outputs = self.forward(input_ids, attention_mask)
            predictions = outputs['predictions']
            regression_predictions = outputs.get('regression_predictions')

            scores = {}
            for dim, pred in predictions.items():
                if self.use_regression:
                    # Return continuous score
                    scores[dim] = pred.item()
                else:
                    reg_pred = regression_predictions[dim] if regression_predictions is not None else None
                    pred_class = self.resolve_class_predictions(pred, reg_pred).item()
                    scores[dim] = pred_class

        return scores

    def predict_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict class probabilities for a single paper.

        Args:
            input_ids: [seq_len] or [1, seq_len]
            attention_mask: [seq_len] or [1, seq_len]

        Returns:
            Dictionary of {dimension: [num_classes] probability distribution}
        """
        self.eval()

        if self.use_regression:
            raise ValueError("predict_probabilities is only available in classification mode.")

        with torch.no_grad():
            # Ensure batch dimension
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']

            probabilities = {}
            for dim, dim_logits in logits.items():
                # Apply softmax to get probabilities
                probs = torch.softmax(dim_logits, dim=-1).squeeze(0)
                probabilities[dim] = probs

        return probabilities


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder for very long documents.

    Splits document into chunks, encodes each chunk, then aggregates.
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

        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)

        # Lightweight per-chunk attention pooling head (used when aggregation='attention').
        # Runs ALWAYS in fp32 to keep softmax numerics stable under AMP.
        if self.aggregation == 'attention':
            self.chunk_attention = nn.Linear(self.config.hidden_size, 1)
        else:
            self.chunk_attention = None
        # Buffer to expose last attention weights for diagnostics (set during forward).
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode document in chunks.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # If sequence fits in one chunk, process normally
        if seq_len <= self.chunk_size:
            outputs = self.encoder(input_ids, attention_mask, return_dict=True)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                hidden_states = outputs.last_hidden_state
                return (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

        # Split into chunks
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        chunk_embeddings = []

        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, seq_len)

            chunk_input_ids = input_ids[:, start_idx:end_idx]
            chunk_attention_mask = attention_mask[:, start_idx:end_idx]

            # Encode chunk
            outputs = self.encoder(chunk_input_ids, chunk_attention_mask, return_dict=True)

            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                chunk_emb = outputs.pooler_output
            else:
                hidden_states = outputs.last_hidden_state
                chunk_emb = (hidden_states * chunk_attention_mask.unsqueeze(-1)).sum(1) / chunk_attention_mask.sum(1, keepdim=True)

            chunk_embeddings.append(chunk_emb)

        # Aggregate chunk embeddings
        chunk_embeddings = torch.stack(chunk_embeddings, dim=1)  # [B, num_chunks, H]

        if self.aggregation == 'mean':
            return chunk_embeddings.mean(dim=1)
        elif self.aggregation == 'max':
            return chunk_embeddings.max(dim=1)[0]
        elif self.aggregation == 'attention':
            # Score per chunk -> softmax weights -> weighted sum
            scores = self.chunk_attention(chunk_embeddings).squeeze(-1)   # [B, num_chunks]
            weights = torch.softmax(scores, dim=-1)                        # [B, num_chunks]
            self.last_attention_weights = weights.detach()
            return (chunk_embeddings * weights.unsqueeze(-1)).sum(dim=1)   # [B, H]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

