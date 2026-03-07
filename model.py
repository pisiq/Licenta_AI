"""
Multi-task ordinal classification model for scientific paper review scoring.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, List

# Per-dimension loss weights: RECOMMENDATION is primary (3x), others auxiliary
# Imported lazily to avoid circular import; fallback defaults provided here.
_DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "RECOMMENDATION":        3.0,
    "IMPACT":                0.5,
    "SUBSTANCE":             0.5,
    "APPROPRIATENESS":       0.5,
    "MEANINGFUL_COMPARISON": 0.5,
    "SOUNDNESS_CORRECTNESS": 0.5,
    "ORIGINALITY":           0.5,
    "CLARITY":               0.5,
}


class RegressionHead(nn.Module):
    """Regression head for a single score dimension - outputs continuous score in [1, 5]."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)
        # Sigmoid + scaling to map output to [1, 5] range

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_output: [batch_size, hidden_size]
        Returns:
            scores: [batch_size, 1] - continuous scores in range [1, 5]
        """
        x = self.dropout(pooled_output)
        # Raw output
        raw_score = self.regressor(x)
        # Map to [1, 5] using sigmoid: 1 + 4 * sigmoid(x)
        score = 1.0 + 4.0 * torch.sigmoid(raw_score)
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
        use_regression: bool = True
    ):
        super().__init__()

        self.score_dimensions = score_dimensions
        self.num_classes = num_classes
        self.use_regression = use_regression

        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)

        hidden_size = self.config.hidden_size

        # Create regression heads for each dimension
        if use_regression:
            self.heads = nn.ModuleDict({
                dim: RegressionHead(hidden_size, dropout)
                for dim in score_dimensions
            })
        else:
            # Keep classification for backward compatibility
            self.heads = nn.ModuleDict({
                dim: ClassificationHead(hidden_size, num_classes, dropout)
                for dim in score_dimensions
            })

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        class_weights: Optional[Dict[str, torch.Tensor]] = None
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

        output = {
            'predictions': predictions,
            'logits': predictions  # For backward compatibility
        }

        # Compute loss if labels provided
        if labels is not None:
            losses = {}
            weighted_loss_sum = torch.tensor(0.0, device=pooled_output.device)
            weight_total      = 0.0

            for dim in self.score_dimensions:
                if dim not in labels:
                    continue

                dim_labels = labels[dim].float()
                dim_preds  = predictions[dim]

                # Per-dimension task weight (RECOMMENDATION=3.0, others=0.5)
                dim_weight = _DEFAULT_SCORE_WEIGHTS.get(dim, 0.5)

                # Skip samples with missing labels (-1 or NaN)
                valid_mask = (dim_labels >= 0) & (~torch.isnan(dim_labels))

                if valid_mask.sum() > 0:
                    if self.use_regression:
                        # Huber Loss — robust to score outliers
                        loss_fn  = nn.HuberLoss(reduction='none', delta=1.0)
                        dim_loss = loss_fn(dim_preds[valid_mask], dim_labels[valid_mask])
                        dim_loss = dim_loss.mean()
                    else:
                        # Classification mode (backward compatibility)
                        dim_labels_int = dim_labels.long()
                        cls_weight = class_weights.get(dim) if class_weights else None
                        if cls_weight is not None:
                            cls_weight = cls_weight.to(dim_preds.device)
                        loss_fn  = nn.CrossEntropyLoss(weight=cls_weight, reduction='none')
                        dim_loss = loss_fn(dim_preds[valid_mask], dim_labels_int[valid_mask])
                        dim_loss = dim_loss.mean()

                    losses[dim]        = dim_loss
                    weighted_loss_sum  = weighted_loss_sum + dim_weight * dim_loss
                    weight_total      += dim_weight

            if weight_total > 0:
                total_loss = weighted_loss_sum / weight_total
                output['loss']          = total_loss
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

            scores = {}
            for dim, pred in predictions.items():
                if self.use_regression:
                    # Return continuous score
                    scores[dim] = pred.item()
                else:
                    # Classification mode - return class + 1
                    pred_class = torch.argmax(pred, dim=-1).item()
                    scores[dim] = pred_class + 1

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
        aggregation: str = 'mean'
    ):
        super().__init__()

        self.chunk_size = chunk_size
        self.aggregation = aggregation

        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)

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
        chunk_embeddings = torch.stack(chunk_embeddings, dim=1)  # [batch_size, num_chunks, hidden_size]

        if self.aggregation == 'mean':
            return chunk_embeddings.mean(dim=1)
        elif self.aggregation == 'max':
            return chunk_embeddings.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

