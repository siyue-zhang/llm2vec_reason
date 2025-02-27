import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather

class TripletLoss:
    def __init__(
        self,
        margin: float = 1.0,
    ):
        """
        Args:
            margin: margin for the triplet loss. The loss will encourage the anchor and positive
                    examples to be closer than the anchor and negative examples by at least this margin.
            similarity_fct: Function to compute the similarity between two tensors. Cosine similarity by default.
        """
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def __call__(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor = None,
    ):
        """
        Args:
            q_reps: Query representations (anchor)
            d_reps_pos: Positive document representations
            d_reps_neg: Negative document representations

        Returns:
            loss: The triplet loss value
        """
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        # Triplet Loss expects three inputs: anchor, positive, and negative examples
        anchor = full_q_reps
        positive = full_d_reps_pos
        negative = full_d_reps_neg
        
        # Compute the Triplet Loss
        loss = self.triplet_loss(anchor, positive, negative)
        return loss
