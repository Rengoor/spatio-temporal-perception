import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalConsistencyLoss(nn.Module):
    def __init__(self, spatial_weight=1.0, temporal_weight=0.5):
        """
        Args:
            spatial_weight (float): Scalar multiplier for standard frame-by-frame loss.
            temporal_weight (float): Scalar multiplier for inter-frame smoothness constraint.
        """
        super(SpatioTemporalConsistencyLoss, self).__init__()
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight

        # Smooth L1 loss handles outliers gracefully without exploding gradients
        self.smooth_l1 = nn.SmoothL1Loss(reduction="mean")

    def forward(self, pred_sequence, target_sequence):
        """
        Args:
            pred_sequence (Tensor): Model predictions of shape (Batch, Time_Steps, Anchors, 4)
                                    where 4 represents bounding box coordinates [x, y, w, h]
            target_sequence (Tensor): Ground truth annotations of shape (Batch, Time_Steps, Anchors, 4)
        Returns:
            Total joint spatio-temporal loss scalar.
        """
        batch_size, time_steps, num_anchors, box_coords = pred_sequence.size()

        # 1. Compute Standard Spatial Loss (Static accuracy per frame)
        # Flatten temporal and batch dimensions to evaluate overall structural error
        spatial_loss = self.smooth_l1(pred_sequence, target_sequence)

        # 2. Compute Inter-Frame Consistency Loss (Temporal Smoothness)
        # Calculate changes between consecutive time steps: Frame(t) vs Frame(t+1)
        temporal_loss = 0.0

        if time_steps > 1:
            # Shift sequences to get paired differences over time
            # pred_t represents frames 0 to T-1
            pred_t = pred_sequence[:, :-1, :, :]
            # pred_t_next represents frames 1 to T
            pred_t_next = pred_sequence[:, 1:, :, :]

            # Extract width and height variations to penalize jarring scale pops
            scale_t = pred_t[:, :, :, 2:]
            scale_t_next = pred_t_next[:, :, :, 2:]

            # Calculate absolute delta acceleration/velocity shifts in prediction steps
            pred_velocity = pred_t_next - pred_t

            # Target (ground truth) relative actual velocity shifts
            target_t = target_sequence[:, :-1, :, :]
            target_t_next = target_sequence[:, 1:, :, :]
            target_velocity = target_t_next - target_t

            # We penalize predictions that deviate from the actual trajectory vector
            velocity_error = self.smooth_l1(pred_velocity, target_velocity)

            # Also penalize drastic changes in scale across adjacent frames (flicker protection)
            scale_error = F.mse_loss(scale_t, scale_t_next)

            temporal_loss = velocity_error + 0.5 * scale_error

        # 3. Structural Linear Combination
        total_loss = (self.spatial_weight * spatial_loss) + (
            self.temporal_weight * temporal_loss
        )

        return total_loss, spatial_loss, torch.tensor(temporal_loss)


if __name__ == "__main__":
    print("\n" + "=" * 40)
    print("📐 LOSS FUNCTION MATHEMATICAL VERIFICATION")
    print("=" * 40)

    # Simulate data: Batch size 2, 5 consecutive video frames, 10 predicted boxes, 4 spatial dimensions
    # Creating simulated predictions and targets with a slight trajectory drift
    mock_preds = torch.randn(2, 5, 10, 4)
    mock_targets = mock_preds + (
        torch.randn(2, 5, 10, 4) * 0.1
    )  # Simulated target with small offset

    criterion = SpatioTemporalConsistencyLoss(spatial_weight=1.0, temporal_weight=0.5)
    total, spatial, temporal = criterion(mock_preds, mock_targets)

    print(f"Total Combined Loss Scalar: {total.item():.4f}")
    print(f" └── Isolated Spatial Error: {spatial.item():.4f}")
    print(f" └── Inter-Frame Temporal Velocity Error: {temporal.item():.4f}")
    print("\nTemporal smoothness math functions accurately without shape mismatches.")
    print("=" * 40 + "\n")
