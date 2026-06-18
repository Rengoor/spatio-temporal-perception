import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        """
        A single ConvLSTM cell that processes one time-step frame.
        Preserves spatial dimensions using explicit padding.
        """
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2

        # Combine input gates and hidden gates into a single convolution for speed
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,  # 4 gates:
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True,
        )

    def forward(self, x, hidden_state=None):
        # x shape: (Batch, Channels, Height, Width)
        if hidden_state is None:
            batch_size, _, height, width = x.size()
            h_cur = torch.zeros(
                batch_size, self.hidden_channels, height, width, device=x.device
            )
            c_cur = torch.zeros(
                batch_size, self.hidden_channels, height, width, device=x.device
            )
        else:
            h_cur, c_cur = hidden_state

        # Concatenate input and hidden state along channel dimension
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)

        # Split gates evenly across channels
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_channels, dim=1)

        # Apply standard LSTM activations
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)  # Cell gate

        # Compute next cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)


class SpatioTemporalNeck(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=512):
        """
        Wraps the ConvLSTMCell to process an entire multi-frame video sequence tensor.
        """
        super(SpatioTemporalNeck, self).__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels)

    def forward(self, x):
        """Expected input shape from spatio-temporal data loader:
        (Batch, Time_Steps, Channels, Height, Width)"""
        batch_size, time_steps, seq_len, height, width = x.size()

        # Initialize hidden state
        hidden_state = None
        outputs = []

        # Unroll the sequence over time
        for t in range(time_steps):
            h_next, hidden_state = self.cell(x[:, t, :, :, :], hidden_state)
            outputs.append(h_next)

        # Stack outputs back along the time dimension: (Batch, Time_Steps, Hidden_Channels, H, W)
        outputs = torch.stack(outputs, dim=1)
        return outputs


if __name__ == "__main__":
    # Quick structural sanity check simulating a batch of video frames
    # 2 Batches, 5 consecutive Video Frames, 512 Channels, 20x20 Feature Map spatial size
    sample_tensor = torch.randn(2, 5, 512, 20, 20)
    neck = SpatioTemporalNeck(in_channels=512, hidden_channels=512)
    output_tensor = neck(sample_tensor)

    print("\n" + "=" * 40)
    print("MODEL SHAPE VERIFICATION")
    print("=" * 40)
    print(f"Input Video Sequence Tensor Shape:  {sample_tensor.shape}")
    print(f"Output Temporal Feature Tensor Shape: {output_tensor.shape}")
    print("=" * 40 + "\n")
