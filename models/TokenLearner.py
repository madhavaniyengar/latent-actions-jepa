import torch
import torch.nn as nn


class TokenLearner(nn.Module):
    """
    Token Learner module:
      - input shape: B x N x D   (B=batch, N=#tokens, D=dim)
      - output shape: B x K x D  (K < N, i.e., fewer learned tokens)
    """
    def __init__(self, embed_dim, num_output_tokens=8):
        super(TokenLearner, self).__init__()
        self.num_output_tokens = num_output_tokens
        self.attn_conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=embed_dim // 4,
                      kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=embed_dim // 4,
                      out_channels=num_output_tokens,
                      kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
          x: B x N x D
        Returns:
          out: B x K x D
        """
        B, N, D = x.shape
        # We apply the conv along the token dimension, so transpose to B x D x N
        x_t = x.transpose(1, 2)  # now B x D x N

        # attn_map: B x K x N (K = num_output_tokens)
        attn_map = self.attn_conv(x_t)       # shape (B, K, N)
        attn_map = attn_map.softmax(dim=-1)  # softmax over the token dimension

        # Weighted sum over tokens:
        #   - x_t is (B, D, N)
        #   - attn_map is (B, K, N)
        # We want out to be (B, K, D).
        # "einops" or manual matmul:
        out = torch.einsum("bdn,bkn->bkd", x_t, attn_map)
        
        return out
