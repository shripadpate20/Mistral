import torch


class ROPE():
    def __init__(self,head_dim:int, seq_len:int, theta:float=10000.0):
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.theta = theta
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (d_model/2) because we are using complex numbers to represent the rotations, so we only need half of the dimensions for the frequencies.
    
    def precompute_freqs(self):
        m = torch.arrange(self.seq_len, device=self.inv_freq.device) # Create a tensor of shape (seq_len,) containing the position indices from 0 to seq_len-1. This will be used to compute the frequencies for each position in the sequence.
        frequs = torch.outer(m,self.inv_freq).float() # Compute the outer product to get the frequencies for each position and each dimension. This will result in a matrix of shape (seq_len, head_dim/2) where each element (i, j) corresponds to the frequency for position i and dimension j.
        freq_complex = torch.polar(torch.ones_like(frequs), frequs) # Convert to complex numbers using polar coordinates (magnitude=1, angle=frequs)
        return freq_complex
    
    
    def forward(self,x:torch.Tensor):
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # Reshape to (batch_size, seq_len, num_heads, head_dim/2) and convert to complex numbers
        freq_complex = self.precompute_freqs().unsqueeze(0).unsqueeze(2) # Reshape to (1, seq_len, 1, head_dim/2) for broadcasting
        x_rotated = x_complex * freq_complex # Element-wise multiplication to apply the rotations
        x_rotated = torch.view_as_real(x_rotated).reshape(*x.shape) # Reshape back to the original shape (batch_size, seq_len, num_heads, head_dim) and convert back to real numbers
        return x_rotated
       
