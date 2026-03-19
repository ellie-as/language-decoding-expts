import numpy as np
import torch
torch.set_default_tensor_type(torch.FloatTensor)

class EncodingModel():
    """class for computing the likelihood of observing brain recordings given a word sequence
    """
    def __init__(self, resp, weights, voxels, sigma, device = "cpu"):
        self.device = device
        self.weights = torch.from_numpy(weights[:, voxels]).float().to(self.device)
        self.resp = torch.from_numpy(resp[:, voxels]).float().to(self.device)
        self.sigma = sigma
        
    def set_shrinkage(self, alpha):
        """compute precision from empirical covariance with shrinkage factor alpha
        """
        precision = np.linalg.inv(self.sigma * (1 - alpha) + np.eye(len(self.sigma)) * alpha)
        self.precision = torch.from_numpy(precision).float().to(self.device)

    def prs(self, stim, trs):
        """compute P(R | S) on affected TRs for each hypothesis
        """
        with torch.no_grad(): 
            stim = stim.float().to(self.device)
            diff = torch.matmul(stim, self.weights) - self.resp[trs] # encoding model residuals
            multi = torch.matmul(torch.matmul(diff, self.precision), diff.permute(0, 2, 1))
            return -0.5 * multi.diagonal(dim1 = -2, dim2 = -1).sum(dim = 1).detach().cpu().numpy()

    def prs_per_voxel(self, stim, trs):
        """Decompose log P(R|S) into per-voxel contributions on affected TRs.

        The quadratic form d^T Σ^{-1} d decomposes exactly as
            Σ_j  d_j * (Σ^{-1} d)_j
        so the per-voxel terms sum to the total returned by prs().

        Returns (n_variants, n_voxels) array.
        """
        with torch.no_grad():
            stim = stim.float().to(self.device)
            diff = torch.matmul(stim, self.weights) - self.resp[trs]
            prec_diff = torch.matmul(diff, self.precision)
            per_voxel = -0.5 * (diff * prec_diff).sum(dim=1)
            return per_voxel.detach().cpu().numpy()