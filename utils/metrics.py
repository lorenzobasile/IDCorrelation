import functools
import math
import torch
from utils.intrinsic_dimension import estimate_id
from utils.utils import cat, normalize, shuffle


def id_correlation(dataset1, dataset2, N=100, algorithm='twoNN', return_pvalue=True):
    dataset1=normalize(dataset1)
    dataset2=normalize(dataset2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    id_1 = estimate_id(dataset1.to(device), algorithm).item()
    id_2 = estimate_id(dataset2.to(device), algorithm).item()
    max_id = max(id_1, id_2)
    upper_bound = id_1+id_2
    lower_bound = min(id_1, id_2)
    original_id = estimate_id(cat([dataset1, dataset2]).to(device), algorithm).item()
    corr= (upper_bound - original_id) / (upper_bound - lower_bound)
    if return_pvalue:
        shuffled_id=torch.zeros(N, dtype=torch.float)
        for i in range(N):
            shuffled_id[i]=estimate_id(cat([dataset1, shuffle(dataset2)]).to(device), algorithm).item()
        p=(((shuffled_id<original_id).sum()+1)/(N+1)).item() #according to permutation test
    else:
        p=None
    return {'corr': corr, 'p': p}



def distance_correlation(latent, control):

    matrix_a = torch.cdist(latent, latent)
    matrix_b = torch.cdist(control, control)
    matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
    matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

    Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

    correlation_r = Gamma_XY / torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
    return correlation_r.item()

def linear_cka(x: torch.Tensor, y: torch.Tensor):
    return cka(x, y, hsic=linear_hsic)


def rbf_cka(x: torch.Tensor, y: torch.Tensor, *, sigma: float = None):
    return cka(x, y, hsic=functools.partial(kernel_hsic, sigma=sigma))


def cka(x: torch.Tensor, y: torch.Tensor, *, hsic: callable, tolerance=1e-6):


    assert x.shape[0] == y.shape[0], "X and Y must have the same number of samples."

    numerator = hsic(x, y)

    var1 = torch.sqrt(hsic(x, x))
    var2 = torch.sqrt(hsic(y, y))

    cka_result = numerator / (var1 * var2)

    assert 0 - tolerance <= cka_result <= 1 + tolerance, "CKA value must be between 0 and 1."

    return cka_result


def linear_hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute HSIC for linear kernels.

    This method is used in the computation of linear CKA.

    Args:
        X: shape (N, D), first embedding matrix.
        Y: shape (N, D'), second embedding matrix.

    Returns:
        The computed HSIC value.
    """
    # inter-sample similarity matrices for both spaces ~(N, N)
    L_X = x @ x.T
    L_Y = y @ y.T

    return torch.sum(center_kernel_matrix(L_X) * center_kernel_matrix(L_Y))


def kernel_hsic(x: torch.Tensor, y: torch.Tensor, *, sigma):
    """Compute HSIC (Hilbert-Schmidt Independence Criterion) for RBF kernels.

    This is used in the computation of kernel CKA.

    Args:
        X: shape (N, D), first embedding matrix.
        Y: shape (N, D'), second embedding matrix.
        sigma: The RBF kernel width.

    Returns:
        The computed HSIC value.
    """
    return torch.sum(center_kernel_matrix(rbf(x, sigma=sigma)) * center_kernel_matrix(rbf(y, sigma=sigma)))


def center_kernel_matrix(k: torch.Tensor) -> torch.Tensor:
    """Center the kernel matrix K using the centering matrix H = I_n - (1/n) 1 * 1^T. (Eq. 3 in the paper).

    This method is used in the calculation of HSIC.

    Args:
        K: The kernel matrix to be centered.

    Returns:
        The centered kernel matrix.
    """
    n = k.shape[0]
    unit = torch.ones([n, n]).type_as(k)
    identity_mat = torch.eye(n).type_as(k)
    H = identity_mat - unit / n

    return H @ k @ H


def rbf(x: torch.Tensor, *, sigma=None):
    """Compute the RBF (Radial Basis Function) kernel for a matrix X.

    If sigma is not provided, it is computed based on the median distance.

    Args:
        X: The input matrix (num_samples, embedding_dim).
        sigma: Optional parameter to specify the RBF kernel width.

    Returns:
        The RBF kernel matrix.
    """
    GX = x @ x.T
    KX = torch.diag(GX).type_as(x) - GX + (torch.diag(GX) - GX).T
    device = KX.device
    KX = KX.cpu()
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)

    KX = KX.to(device)

    KX *= -0.5 / (sigma * sigma)
    KX = torch.exp(KX)

    return KX
