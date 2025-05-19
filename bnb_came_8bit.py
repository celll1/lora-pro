import torch
import math
# import types # For SimpleNamespace if constructing q_args (not strictly needed for now)
from torch.optim.optimizer import Optimizer
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise

class Came8bit(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99, 0.999), # beta1 for m_t, beta2 for r_t/c_t, beta3 for R_t/C_t (CAME)
                 eps1=1e-6,  # For variance r_t, c_t (from g_t^2)
                 eps2=1e-6,  # For variance R_came_t, C_came_t (from U_t^2)
                 clipping_threshold=1.0,
                 weight_decay=0, # AdamW style weight decay handled here
                 block_size=4096,
                 optim_dtype=torch.float32): # dtype for calculations
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps1:
            raise ValueError(f"Invalid eps1 value: {eps1}")
        if not 0.0 <= eps2:
            raise ValueError(f"Invalid eps2 value: {eps2}")
        if not 0.0 < clipping_threshold: # d > 0
            raise ValueError(f"Invalid clipping_threshold value: {clipping_threshold}")

        defaults = dict(lr=lr, betas=betas, eps1=eps1, eps2=eps2,
                        clipping_threshold=clipping_threshold,
                        weight_decay=weight_decay)
        super(Came8bit, self).__init__(params, defaults)

        self.block_size = block_size
        self.optim_dtype = optim_dtype

    def _get_shape_for_state(self, grad_shape, mean_dim):
        """ Helper to determine shape for row/col mean states based on grad_shape """
        if not grad_shape: # scalar
            return torch.Size([])
        if len(grad_shape) == 1:
            if mean_dim == -1: # row mean for (M,) -> (M,) or scalar depending on interpretation
                return grad_shape
            elif mean_dim == -2: # col mean for (M,) -> scalar (1,)
                return torch.Size([1]) if grad_shape[0] > 0 else torch.Size([])
        elif len(grad_shape) > 1:
            if mean_dim == -1: # row mean
                return torch.Size([grad_shape[0], 1])
            elif mean_dim == -2: # col mean
                return torch.Size([1, grad_shape[1]])
        return torch.Size([]) # fallback

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            lr = group['lr']
            eps1 = group['eps1']
            eps2 = group['eps2']
            d_clip = group['clipping_threshold']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.to(self.optim_dtype)
                if grad.is_sparse:
                    raise RuntimeError('Came8bit does not support sparse gradients')

                state = self.state[p]
                param_shape = p.shape

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_8bit'], state['exp_avg_absmax'] = quantize_blockwise(
                        torch.zeros(param_shape, dtype=self.optim_dtype, device=p.device), blocksize=self.block_size
                    )
                    # r_t, c_t, R_came_t, C_came_t shapes based on grad
                    # These are factorized representations, so their shapes depend on grad's dimensions
                    # For a 2D grad (N, M): r_t is (N,1), c_t is (1,M)
                    # For a 1D grad (M,): r_t could be (M,), c_t could be (1,) (scalar-like)
                    # For a scalar grad (0-dim): r_t, c_t are scalars
                    
                    s_rt_shape = self._get_shape_for_state(grad.shape, -1)
                    s_ct_shape = self._get_shape_for_state(grad.shape, -2)
                    s_R_came_t_shape = self._get_shape_for_state(grad.shape, -1) # U_t has same shape as grad
                    s_C_came_t_shape = self._get_shape_for_state(grad.shape, -2)

                    state['r_t_8bit'], state['r_t_absmax'] = quantize_blockwise(torch.zeros(s_rt_shape, dtype=self.optim_dtype, device=p.device), blocksize=self.block_size)
                    state['c_t_8bit'], state['c_t_absmax'] = quantize_blockwise(torch.zeros(s_ct_shape, dtype=self.optim_dtype, device=p.device), blocksize=self.block_size)
                    state['R_came_t_8bit'], state['R_came_t_absmax'] = quantize_blockwise(torch.zeros(s_R_came_t_shape, dtype=self.optim_dtype, device=p.device), blocksize=self.block_size)
                    state['C_came_t_8bit'], state['C_came_t_absmax'] = quantize_blockwise(torch.zeros(s_C_came_t_shape, dtype=self.optim_dtype, device=p.device), blocksize=self.block_size)

                state['step'] += 1
                
                m_t = dequantize_blockwise(state['exp_avg_8bit'], state['exp_avg_absmax'], blocksize=self.block_size).to(self.optim_dtype)
                r_t = dequantize_blockwise(state['r_t_8bit'], state['r_t_absmax'], blocksize=self.block_size).to(self.optim_dtype)
                c_t = dequantize_blockwise(state['c_t_8bit'], state['c_t_absmax'], blocksize=self.block_size).to(self.optim_dtype)
                R_came_t = dequantize_blockwise(state['R_came_t_8bit'], state['R_came_t_absmax'], blocksize=self.block_size).to(self.optim_dtype)
                C_came_t = dequantize_blockwise(state['C_came_t_8bit'], state['C_came_t_absmax'], blocksize=self.block_size).to(self.optim_dtype)

                g_t = grad
                grad_sq_plus_eps1 = g_t.pow(2).add_(eps1)

                # r_t, c_t updates
                if g_t.ndim == 0: # scalar
                    r_t_update_term = grad_sq_plus_eps1
                    c_t_update_term = grad_sq_plus_eps1
                elif g_t.ndim == 1: # 1D tensor (M,)
                    r_t_update_term = grad_sq_plus_eps1 # (M,)
                    c_t_update_term = grad_sq_plus_eps1.mean() # scalar
                else: # >= 2D
                    r_t_update_term = grad_sq_plus_eps1.mean(dim=-1, keepdim=True) # (N,1)
                    c_t_update_term = grad_sq_plus_eps1.mean(dim=-2, keepdim=True) # (1,M)
                
                r_t.mul_(beta2).add_(r_t_update_term.view_as(r_t), alpha=1 - beta2)
                c_t.mul_(beta2).add_(c_t_update_term.view_as(c_t), alpha=1 - beta2)
                
                # v_t = r_t c_t / (1_n^T r_t)
                if g_t.ndim == 0:
                    v_t_num = r_t * c_t
                    v_t_den = r_t.sum().add_(eps1)
                elif g_t.ndim == 1: # r_t is (M,), c_t is scalar
                    v_t_num = r_t * c_t # (M,)
                    v_t_den = r_t.sum().add_(eps1) # scalar
                else: # r_t is (N,1), c_t is (1,M)
                    v_t_num = r_t * c_t # (N,M) via broadcast
                    v_t_den = r_t.sum(dim=0, keepdim=True).add_(eps1) # (1,M) or (1,1)
                
                v_t = (v_t_num / v_t_den).expand_as(g_t) # Ensure broadcast to g_t shape
                u_t = g_t / (v_t.sqrt().add_(eps1))

                rms_u_t = u_t.pow(2).mean().sqrt()
                u_hat_t = u_t / torch.max(torch.ones_like(rms_u_t), rms_u_t / d_clip)

                m_t.mul_(beta1).add_(u_hat_t, alpha=1 - beta1)
                U_t = (u_hat_t - m_t).pow(2)
                U_t_plus_eps2 = U_t.add(eps2)

                # R_came_t, C_came_t updates
                if U_t.ndim == 0:
                    R_came_t_update_term = U_t_plus_eps2
                    C_came_t_update_term = U_t_plus_eps2
                elif U_t.ndim == 1:
                    R_came_t_update_term = U_t_plus_eps2 # (M,)
                    C_came_t_update_term = U_t_plus_eps2.mean() # scalar
                else: # >= 2D
                    R_came_t_update_term = U_t_plus_eps2.mean(dim=-1, keepdim=True) # (N,1)
                    C_came_t_update_term = U_t_plus_eps2.mean(dim=-2, keepdim=True) # (1,M)

                R_came_t.mul_(beta3).add_(R_came_t_update_term.view_as(R_came_t), alpha=1 - beta3)
                C_came_t.mul_(beta3).add_(C_came_t_update_term.view_as(C_came_t), alpha=1 - beta3)

                # S_t = R_came_t C_came_t / (1_n^T R_came_t)
                if U_t.ndim == 0:
                    S_t_num = R_came_t * C_came_t
                    S_t_den = R_came_t.sum().add_(eps2)
                elif U_t.ndim == 1:
                    S_t_num = R_came_t * C_came_t
                    S_t_den = R_came_t.sum().add_(eps2)
                else:
                    S_t_num = R_came_t * C_came_t
                    S_t_den = R_came_t.sum(dim=0, keepdim=True).add_(eps2)
                
                S_t = (S_t_num / S_t_den).expand_as(g_t)
                update_val_denom_sqrt = S_t.sqrt().add_(eps2)
                
                # AdamW style weight decay
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay) 
                    # Note: CAME paper doesn't specify WD. This is AdamW style.
                    # Original Adam WD would be: m_t.add_(p.data, alpha=weight_decay) before normalization.

                p.data.addcdiv_(m_t, update_val_denom_sqrt, value=-lr)
                
                state['exp_avg_8bit'], state['exp_avg_absmax'] = quantize_blockwise(m_t, blocksize=self.block_size)
                state['r_t_8bit'], state['r_t_absmax'] = quantize_blockwise(r_t, blocksize=self.block_size)
                state['c_t_8bit'], state['c_t_absmax'] = quantize_blockwise(c_t, blocksize=self.block_size)
                state['R_came_t_8bit'], state['R_came_t_absmax'] = quantize_blockwise(R_came_t, blocksize=self.block_size)
                state['C_came_t_8bit'], state['C_came_t_absmax'] = quantize_blockwise(C_came_t, blocksize=self.block_size)
        return loss 