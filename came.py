import torch
import math
from torch.optim.optimizer import Optimizer

class CAME(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99, 0.999), eps1=1e-6, eps2=1e-6,
                 clipping_threshold=1.0, weight_decay=0, amsgrad=False): # amsgrad for potential future use
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
        if not 0.0 < clipping_threshold: # d > 0 in paper
            raise ValueError(f"Invalid clipping_threshold value: {clipping_threshold}")

        defaults = dict(lr=lr, betas=betas, eps1=eps1, eps2=eps2,
                        clipping_threshold=clipping_threshold,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(CAME, self).__init__(params, defaults)
        self.stable_sqrt_eps = eps1 # Or a fixed small value like 1e-16

    def _get_shape_for_state(self, grad_shape, mean_dim):
        """ Helper to determine shape for row/col mean states based on grad_shape """
        if not grad_shape: # scalar
            return torch.Size([])
        if len(grad_shape) == 1: # 1D tensor
            if mean_dim == -1: # row mean for (M,) -> (M,)
                return grad_shape
            elif mean_dim == -2: # col mean for (M,) -> scalar (1,)
                return torch.Size([1]) if grad_shape[0] > 0 else torch.Size([])
        elif len(grad_shape) > 1: # 2D or more
            if mean_dim == -1: # row mean, collapses last dim
                return torch.Size(list(grad_shape[:-1]) + [1]) if len(grad_shape) > 1 else torch.Size([1])

            elif mean_dim == -2: # col mean, collapses second to last dim
                return torch.Size(list(grad_shape[:-2]) + [1] + [grad_shape[-1]]) if len(grad_shape) > 1 else torch.Size([1])

        # Fallback or for higher dimensions if specific logic is needed
        # For 2D (N,M): row mean -> (N,1), col mean -> (1,M)
        if len(grad_shape) == 2:
            if mean_dim == -1: return torch.Size([grad_shape[0], 1])
            if mean_dim == -2: return torch.Size([1, grad_shape[1]])
        
        return torch.Size([]) # Default fallback for unhandled cases

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
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('CAME does not support sparse gradients')

                state = self.state[p]
                param_shape = p.shape

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    # Determine shapes for factorized states based on grad_shape
                    # For 2D grad (N, M): r_t is (N,1), c_t is (1,M)
                    # For 1D grad (M,): r_t is (M,), c_t is (1,)
                    if grad.ndim == 0: # scalar
                        s_rt_shape = s_ct_shape = s_R_came_t_shape = s_C_came_t_shape = torch.Size([])
                    elif grad.ndim == 1: # 1D
                        s_rt_shape = grad.shape
                        s_ct_shape = torch.Size([1])
                        s_R_came_t_shape = grad.shape
                        s_C_came_t_shape = torch.Size([1])
                    else: # >= 2D
                        s_rt_shape = torch.Size([grad.shape[0]] + [1] * (grad.ndim - 1)) # Keep first dim, others 1
                        s_ct_shape = torch.Size([1] * (grad.ndim - 1) + [grad.shape[-1]]) # Keep last dim, others 1
                        if grad.ndim == 2: # Specific for 2D for clarity
                             s_rt_shape = torch.Size([grad.shape[0], 1])
                             s_ct_shape = torch.Size([1, grad.shape[1]])
                        s_R_came_t_shape = s_rt_shape
                        s_C_came_t_shape = s_ct_shape


                    state['r_t'] = torch.zeros(s_rt_shape, dtype=p.dtype, device=p.device)
                    state['c_t'] = torch.zeros(s_ct_shape, dtype=p.dtype, device=p.device)
                    state['R_came_t'] = torch.zeros(s_R_came_t_shape, dtype=p.dtype, device=p.device)
                    state['C_came_t'] = torch.zeros(s_C_came_t_shape, dtype=p.dtype, device=p.device)
                    if amsgrad:
                        state['max_S_t'] = torch.zeros_like(p, memory_format=torch.preserve_format)


                state['step'] += 1
                
                m_t = state['exp_avg']
                r_t = state['r_t']
                c_t = state['c_t']
                R_came_t = state['R_came_t']
                C_came_t = state['C_came_t']
                
                g_t = grad
                # For numerical stability, add eps1 before pow(2) if g_t can be very small,
                # or ensure grad_sq_plus_eps1 is positive. Paper adds after pow(2).
                grad_sq_plus_eps1 = g_t.pow(2).add_(eps1)


                if g_t.ndim == 0: # scalar
                    r_t_update_term = grad_sq_plus_eps1
                    c_t_update_term = grad_sq_plus_eps1
                elif g_t.ndim == 1: # 1D tensor (M,)
                    r_t_update_term = grad_sq_plus_eps1 # (M,)
                    c_t_update_term = grad_sq_plus_eps1.mean() # scalar
                else: # >= 2D
                    r_t_update_term = grad_sq_plus_eps1.mean(dim=-1, keepdim=True) # (N,1) for (N,M) grad
                    c_t_update_term = grad_sq_plus_eps1.mean(dim=-2, keepdim=True) # (1,M) for (N,M) grad
                
                r_t.mul_(beta2).add_(r_t_update_term.view_as(r_t), alpha=1 - beta2)
                c_t.mul_(beta2).add_(c_t_update_term.view_as(c_t), alpha=1 - beta2)
                
                # v_t = r_t c_t / (1_n^T r_t)
                # The division by sum is crucial. For 2D, r_t is (N,1), c_t is (1,M)
                # r_t.sum(dim=0) would be (1,1)
                if g_t.ndim == 0:
                    v_t_num = r_t * c_t
                    v_t_den = r_t.sum() + self.stable_sqrt_eps # Denom is scalar
                elif g_t.ndim == 1: # r_t is (M,), c_t is (1,)
                    v_t_num = r_t * c_t.view(-1) # (M,)
                    v_t_den = r_t.sum() + self.stable_sqrt_eps # Denom is scalar
                else: # r_t is (N,1), c_t is (1,M) for 2D
                    v_t_num = r_t * c_t # (N,M) via broadcast
                    # Denominator should normalize along the dimension that r_t varies
                    v_t_den = r_t.sum(dim=0, keepdim=True) + self.stable_sqrt_eps # (1,1) for 2D
                    if g_t.ndim > 2: # For >2D, sum over all but last of r_t's dims
                        sum_dims = tuple(range(r_t.ndim -1))
                        v_t_den = r_t.sum(dim=sum_dims, keepdim=True) + self.stable_sqrt_eps


                v_t = (v_t_num / v_t_den).expand_as(g_t) # Ensure broadcast to g_t shape
                u_t = g_t / (v_t.sqrt().add_(self.stable_sqrt_eps))

                rms_u_t = u_t.pow(2).mean().sqrt() # Global RMS for u_t
                # Clipping u_t
                u_hat_t = u_t / torch.max(torch.ones_like(rms_u_t), rms_u_t / d_clip)

                m_t.mul_(beta1).add_(u_hat_t, alpha=1 - beta1)
                U_t = (u_hat_t - m_t).pow(2) # Paper states (u_hat - m_t)^2
                U_t_plus_eps2 = U_t.add(eps2)

                if U_t.ndim == 0:
                    R_came_t_update_term = U_t_plus_eps2
                    C_came_t_update_term = U_t_plus_eps2
                elif U_t.ndim == 1:
                    R_came_t_update_term = U_t_plus_eps2
                    C_came_t_update_term = U_t_plus_eps2.mean()
                else: # >= 2D
                    R_came_t_update_term = U_t_plus_eps2.mean(dim=-1, keepdim=True)
                    C_came_t_update_term = U_t_plus_eps2.mean(dim=-2, keepdim=True)

                R_came_t.mul_(beta3).add_(R_came_t_update_term.view_as(R_came_t), alpha=1 - beta3)
                C_came_t.mul_(beta3).add_(C_came_t_update_term.view_as(C_came_t), alpha=1 - beta3)

                # S_t = R_came_t C_came_t / (1_n^T R_came_t)
                if U_t.ndim == 0:
                    S_t_num = R_came_t * C_came_t
                    S_t_den = R_came_t.sum() + self.stable_sqrt_eps
                elif U_t.ndim == 1:
                    S_t_num = R_came_t * C_came_t.view(-1)
                    S_t_den = R_came_t.sum() + self.stable_sqrt_eps
                else:
                    S_t_num = R_came_t * C_came_t
                    S_t_den = R_came_t.sum(dim=0, keepdim=True) + self.stable_sqrt_eps
                    if U_t.ndim > 2:
                        sum_dims = tuple(range(R_came_t.ndim -1))
                        S_t_den = R_came_t.sum(dim=sum_dims, keepdim=True) + self.stable_sqrt_eps
                
                S_t = (S_t_num / S_t_den).expand_as(g_t)

                if amsgrad:
                    max_S_t = state['max_S_t']
                    torch.maximum(max_S_t, S_t, out=max_S_t)
                    update_val_denom_sqrt = max_S_t.sqrt().add_(self.stable_sqrt_eps)
                else:
                    update_val_denom_sqrt = S_t.sqrt().add_(self.stable_sqrt_eps)
                
                # AdamW style weight decay
                if weight_decay != 0:
                    # Apply WD before the main update step, common for AdamW
                    p.data.mul_(1.0 - lr * weight_decay) 
                    # Note: Original Adam applies WD to grad: grad = grad.add(p.data, alpha=weight_decay)
                    # CAME paper does not specify WD. AdamW style is common.

                p.data.addcdiv_(m_t, update_val_denom_sqrt, value=-lr)
                
        return loss 