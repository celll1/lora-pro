import torch
import math
from torch.optim.optimizer import Optimizer
import warnings

class Adafactor(Optimizer):
    def __init__(self, params, lr=None, eps=(1e-30, 1e-3), # Changed eps1, eps2 to eps tuple
                 clip_threshold=1.0, beta2_decay=-0.8, # Renamed decay_rate to beta2_decay
                 beta1=None, weight_decay=0.0, 
                 scale_parameter=True, relative_step=None, # relative_step determined by lr
                 warmup_init=False, ams_grad=False):

        if lr is not None and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # Determine relative_step based on lr (if not explicitly passed for some reason)
        _relative_step = relative_step
        if _relative_step is None:
            _relative_step = lr is None
        
        if lr is None and not _relative_step:
            warnings.warn("lr is None but relative_step is False. Disabling relative_step.")
            # Or raise error, but for now, let's assume lr=None implies relative_step=True if not set.

        defaults = dict(lr=lr, beta1=beta1, eps1=eps[0], eps2=eps[1], # Store eps1 and eps2
                        clip_threshold=clip_threshold, beta2_decay=beta2_decay, 
                        weight_decay=weight_decay, scale_parameter=scale_parameter,
                        relative_step=_relative_step, warmup_init=warmup_init, 
                        ams_grad=ams_grad)
        super(Adafactor, self).__init__(params, defaults)
        # self.stable_sqrt_eps = eps[1] # eps2 is now in defaults

    def _get_learning_rate(self, group, p_data_for_scale, step):
        """Calculates the learning rate for the current step."""
        lr_from_group = group['lr']
        eps2 = group['eps2']
        
        # Determine base learning rate (rho_t)
        if not group['relative_step']: # lr is fixed or explicit relative_step=False
            if lr_from_group is None:
                rho_t = 1e-3 
                warnings.warn("lr is None and relative_step is False. Using a default rho_t.")
            else:
                rho_t = lr_from_group
        else: # relative_step is True (typically lr was None)
            if step == 0: # Should ideally be step >= 1 for Adafactor schedules
                rho_t = 1e-6 # Avoid issues with step=0
            else:
                # Reverting to use beta2_decay for rho_t calculation, as it might be working for 8bit version.
                rho_t = step ** group['beta2_decay'] 
            
            # Apply warmup if relative_step is True and warmup_init is True
            if group['warmup_init']:
                warmup_steps = 10000 # Default from Fairseq, can be a param
                if step < warmup_steps:
                    rho_t *= (step + 1.0) / warmup_steps

        # Scale learning rate by parameter scale if scale_parameter is True
        alpha_t = rho_t
        if group['scale_parameter']:
            param_rms = 1.0
            if p_data_for_scale is not None and p_data_for_scale.numel() > 0:
                param_rms = torch.sqrt(p_data_for_scale.pow(2).mean())
            else: # Should not happen for valid params, but as a fallback
                param_rms = torch.tensor(1.0, device=p_data_for_scale.device, dtype=p_data_for_scale.dtype)

            # Official: alpha_t = max(eps2, RMS(param)) * rho_t
            # Using torch.maximum for tensor-scalar comparison if param_rms is tensor
            alpha_scaler = torch.maximum(torch.tensor(eps2, device=param_rms.device, dtype=param_rms.dtype), param_rms)
            alpha_t = alpha_scaler * rho_t
            
        return alpha_t

    def _get_rms(self, tensor):
        return tensor.pow(2).mean().sqrt()

    def _get_shape_for_factorized_state(self, grad_shape, factor_dim):
        # factor_dim=0 for row (collapses last dim), factor_dim=1 for col (collapses second to last dim for 2D)
        if not grad_shape: return torch.Size([]) # scalar
        if len(grad_shape) == 1: # 1D tensor, no factorization in the same way, state is full shape
            return grad_shape
        # For 2D (N, M): row factor is (N,1), col factor is (1,M)
        if len(grad_shape) == 2:
            if factor_dim == 0: return torch.Size([grad_shape[0], 1]) # Row factor
            if factor_dim == 1: return torch.Size([1, grad_shape[1]]) # Col factor
        # Higher dimensions: Collapse all but the specified dimension for factorization
        # This simplification might need adjustment for >2D based on exact factorization strategy
        # For now, let's assume factor_dim refers to the dimension to keep (or similar)
        # Paper: if tensor is d_0 x d_1 x ... x d_k, factor R is d_0 x 1 x ... x 1, C is 1 x d_1 x ... x d_k etc.
        # This code assumes 2D or 1D factorization mainly.
        # If more complex, the shape logic would need to be more general.
        # Default to full shape if not clearly 2D factorization logic
        return grad_shape 

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1 = group["beta1"]
            eps1 = group["eps1"]
            eps2 = group["eps2"]
            clip_threshold = group["clip_threshold"] # This is 'd' in official
            weight_decay = group["weight_decay"]
            ams_grad = group["ams_grad"]
            beta2_decay = group["beta2_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                param_shape = p.shape

                if len(state) == 0:
                    state["step"] = 0
                    if beta1 is not None:
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    if len(param_shape) >= 2:
                        row_dim_shape = torch.Size([param_shape[0], 1])
                        col_dim_shape = torch.Size([1, param_shape[1]])
                        state["exp_avg_sq_row"] = torch.zeros(row_dim_shape, dtype=p.dtype, device=p.device)
                        state["exp_avg_sq_col"] = torch.zeros(col_dim_shape, dtype=p.dtype, device=p.device)
                        if ams_grad:
                            state['max_exp_avg_sq_row'] = torch.zeros_like(state["exp_avg_sq_row"])
                            state['max_exp_avg_sq_col'] = torch.zeros_like(state["exp_avg_sq_col"])
                    else: 
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if ams_grad:
                            state['max_exp_avg_sq'] = torch.zeros_like(state["exp_avg_sq"])
                
                state["step"] += 1
                current_lr = self._get_learning_rate(group, p.data, state["step"])
                
                # beta2_t: The decay factor for second moment estimates
                # Official: beta2_t = 1 - step^beta2_decay
                # beta2_decay is typically negative (e.g., -0.8), so step^beta2_decay is t^-0.8
                # which means beta2_t = 1 - t^-0.8 (increases towards 1)
                beta2_t = 1.0 - (state["step"] ** beta2_decay)


                grad_sq = grad.pow(2) # eps1 is not added here yet, will be used in sqrt

                rms_approx = None # This will be V_hat
                if len(param_shape) >= 2:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    
                    row_update_term = grad_sq.mean(dim=-1, keepdim=True)
                    col_update_term = grad_sq.mean(dim=-2, keepdim=True)
                    
                    exp_avg_sq_row.mul_(beta2_t).add_(row_update_term, alpha=1.0 - beta2_t)
                    exp_avg_sq_col.mul_(beta2_t).add_(col_update_term, alpha=1.0 - beta2_t)

                    if ams_grad:
                        # AMSGrad logic needs to be carefully checked against official if used
                        torch.maximum(state['max_exp_avg_sq_row'], exp_avg_sq_row, out=state['max_exp_avg_sq_row'])
                        torch.maximum(state['max_exp_avg_sq_col'], exp_avg_sq_col, out=state['max_exp_avg_sq_col'])
                        current_row_factor = state['max_exp_avg_sq_row']
                        current_col_factor = state['max_exp_avg_sq_col']
                    else:
                        current_row_factor = exp_avg_sq_row
                        current_col_factor = exp_avg_sq_col
                    
                    # Normalize column factor (sum over columns, add eps1 for stability before division)
                    # Using eps1 here because col_sum could be zero.
                    # PyTorch official uses L2 norm of col_factor for normalization.
                    # Current: col_sum = current_col_factor.sum(dim=-1, keepdim=True)
                    # normalized_col_factor = current_col_factor / (col_sum + eps1)
                    # Let's try to mimic L2 norm for col_factor normalization
                    col_l2_norm = torch.norm(current_col_factor, p=2, dim=-1, keepdim=True)
                    # Add eps1 to prevent division by zero if norm is zero
                    normalized_col_factor = current_col_factor / (col_l2_norm + eps1) 
                    
                    rms_approx = current_row_factor * normalized_col_factor

                else: # 1D or 0D (scalar) parameter
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2_t).add_(grad_sq, alpha=1.0 - beta2_t) # Use grad_sq
                    if ams_grad:
                        torch.maximum(state['max_exp_avg_sq'], exp_avg_sq, out=state['max_exp_avg_sq'])
                        rms_approx = state['max_exp_avg_sq']
                    else:
                        rms_approx = exp_avg_sq
                
                # Denominator for update: sqrt(V_hat + eps1) + eps2
                # Add eps1 inside sqrt for stability if V_hat is zero/small positive.
                # Then add eps2 outside sqrt.
                update_denominator = torch.sqrt(rms_approx.add(eps1)).add_(eps2)
                
                # Scaled gradient update term
                update = grad / update_denominator
                
                # Clipping the update term (U_hat = U_t / max(1, RMS(U_t)/d))
                # RMS(U_t)
                update_rms = torch.sqrt(update.pow(2).mean())
                clipping_denom = torch.maximum(torch.tensor(1.0, device=update_rms.device, dtype=update_rms.dtype), 
                                               update_rms / clip_threshold) # clip_threshold is 'd'
                update.div_(clipping_denom)


                if beta1 is not None:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)
                    update = exp_avg 

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * current_lr) # WD applied to p directly
                
                p.data.add_(update, alpha=-current_lr)
        return loss 