import torch
import math
from torch.optim.optimizer import Optimizer
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
import warnings # Added for warnings

class Adafactor8bit(Optimizer):
    def __init__(self, params, lr=None, eps=(1e-30, 1e-3), # Changed eps1, eps2 to eps tuple
                 clip_threshold=1.0, beta2_decay=-0.8, # Renamed decay_rate to beta2_decay
                 beta1=None, weight_decay=0.0,
                 scale_parameter=True, relative_step=None, # relative_step determined by lr
                 warmup_init=False,    # Adafactor default
                 block_size=4096,
                 optim_dtype=torch.float32):
                 # stable_sqrt_eps was removed, eps tuple is used

        if lr is not None and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        _relative_step = relative_step
        if _relative_step is None:
            _relative_step = lr is None

        defaults = dict(
            lr=lr, beta1=beta1, eps1=eps[0], eps2=eps[1], 
            clip_threshold=clip_threshold, beta2_decay=beta2_decay, 
            weight_decay=weight_decay, scale_parameter=scale_parameter,
            relative_step=_relative_step, warmup_init=warmup_init
        )
        super().__init__(params, defaults)

        self.block_size = block_size
        self.optim_dtype = optim_dtype

    def _get_learning_rate(self, group, p_data_for_scale, step):
        """Calculates the learning rate for the current step."""
        lr_from_group = group['lr']
        eps2 = group['eps2']
        
        if not group['relative_step']:
            if lr_from_group is None:
                rho_t = 1e-3 
                warnings.warn("lr is None and relative_step is False. Using a default rho_t for 8bit.")
            else:
                rho_t = lr_from_group
        else: 
            if step == 0:
                rho_t = 1e-6 
            else:
                rho_t = step ** group['beta2_decay']
            
            if group['warmup_init']:
                warmup_steps = 10000 
                if step < warmup_steps:
                    rho_t *= (step + 1.0) / warmup_steps

        alpha_t = rho_t
        if group['scale_parameter']:
            param_rms = 1.0
            if p_data_for_scale is not None and p_data_for_scale.numel() > 0:
                # Ensure p_data_for_scale is in optim_dtype for RMS calculation
                param_rms = torch.sqrt(p_data_for_scale.to(self.optim_dtype).pow(2).mean())
            else: 
                param_rms = torch.tensor(1.0, device=p_data_for_scale.device, dtype=self.optim_dtype)
            
            alpha_scaler = torch.maximum(torch.tensor(eps2, device=param_rms.device, dtype=param_rms.dtype), param_rms)
            alpha_t = alpha_scaler * rho_t
            
        return alpha_t

    def _get_shape_for_factorized_state(self, grad_shape, factor_dim):
        # Re-using the logic from original Adafactor for consistency if needed
        if not grad_shape: return torch.Size([]) 
        if len(grad_shape) == 1:
            return grad_shape 
        if len(grad_shape) == 2:
            if factor_dim == 0: return torch.Size([grad_shape[0], 1])
            if factor_dim == 1: return torch.Size([1, grad_shape[1]])
        # For >2D, this might need more specific logic from Adafactor paper if factorization is more complex.
        # The bnb_adafactor_8bit seems to handle 1D vs >=2D for exp_avg_sq_row/col state init.
        # Let's assume current state init in step() is okay for shape, this func is for reference.
        if factor_dim == 0: # Row factor, collapses columns (last dim for 2D)
            return torch.Size(list(grad_shape[:-1]) + [1])
        elif factor_dim == 1: # Col factor, collapses rows (second to last dim for 2D)
             return torch.Size(list(grad_shape[:-2]) + [1] + [grad_shape[-1]]) if len(grad_shape) > 1 else torch.Size([1])
        return grad_shape # Fallback

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
            clip_threshold = group["clip_threshold"]
            weight_decay = group["weight_decay"]
            beta2_decay = group["beta2_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.to(self.optim_dtype) 
                if grad.is_sparse:
                    raise RuntimeError("Adafactor8bit does not support sparse gradients.")

                state = self.state[p]
                param_shape = p.shape

                if len(state) == 0:
                    state["step"] = 0
                    if beta1 is not None:
                        state["exp_avg_8bit"], state["exp_avg_absmax"] = quantize_blockwise(
                            torch.zeros(param_shape, dtype=self.optim_dtype, device=p.device), blocksize=self.block_size
                        )
                    if len(param_shape) >= 2:
                        s_rt_shape = torch.Size([param_shape[0], 1]) 
                        s_ct_shape = torch.Size([1, param_shape[1]]) 
                        state["exp_avg_sq_row_8bit"], state["exp_avg_sq_row_absmax"] = quantize_blockwise(torch.zeros(s_rt_shape, dtype=self.optim_dtype, device=p.device), blocksize=self.block_size)
                        state["exp_avg_sq_col_8bit"], state["exp_avg_sq_col_absmax"] = quantize_blockwise(torch.zeros(s_ct_shape, dtype=self.optim_dtype, device=p.device), blocksize=self.block_size)
                    else: 
                        state["exp_avg_sq_8bit"], state["exp_avg_sq_absmax"] = quantize_blockwise(torch.zeros(param_shape, dtype=self.optim_dtype, device=p.device), blocksize=self.block_size)
                        
                state["step"] += 1
                current_lr = self._get_learning_rate(group, p.data, state["step"])
                beta2_t = 1.0 - (state["step"] ** beta2_decay)
                
                grad_sq = grad.pow(2) # eps1 not added here

                rms_approx = None # V_hat
                if len(param_shape) >= 2:
                    exp_avg_sq_row = dequantize_blockwise(state["exp_avg_sq_row_8bit"], state["exp_avg_sq_row_absmax"], blocksize=self.block_size).to(self.optim_dtype)
                    exp_avg_sq_col = dequantize_blockwise(state["exp_avg_sq_col_8bit"], state["exp_avg_sq_col_absmax"], blocksize=self.block_size).to(self.optim_dtype)
                    
                    row_update_term = grad_sq.mean(dim=-1, keepdim=True)
                    col_update_term = grad_sq.mean(dim=-2, keepdim=True)

                    exp_avg_sq_row.mul_(beta2_t).add_(row_update_term.view_as(exp_avg_sq_row), alpha=1.0 - beta2_t)
                    exp_avg_sq_col.mul_(beta2_t).add_(col_update_term.view_as(exp_avg_sq_col), alpha=1.0 - beta2_t)
                    
                    col_l2_norm = torch.norm(exp_avg_sq_col, p=2, dim=-1, keepdim=True)
                    normalized_col_factor = exp_avg_sq_col / (col_l2_norm + eps1)
                    rms_approx = exp_avg_sq_row * normalized_col_factor
                    
                    state["exp_avg_sq_row_8bit"], state["exp_avg_sq_row_absmax"] = quantize_blockwise(exp_avg_sq_row, blocksize=self.block_size)
                    state["exp_avg_sq_col_8bit"], state["exp_avg_sq_col_absmax"] = quantize_blockwise(exp_avg_sq_col, blocksize=self.block_size)
                else: 
                    exp_avg_sq = dequantize_blockwise(state["exp_avg_sq_8bit"], state["exp_avg_sq_absmax"], blocksize=self.block_size).to(self.optim_dtype)
                    exp_avg_sq.mul_(beta2_t).add_(grad_sq, alpha=1.0 - beta2_t) # use grad_sq
                    rms_approx = exp_avg_sq
                    state["exp_avg_sq_8bit"], state["exp_avg_sq_absmax"] = quantize_blockwise(exp_avg_sq, blocksize=self.block_size)

                update_denominator = torch.sqrt(rms_approx.add(eps1)).add_(eps2)
                update = grad / update_denominator

                update_rms = torch.sqrt(update.pow(2).mean())
                clipping_denom = torch.maximum(torch.tensor(1.0, device=update_rms.device, dtype=update_rms.dtype), 
                                               update_rms / clip_threshold)
                update.div_(clipping_denom)

                if beta1 is not None:
                    exp_avg = dequantize_blockwise(state["exp_avg_8bit"], state["exp_avg_absmax"], blocksize=self.block_size).to(self.optim_dtype)
                    exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)
                    update = exp_avg
                    state["exp_avg_8bit"], state["exp_avg_absmax"] = quantize_blockwise(exp_avg, blocksize=self.block_size)
                
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * current_lr)
                
                p.data.add_(update, alpha=-current_lr)

        return loss 