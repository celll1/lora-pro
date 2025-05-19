import torch
from torch.optim.optimizer import Optimizer
import bitsandbytes as bnb
import warnings
from typing import Dict, List, Tuple, Optional, Any, Iterable # Added Iterable
import math # Added for example usage
import copy
from came import CAME # Add import for CAME
from adafactor import Adafactor # Add import for Adafactor
from bnb_came_8bit import Came8bit # Add import for Came8bit
from bnb_adafactor_8bit import Adafactor8bit # Add import for Adafactor8bit

# --- Sylvester Equation Solver ---
# From: https://github.com/mrflogs/LoRA-Pro/blob/main/lora_pro/lora_pro/adamw.py
# Based on: https://stackoverflow.com/questions/73713072/solving-sylvester-equations-in-pytorch
def solve_sylvester(A, B, C, X=None):
    """
    Solves the Sylvester equation AY + YB = C for Y.
    Handles bfloat16 inputs by temporarily casting to float32.
    """
    original_dtype = A.dtype
    compute_dtype = torch.float32 # Default compute precision
    if original_dtype == torch.float64:
        compute_dtype = torch.float64

    # Cast bf16 inputs up to float32 for computation
    if original_dtype is torch.bfloat16 or A.dtype is torch.bfloat16 or B.dtype is torch.bfloat16 or C.dtype is torch.bfloat16:
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        C = C.to(torch.float32)
        # Ensure compute_dtype reflects this change if it wasn't already float32
        if compute_dtype != torch.float64: # Don't downgrade from float64
             compute_dtype = torch.float32
        # print("Warning: Casting Sylvester inputs to float32 for computation.")
    else:
        # Ensure inputs match the chosen compute_dtype otherwise
        A = A.to(compute_dtype)
        B = B.to(compute_dtype)
        C = C.to(compute_dtype)

    # The equation to solve is typically written AX + XB = C.
    # The solver here assumes AY + YB = C.
    # The LoRA-Pro code calls it with (B.T @ B, A @ A.T, -(...))
    # So, in the solver's notation: A_syl = B.T @ B, B_syl = A @ A.T, C_syl = -(...)
    # We are solving A_syl @ Y + Y @ B_syl = C_syl
    A_solver = A
    B_solver = B

    m = A_solver.shape[-1]
    n = B_solver.shape[-1]

    if A_solver.shape[-1] != A_solver.shape[-2] or B_solver.shape[-1] != B_solver.shape[-2]:
         raise ValueError(f"Matrices A ({A_solver.shape}) and B ({B_solver.shape}) must be square for eigenvalue decomposition.")

    # Determine the complex dtype corresponding to the compute_dtype
    complex_dtype = torch.complex64 if compute_dtype == torch.float32 else torch.complex128

    try:
        R, U = torch.linalg.eig(A_solver) # R, U can be complex
        S, V = torch.linalg.eig(B_solver) # S, V can be complex
    except torch.linalg.LinAlgError as e:
        warnings.warn(f"Eigenvalue decomposition failed in solve_sylvester: {e}. Returning zero matrix.")
        return torch.zeros_like(C).to(original_dtype)

    # Ensure subsequent calculations use complex numbers
    R_complex = R.to(complex_dtype)
    U_complex = U.to(complex_dtype)
    S_complex = S.to(complex_dtype)
    V_complex = V.to(complex_dtype)
    C_complex = C.to(complex_dtype)

    # Compute F = U_inv @ C @ V
    try:
        C_V = C_complex @ V_complex
        F = torch.linalg.solve(U_complex, C_V) # F is complex
    except torch.linalg.LinAlgError as e:
         warnings.warn(f"Solving F = U_inv @ C @ V failed: {e}. Using pseudo-inverse for U.")
         try:
             U_pinv = torch.linalg.pinv(U_complex)
             F = U_pinv @ C_complex @ V_complex
         except torch.linalg.LinAlgError as e_pinv:
             warnings.warn(f"Pseudo-inverse for U also failed: {e_pinv}. Returning zero matrix.")
             return torch.zeros_like(C).to(original_dtype)

    # Compute W = R[:, None] + S[None, :]
    try:
        W = R_complex.unsqueeze(-1) + S_complex.unsqueeze(-2) # W is complex
    except Exception as e:
        warnings.warn(f"Failed to compute W = R + S: {e}. Returning zero matrix.")
        return torch.zeros_like(C).to(original_dtype)

    if W.shape != F.shape:
        # This might happen if eig decomposition failed subtly
        warnings.warn(f"Shape mismatch between W {W.shape} and F {F.shape}. Returning zero matrix.")
        return torch.zeros_like(C).to(original_dtype)
        # raise ValueError(f"Shape mismatch between W {W.shape} and F {F.shape} in solve_sylvester.")

    # Avoid division by zero for complex numbers using magnitude
    W_abs = W.abs()
    # Add epsilon scaled by W itself to maintain direction, prevent division by zero
    epsilon_tensor = torch.full_like(W_abs, 1e-8)
    W_stable = W + torch.where(W_abs < epsilon_tensor, epsilon_tensor, W / (W_abs + epsilon_tensor) * epsilon_tensor)

    # Alternative stabilization: Just clamp small magnitudes in the denominator
    # W_denom = W.clone()
    # small_W_mask = W_abs < 1e-8
    # W_denom[small_W_mask] = 1e-8 # Replace small complex numbers with small real number
    # Y = F / W_denom

    Y = F / W_stable # Y is complex

    # Compute X = U @ Y @ V_inv
    try:
        V_inv = torch.linalg.inv(V_complex) # V_inv is complex
    except torch.linalg.LinAlgError as e:
        warnings.warn(f"Inverse of V failed: {e}. Using pseudo-inverse.")
        try:
            V_inv = torch.linalg.pinv(V_complex)
        except torch.linalg.LinAlgError as e_pinv:
            warnings.warn(f"Pseudo-inverse of V also failed: {e_pinv}. Returning zero matrix.")
            return torch.zeros_like(C).to(original_dtype)

    X = U_complex @ Y @ V_inv # X is complex

    # The solution X should be real if inputs A,B,C are real.
    # Take the real part to discard small imaginary components from numerical errors.
    final_X = X.real

    # Cast back to the original input dtype
    return final_X.to(original_dtype)
# ---------------------------------


class LoRAProOptimizerWrapper(Optimizer):
    """
    Wraps a bitsandbytes Optimizer or standard PyTorch Optimizer to apply LoRA-Pro adjustments.

    Args:
        base_optimizer (Optimizer): The underlying optimizer
            (e.g., bnb.optim.Adam8bit, torch.optim.AdamW).
        lora_alpha (float): The LoRA scaling factor alpha.
        lora_rank (int): The LoRA rank r.
        lora_param_map (Dict[int, Dict[str, List[torch.Tensor]]]):
            A map identifying LoRA parameters. Keys are parameter group indices.
            Values are dicts with keys 'A', 'B' containing lists of corresponding
            LoRA A and B parameter tensors in that group.
            Example: {0: {'A': [param_A1, param_A2], 'B': [param_B1, param_B2]}}
        lorapro_eps (float, optional): Epsilon value for numerical stability in
            LoRA-Pro calculations (matrix inversion, Sylvester solver). Defaults to 1e-8.
    """
    def __init__(self,
                 base_optimizer: Optimizer,
                 lora_alpha: float,
                 lora_rank: int,
                 lora_param_map: Dict[int, Dict[str, List[torch.Tensor]]],
                 lorapro_eps: float = 1e-8): # Added lorapro_eps

        # Check if the base optimizer is supported (moved check below param init)
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state
        self.lora_rank = lora_rank
        self.lora_scaling = lora_alpha / lora_rank if lora_rank > 0 else 0.0
        self.lora_param_map = lora_param_map
        self.lorapro_eps = lorapro_eps # Store eps

        # --- Check Base Optimizer Support ---
        supported_bnb_optimizers = []
        if bnb and hasattr(bnb, 'optim'): # Check if bnb is imported and has optim submodule
            supported_bnb_optimizers = [
                getattr(bnb.optim, 'Adam8bit', None),
                getattr(bnb.optim, 'AdamW8bit', None),
                getattr(bnb.optim, 'Lion8bit', None),
                getattr(bnb.optim, 'SGD8bit', None),
                Came8bit, # Add Came8bit
                Adafactor8bit # Add Adafactor8bit
            ]
        elif bnb: # Handle cases where optim might be directly under bnb (newer versions?)
             supported_bnb_optimizers = [
                getattr(bnb, 'Adam8bit', None),
                getattr(bnb, 'AdamW8bit', None),
                getattr(bnb, 'Lion8bit', None),
                getattr(bnb, 'SGD8bit', None),
                Came8bit, # Add Came8bit
                Adafactor8bit # Add Adafactor8bit
             ]


        supported_torch_optimizers = [
            getattr(torch.optim, 'Adam', None),
            getattr(torch.optim, 'AdamW', None),
            getattr(torch.optim, 'SGD', None),
            # Lion might need a specific import if not in torch.optim directly
            # Example: try: from lion_pytorch import Lion except ImportError: Lion = None
        ]

        # Add CAME and Adafactor to supported torch optimizers
        supported_torch_optimizers.extend([CAME, Adafactor])

        # Combine and filter None values
        all_supported_optimizers = supported_bnb_optimizers + supported_torch_optimizers
        supported_optimizers_filtered = tuple(opt for opt in all_supported_optimizers if opt is not None)

        is_supported = False
        # Allow inheritance checks
        for opt_type in supported_optimizers_filtered:
             if isinstance(base_optimizer, opt_type):
                 is_supported = True
                 break

        if not is_supported:
            # Get names safely, handling potential None in the original list before filtering
            supported_names = [opt.__name__ for opt in all_supported_optimizers if opt is not None]
            warnings.warn(f"Base optimizer type {type(base_optimizer)} might not be fully compatible "
                          f"or tested with LoRAProOptimizerWrapper. Supported base types include: "
                          f"{supported_names}")
        # ---------------------------------

        self._validate_lora_param_map()


    def _validate_lora_param_map(self):
        """Checks if the provided lora_param_map seems valid."""
        if not isinstance(self.lora_param_map, dict):
            raise TypeError("lora_param_map must be a dictionary.")
        param_set_in_map = set()
        for group_idx, map_dict in self.lora_param_map.items():
            if group_idx not in range(len(self.param_groups)):
                 raise ValueError(f"Group index {group_idx} in lora_param_map is out of range "
                                  f"for the optimizer's param_groups (found {len(self.param_groups)} groups).")
            if not isinstance(map_dict, dict) or 'A' not in map_dict or 'B' not in map_dict:
                raise ValueError(f"Invalid map structure for group {group_idx}. "
                                 "Expected {'A': [...], 'B': [...]}.")
            if len(map_dict['A']) != len(map_dict['B']):
                raise ValueError(f"Mismatch in number of LoRA A ({len(map_dict['A'])}) "
                                 f"and B ({len(map_dict['B'])}) parameters for group {group_idx}.")

            group_params = set(self.param_groups[group_idx]['params'])
            for p_a, p_b in zip(map_dict['A'], map_dict['B']):
                if not isinstance(p_a, torch.Tensor) or not isinstance(p_b, torch.Tensor):
                     raise TypeError(f"LoRA parameters must be torch.Tensor. Found {type(p_a)}, {type(p_b)}.")
                if p_a not in group_params or p_b not in group_params:
                     warnings.warn(f"LoRA pair (A: {p_a.shape}, B: {p_b.shape}) in group {group_idx} map "
                                   f"not found in the corresponding optimizer param group.")
                if not p_a.requires_grad or not p_b.requires_grad:
                     warnings.warn(f"LoRA pair (A: {p_a.shape}, B: {p_b.shape}) in group {group_idx} map "
                                   f"contains parameters where requires_grad=False.")
                param_set_in_map.add(p_a)
                param_set_in_map.add(p_b)

        # Check if mapped params are actually in the optimizer's trainable parameters *overall*
        # param_set_in_optimizer = set()
        # for group in self.param_groups:
        #     param_set_in_optimizer.update(p for p in group['params'] if p.requires_grad)
        # if not param_set_in_map.issubset(param_set_in_optimizer):
        #      warnings.warn("Some parameters in lora_param_map are not found in the optimizer's "
        #                    "trainable parameters across all groups.")


    def zero_grad(self, set_to_none: bool = True):
        """Zeros the gradients of all parameters managed by the base optimizer."""
        # Use the base optimizer's zero_grad for potentially optimized implementation
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        # Or manually:
        # for group in self.param_groups:
        #     for p in group['params']:
        #         if p.grad is not None:
        #             if set_to_none:
        #                 p.grad = None
        #             else:
        #                 if p.grad.grad_fn is not None:
        #                     p.grad.detach_()
        #                 else:
        #                     p.grad.requires_grad_(False)
        #                 p.grad.zero_()

    def _compute_adjusted_gradients(self,
                                    A: torch.Tensor,
                                    B: torch.Tensor,
                                    grad_A_origin: torch.Tensor,
                                    grad_B_origin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes LoRA-Pro adjusted gradients based on Theorem 2.3.
        This implements the Sylvester equation solution approach.

        Args:
            A: LoRA A parameter tensor (expects A.data).
            B: LoRA B parameter tensor (expects B.data).
            grad_A_origin: Original gradient for LoRA A (g_lora^A).
            grad_B_origin: Original gradient for LoRA B (g_lora^B).

        Returns:
            A tuple containing the adjusted gradients (g_A, g_B).
        """
        if self.lora_scaling == 0:
            warnings.warn("LoRA scaling factor is 0. Returning original gradients.")
            return grad_A_origin, grad_B_origin

        # Ensure calculations are done with appropriate precision, match reference code's dtype handling
        calc_dtype = A.dtype # Use parameter dtype by default
        # Cast to float32 if bfloat16 for stability, as done in solve_sylvester
        if calc_dtype == torch.bfloat16:
             calc_dtype = torch.float32
             A = A.to(calc_dtype)
             B = B.to(calc_dtype)
             grad_A_origin = grad_A_origin.to(calc_dtype)
             grad_B_origin = grad_B_origin.to(calc_dtype)


        scaling_sq = self.lora_scaling * self.lora_scaling
        eps = self.lorapro_eps

        # Precompute matrix products
        try:
            AA_T = A @ A.T # r x r
            B_TB = B.T @ B # r x r
        except RuntimeError as e:
            warnings.warn(f"Matrix multiplication failed during AA_T or B_TB calculation: {e}. "
                          f"Shapes: A={A.shape}, B={B.shape}. Returning original gradients.")
            return grad_A_origin.to(A.dtype), grad_B_origin.to(B.dtype) # Return original dtype

        # Compute pseudo-inverses with added epsilon for stability
        eye_A = torch.eye(A.shape[0], device=A.device, dtype=calc_dtype) # A.shape[0] is rank r
        eye_B = torch.eye(B.shape[0], device=B.device, dtype=calc_dtype) # B.shape[0] is output dim m

        try:
            AA_T_inv = torch.linalg.pinv(AA_T + eps * eye_A)
            B_TB_inv = torch.linalg.pinv(B_TB + eps * eye_A) # B_TB is also r x r
        except torch.linalg.LinAlgError as e:
             warnings.warn(f"Pseudo-inverse calculation failed: {e}. Returning original gradients.")
             return grad_A_origin.to(A.dtype), grad_B_origin.to(B.dtype) # Return original dtype


        # Compute intermediate gradients g_lora_A_tilde and g_lora_B_tilde
        # Reference: LoRA-Pro repo adamw.py, lines 175-176 (else block)
        try:
            g_lora_A_tilde = (1 / scaling_sq) * B_TB_inv @ grad_A_origin
            g_lora_B_tilde_term1 = (eye_B - B @ B_TB_inv @ B.T)
            g_lora_B_tilde = (1 / scaling_sq) * (g_lora_B_tilde_term1 @ grad_B_origin @ AA_T_inv)
        except RuntimeError as e:
             warnings.warn(f"Calculation of g_lora_A_tilde or g_lora_B_tilde failed: {e}. Returning original gradients.")
             return grad_A_origin.to(A.dtype), grad_B_origin.to(B.dtype) # Return original dtype

        # Solve the Sylvester equation: (B^T B) X + X (A A^T) = - (1/s^2) (B^T B)^+ g_lora^A A^T
        # Let Syl_A = B_TB, Syl_B = AA_T
        # Let Syl_C = - (1 / scaling_sq) * B_TB_inv @ grad_A_origin @ A.T
        Syl_A = B_TB
        Syl_B = AA_T
        try:
             Syl_C = - (1 / scaling_sq) * (B_TB_inv @ grad_A_origin @ A.T)
        except RuntimeError as e:
             warnings.warn(f"Calculation of Sylvester C matrix failed: {e}. Returning original gradients.")
             return grad_A_origin.to(A.dtype), grad_B_origin.to(B.dtype) # Return original dtype


        # Solve AY + YB = C where A=Syl_A, B=Syl_B, C=Syl_C
        try:
             # Ensure inputs to solver match expected precision
             X = solve_sylvester(Syl_A.to(calc_dtype), Syl_B.to(calc_dtype), Syl_C.to(calc_dtype))
             X = X.to(A.device) # Ensure device matches
        except Exception as e:
             warnings.warn(f"Sylvester solver failed: {e}. Returning original gradients.")
             # Attempt to return original gradients in the correct (original) dtype
             return grad_A_origin.to(A.dtype), grad_B_origin.to(B.dtype)


        # Compute final adjusted gradients g_A and g_B (Eq. 7)
        # g_A = g_lora_A_tilde + X @ A
        # g_B = g_lora_B_tilde - B @ X
        try:
            adjusted_grad_A = g_lora_A_tilde + X @ A
            adjusted_grad_B = g_lora_B_tilde - B @ X
        except RuntimeError as e:
            warnings.warn(f"Final adjusted gradient calculation failed: {e}. Returning original gradients.")
            return grad_A_origin.to(A.dtype), grad_B_origin.to(B.dtype)


        # Cast back to original parameter dtypes if necessary
        return adjusted_grad_A.to(A.dtype), adjusted_grad_B.to(B.dtype)


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step (parameter update)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        original_lora_grads: Dict[torch.Tensor, Optional[torch.Tensor]] = {} # Store Optional[Tensor]

        # 1. Compute and Store LoRA-Pro adjusted gradients
        adjusted_lora_grads: Dict[torch.Tensor, torch.Tensor] = {}

        for group_idx, lora_defs in self.lora_param_map.items():
            lora_A_params = lora_defs['A']
            lora_B_params = lora_defs['B']

            for A, B in zip(lora_A_params, lora_B_params):
                if A.grad is None or B.grad is None:
                    # Store None if grad is missing, important for restoration
                    original_lora_grads[A] = None
                    original_lora_grads[B] = None
                    continue

                # Store original grad before potentially calculating adjusted ones
                original_lora_grads[A] = A.grad.clone()
                original_lora_grads[B] = B.grad.clone()

                # Compute adjusted gradients
                try:
                    grad_A_adj, grad_B_adj = self._compute_adjusted_gradients(
                        A.data, B.data, A.grad, B.grad # Pass .data for A, B
                    )
                    adjusted_lora_grads[A] = grad_A_adj
                    adjusted_lora_grads[B] = grad_B_adj
                except Exception as e:
                     warnings.warn(f"Failed to compute adjusted gradients for a LoRA pair "
                                   f"(A:{A.shape}, B:{B.shape}): {e}. Using original gradients for this pair.")
                     # Keep original grads for the base optimizer step if adjustment fails
                     adjusted_lora_grads[A] = A.grad
                     adjusted_lora_grads[B] = B.grad


        # 2. Temporarily replace .grad for LoRA parameters with adjusted grads
        for param, adj_grad in adjusted_lora_grads.items():
             if param.grad is not None: # Check again, grad might have changed
                 # Ensure grad has same device and dtype BEFORE copy_
                 if param.grad.device != adj_grad.device or param.grad.dtype != adj_grad.dtype:
                      warnings.warn(f"Adjusted gradient dtype/device mismatch for param {param.shape}. "
                                    f"Grad: {param.grad.dtype}/{param.grad.device}, "
                                    f"Adjusted: {adj_grad.dtype}/{adj_grad.device}. Attempting copy anyway.")
                 try:
                     param.grad.copy_(adj_grad)
                 except RuntimeError as e:
                      warnings.warn(f"Failed to copy adjusted gradient to param {param.shape}: {e}")


        # 3. Call the base optimizer's step function
        # It will use the adjusted gradients for LoRA params in adjusted_lora_grads
        # and original gradients for all other parameters.
        self.base_optimizer.step()


        # 4. Restore original LoRA gradients in .grad
        # Crucial for correctness if step() is called multiple times, or for gradient accumulation,
        # or if external code inspects .grad after the step.
        for param, orig_grad in original_lora_grads.items():
            if orig_grad is None:
                param.grad = None # Restore None if it was None originally
            elif param.grad is not None: # Only restore if .grad still exists
                 # Ensure device and dtype match if necessary, though copy_ should handle it
                 try:
                     param.grad.copy_(orig_grad)
                 except RuntimeError as e:
                      warnings.warn(f"Could not restore original gradient for param {param.shape}: {e}")
            # else: Grad became None after base_optimizer.step(), cannot restore.


        return loss

    # --- Delegate other Optimizer methods to base_optimizer ---
    def state_dict(self):
        """Returns the state of the base optimizer as a :class:`dict`."""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Loads the base optimizer state."""
        self.base_optimizer.load_state_dict(state_dict)
        # Update self references after loading state (important!)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state


    def add_param_group(self, param_group):
        """Add a param group to the Optimizer's `param_groups` via the base optimizer."""
        warnings.warn("Adding param groups after LoRAProOptimizerWrapper initialization. "
                      "Ensure lora_param_map is updated MANUALLY if the new group contains LoRA parameters, "
                      "as the wrapper does not automatically detect them in added groups.")
        self.base_optimizer.add_param_group(param_group)
        # Update self.param_groups to stay in sync
        self.param_groups = self.base_optimizer.param_groups


# --- Helper function to identify LoRA parameters (Example) ---
def identify_lora_parameters(model: torch.nn.Module, param_groups: Optional[List[Dict]] = None) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    """
    Identifies LoRA A and B parameter pairs from a model assuming a common
    naming convention (e.g., '.lora_A.weight', '.lora_B.weight').

    Maps identified pairs to the correct parameter group index if param_groups are provided.

    Args:
        model: The model containing LoRA parameters.
        param_groups: Optional list of parameter groups from the optimizer,
                      used to map LoRA pairs to the correct group index. If None,
                      all pairs are assigned to group 0.

    Returns:
        A dictionary mapping parameter group index to LoRA A and B parameter lists.
        {group_idx: {'A': [lora_A1, ...], 'B': [lora_B1, ...]}}
    """
    lora_layers: Dict[str, Dict[str, Optional[torch.Tensor]]] = {} # Stores potential pairs by base layer name

    param_to_group_idx = {}
    if param_groups:
        for i, group in enumerate(param_groups):
            for p in group['params']:
                param_to_group_idx[p] = i

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_A = False
        is_B = False
        base_name = None

        # Adapt this logic based on your actual LoRA implementation's naming
        # Example: HuggingFace PEFT uses 'lora_A.default.weight', 'lora_B.default.weight'
        if "lora_A" in name: # More general check
            parts = name.split("lora_A")
            if len(parts) == 2 and (parts[1].startswith('.') or parts[1] == ''): # Check structure
                 base_name = parts[0]
                 is_A = True
        elif "lora_B" in name: # More general check
            parts = name.split("lora_B")
            if len(parts) == 2 and (parts[1].startswith('.') or parts[1] == ''): # Check structure
                 base_name = parts[0]
                 is_B = True

        if base_name is None:
            continue # Not identified as LoRA A or B by this convention

        if base_name not in lora_layers:
            lora_layers[base_name] = {'A': None, 'B': None}

        if is_A:
            if lora_layers[base_name]['A'] is not None:
                 warnings.warn(f"Multiple LoRA A candidates found for base name {base_name} ('{name}' vs existing). Using last found.")
            lora_layers[base_name]['A'] = param
        elif is_B:
            if lora_layers[base_name]['B'] is not None:
                 warnings.warn(f"Multiple LoRA B candidates found for base name {base_name} ('{name}' vs existing). Using last found.")
            lora_layers[base_name]['B'] = param


    # Organize pairs by group index
    grouped_lora_map: Dict[int, Dict[str, List[torch.Tensor]]] = {}

    for base_name, params in lora_layers.items():
        param_A = params['A']
        param_B = params['B']

        if param_A is not None and param_B is not None:
            group_idx_A = param_to_group_idx.get(param_A, 0) # Default to group 0 if no groups provided
            group_idx_B = param_to_group_idx.get(param_B, 0)

            if group_idx_A != group_idx_B:
                warnings.warn(f"LoRA pair for base name {base_name} belongs to different "
                              f"parameter groups ({group_idx_A} vs {group_idx_B}). Assigning to group of A ({group_idx_A}).")
                group_idx = group_idx_A
            else:
                group_idx = group_idx_A

            if group_idx not in grouped_lora_map:
                 grouped_lora_map[group_idx] = {'A': [], 'B': []}

            grouped_lora_map[group_idx]['A'].append(param_A)
            grouped_lora_map[group_idx]['B'].append(param_B)

        elif param_A is not None or param_B is not None:
            warnings.warn(f"Incomplete LoRA pair for base name {base_name}. "
                          f"Found A: {param_A is not None}, Found B: {param_B is not None}. Skipping pair.")

    if not grouped_lora_map:
         warnings.warn("No complete LoRA parameter pairs identified using the naming convention.")

    return grouped_lora_map


# --- Example Usage (Conceptual) ---
if __name__ == '__main__':

    # 1. Create your model with LoRA layers (using a dummy structure)
    class DummyLoRALayer(torch.nn.Module):
        def __init__(self, in_features, out_features, rank, alpha):
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features, bias=False)
            # LoRA parameters - use a naming convention identify_lora_parameters recognizes
            self.lora_A = torch.nn.Parameter(torch.zeros((rank, in_features))) # Name ends with lora_A
            self.lora_B = torch.nn.Parameter(torch.zeros((out_features, rank))) # Name ends with lora_B
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)
            self.scaling = alpha / rank if rank > 0 else 0.0
            self.rank = rank
            # Freeze original weight (typical for LoRA)
            self.linear.weight.requires_grad = False

        def forward(self, x):
             orig_output = self.linear(x)
             if self.rank > 0 and self.scaling > 0:
                 # Standard LoRA forward: Wx + sBAx
                 lora_delta = self.lora_B @ self.lora_A # Compute change
                 lora_output = torch.nn.functional.linear(x, lora_delta * self.scaling)
                 return orig_output + lora_output
             else:
                 return orig_output

    # Example model
    model = torch.nn.Sequential(
        DummyLoRALayer(10, 20, rank=4, alpha=8),
        torch.nn.ReLU(),
        DummyLoRALayer(20, 5, rank=4, alpha=8)
    )
    model.train()
    # Try with bfloat16
    try:
        model = model.to(torch.bfloat16)
        print("Model converted to bfloat16")
    except Exception as e:
        print(f"Could not convert model to bfloat16: {e}")


    # 2. Define parameter groups (optional, but good practice)
    lora_params = [p for name, p in model.named_parameters() if ('lora_A' in name or 'lora_B' in name) and p.requires_grad]
    other_params = [p for name, p in model.named_parameters() if not ('lora_A' in name or 'lora_B' in name) and p.requires_grad]

    param_groups = [
        {'params': lora_params, 'lr': 1e-4, 'weight_decay': 0.0}, # No WD for LoRA usually
        # Potentially add other groups here if needed
        # {'params': other_params, 'lr': 5e-5, 'weight_decay': 0.01}
    ]
    if not other_params:
         print("No non-LoRA trainable parameters found.")
         # If only LoRA params are trainable, just use one group:
         param_groups = [{'params': lora_params, 'lr': 1e-4}]


    # 3. Identify LoRA parameters using the helper, passing param_groups
    lora_param_map = identify_lora_parameters(model, param_groups)
    print("\nIdentified LoRA Param Map:", lora_param_map)
    if not lora_param_map:
        print("ERROR: LoRA parameter identification failed. Check model naming and helper function.")
        exit()


    # 4. Create the base bitsandbytes optimizer
    print(f"Number of parameter groups: {len(param_groups)}")
    # Ensure all required params are in groups
    all_params_in_groups = set()
    for group in param_groups:
        all_params_in_groups.update(group['params'])
    if not all_params_in_groups == set(p for p in model.parameters() if p.requires_grad):
         print("Warning: Not all trainable parameters are included in param_groups!")


    try:
        # Example: Using AdamW8bit with parameter groups
        # Use optim_dtype compatible with model dtype if possible (e.g., bfloat16)
        base_optimizer = bnb.optim.AdamW8bit(param_groups, optim_dtype=torch.bfloat16, betas=(0.9, 0.99), eps=1e-8)
        print("Using bnb.optim.AdamW8bit")

        # 5. Create the LoRA-Pro wrapper
        lora_alpha = 8 # Must match the value used in the model layers if scaling matters
        lora_rank = 4  # Must match the value used in the model layers
        lorapro_eps = 1e-6 # Slightly larger epsilon for LoRA-Pro stability
        optimizer = LoRAProOptimizerWrapper(
            base_optimizer=base_optimizer,
            lora_alpha=lora_alpha,
            lora_rank=lora_rank,
            lora_param_map=lora_param_map,
            lorapro_eps=lorapro_eps
        )

        # 6. Training loop
        dummy_input = torch.randn(4, 10).to(model[0].lora_A.device, model[0].lora_A.dtype) # Match device/dtype
        dummy_target = torch.randn(4, 5).to(model[0].lora_A.device, model[0].lora_A.dtype)
        criterion = torch.nn.MSELoss()

        print("\nStarting dummy training step...")
        for i in range(3): # More steps to see if state changes
            print(f"\n--- Step {i+1} ---")
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            print(f"Loss: {loss.item()}")

            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("ERROR: Loss is NaN or Inf. Stopping.")
                break

            loss.backward()
            print("Gradients computed.")

            # Check gradients before step (optional)
            first_lora_A = lora_param_map[0]['A'][0] if 0 in lora_param_map and lora_param_map[0]['A'] else None
            if first_lora_A and first_lora_A.grad is not None:
                 print(f"Grad LoRA A (before step, mean abs): {first_lora_A.grad.abs().mean().item():.4e}")
                 if torch.isnan(first_lora_A.grad).any() or torch.isinf(first_lora_A.grad).any():
                      print("ERROR: Grad LoRA A is NaN or Inf before step!")

            # Clip gradients (optional but often useful)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            print("Optimizer step completed.")

            # Check parameters after step (optional)
            if first_lora_A:
                 print(f"Param LoRA A (after step, mean abs): {first_lora_A.abs().mean().item():.4e}")
                 if torch.isnan(first_lora_A).any() or torch.isinf(first_lora_A).any():
                     print("ERROR: Param LoRA A is NaN or Inf after step!")


    except ImportError:
        print("\nSkipping example usage: bitsandbytes not found or import failed.")
        print("Please install bitsandbytes: pip install bitsandbytes")
    except Exception as e:
         print(f"\nAn error occurred during example usage: {e}")
         import traceback
         traceback.print_exc() 