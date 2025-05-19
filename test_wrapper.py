import torch
import torch.nn as nn # Added nn
from torch.utils.data import DataLoader # Added DataLoader
import torchvision # Added torchvision
import torchvision.transforms as transforms # Added transforms
import warnings
import math # For dummy layer init
import copy # For deep copying state_dict
import time # For timing
import os # For process info
import itertools # For cycling dataloader

# Attempt tqdm import
try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    tqdm_available = False
    warnings.warn("tqdm not found. Progress bar will be disabled. Install with: pip install tqdm")

# Attempt psutil import for memory usage
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
    warnings.warn("psutil not found. Memory usage monitoring will be skipped. "
                  "Install with: pip install psutil")

# Attempt to import bitsandbytes, but don't fail if it's not installed or CPU-only env
try:
    import bitsandbytes as bnb
    is_bnb_available = torch.cuda.is_available()
    if not is_bnb_available:
         warnings.warn("bitsandbytes is imported, but CUDA is not available. "
                       "bitsandbytes optimizers will likely fail.")
except ImportError:
    warnings.warn("bitsandbytes not found. Skipping tests involving bnb optimizers.")
    bnb = None
    is_bnb_available = False

# Attempt matplotlib import
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    warnings.warn("matplotlib not found. Plotting will be skipped. Install with: pip install matplotlib")

# Import the wrapper and helper functions
try:
    from lora_pro_optimizer_wrapper import LoRAProOptimizerWrapper, identify_lora_parameters
except ImportError:
    print("ERROR: Could not import from lora_pro_optimizer_wrapper.py. Make sure it's in the same directory.")
    exit()

# Import new 8-bit optimizers
try:
    from bnb_came_8bit import Came8bit
    came_8bit_available = True
except ImportError:
    warnings.warn("Came8bit optimizer not found. Skipping tests involving Came8bit.")
    Came8bit = None
    came_8bit_available = False

try:
    from bnb_adafactor_8bit import Adafactor8bit
    adafactor_8bit_available = True
except ImportError:
    warnings.warn("Adafactor8bit optimizer not found. Skipping tests involving Adafactor8bit.")
    Adafactor8bit = None
    adafactor_8bit_available = False

# Import original optimizers
try:
    from came import CAME
    came_available = True
except ImportError:
    warnings.warn("Original CAME optimizer (came.py) not found. Skipping tests involving CAME.")
    CAME = None
    came_available = False

try:
    from adafactor import Adafactor
    adafactor_available = True
except ImportError:
    warnings.warn("Original Adafactor optimizer (adafactor.py) not found. Skipping tests involving Adafactor.")
    Adafactor = None
    adafactor_available = False


# --- LoRA Wrapper Layer Definition ---
class LoRAWrapperLayer(torch.nn.Module):
    """Wraps a linear layer and adds LoRA parameters."""
    def __init__(self, linear_layer: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear_layer
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        # Freeze the original linear layer weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
             self.linear.bias.requires_grad = False # Also freeze bias if exists

        # LoRA parameters (ensure they match linear layer's dtype and device)
        dtype = self.linear.weight.dtype
        device = self.linear.weight.device
        self.lora_A = torch.nn.Parameter(torch.zeros((rank, in_features), dtype=dtype, device=device))
        self.lora_B = torch.nn.Parameter(torch.zeros((out_features, rank), dtype=dtype, device=device))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)

        self.scaling = alpha / rank if rank > 0 else 0.0
        self.rank = rank

    def forward(self, x):
         orig_output = self.linear(x)
         if self.rank > 0 and self.scaling > 0:
             # Match calculation dtype with input for mixed precision
             lora_A_cal = self.lora_A.to(x.dtype)
             lora_B_cal = self.lora_B.to(x.dtype)
             lora_delta = lora_B_cal @ lora_A_cal
             # Apply LoRA update using functional linear
             lora_output = torch.nn.functional.linear(x, lora_delta * self.scaling)
             return orig_output + lora_output
         else:
             return orig_output

def create_dummy_model(device, dtype=torch.float32, rank=4, alpha=8):
    """Creates a deeper dummy model compatible with MNIST."""
    input_dim = 784 # MNIST 28*28
    output_dim = 10 # MNIST classes
    hidden_dim = 64 # Reduced hidden dim for faster CPU testing

    # Base layers (will be frozen)
    base_linear1 = torch.nn.Linear(input_dim, hidden_dim, bias=False, dtype=dtype)
    base_linear2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)
    base_linear3 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)
    head_layer = torch.nn.Linear(hidden_dim, output_dim, bias=False, dtype=dtype) # Trainable head

    # Wrap base layers with LoRA
    lora_layer1 = LoRAWrapperLayer(base_linear1, rank=rank, alpha=alpha)
    lora_layer2 = LoRAWrapperLayer(base_linear2, rank=rank, alpha=alpha)
    lora_layer3 = LoRAWrapperLayer(base_linear3, rank=rank, alpha=alpha)

    model = torch.nn.Sequential(
        nn.Flatten(),       # Flatten input image
        lora_layer1,
        nn.ReLU(),
        lora_layer2,
        nn.ReLU(),
        lora_layer3,
        nn.ReLU(),
        head_layer          # Output layer
    ).to(device)
    model.train()
    return model

# --- Helper for Memory Usage ---
def get_memory_usage_mb():
    """Returns current process RSS memory usage in MB."""
    if not psutil_available:
        return None
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) # Convert bytes to MB

# --- Helper for Smoothing Losses ---
def smooth_losses(losses, window_size=25):
    if not losses or len(losses) < window_size:
        return losses # Not enough data to smooth or no data
    smoothed = []
    for i in range(len(losses)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(losses), i + window_size // 2 + 1)
        window = losses[start_index:end_index]
        if window:
            smoothed.append(sum(window) / len(window))
        else:
            smoothed.append(losses[i]) # Should not happen if losses is not empty
    return smoothed

# --- Test Function (Modified for DataLoader) ---
def run_test(model: torch.nn.Module,
             initial_model_state: dict,
             dataloader: DataLoader, # Changed from dummy data
             num_steps: int,         # Added num_steps argument
             use_wrapper: bool,
             base_optimizer_cls,
             device,
             use_bnb=False,
             test_name=""):
    """
    Runs a test sequence using DataLoader for num_steps.
    Returns a list of loss values per step.
    """
    # --- Setup & Initialization ---
    wrapper_status = "with LoRA-Pro Wrapper" if use_wrapper else "without Wrapper (Baseline)"
    print(f"\n--- Running Test: {test_name} ({wrapper_status}) ---")
    print(f"    Device: {device}, Base Optimizer: {base_optimizer_cls.__name__}")

    # Memory usage at start
    start_mem_mb = get_memory_usage_mb()
    if start_mem_mb is not None:
        print(f"    Initial Memory (RSS): {start_mem_mb:.2f} MB")

    if use_bnb and not is_bnb_available:
        print("Skipping test: bitsandbytes optimizer requires CUDA.")
        return None

    # Reset model to initial state
    model.load_state_dict(initial_model_state)
    model.train() # Ensure model is in train mode

    # --- Add Debug Print ---
    current_param_sum = sum(p.data.sum().item() for p in model.parameters() if p.requires_grad)
    print(f"    Model state loaded. Trainable param sum: {current_param_sum:.8f}")
    # --- End Debug Print ---

    # Config (rank/alpha fixed based on model creation)
    lora_alpha = 8
    lora_rank = 4
    lorapro_eps = 1e-6

    # Parameter Groups (Re-identify based on the current model state)
    lora_params = []
    head_params = []
    # Use name check for more robustness than assuming Sequential index
    head_layer_name_part = "7." # Updated index for head layer name check
    # Or ideally, name the layer during creation: model.add_module("head", head_layer)

    # --- Debug: Print all named parameters to verify names and requires_grad ---
    # print("Model named parameters for grouping:")
    # for name, param in model.named_parameters():
    #     print(f"  {name}: requires_grad={param.requires_grad}")
    # --- End Debug ---

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_A' in name or 'lora_B' in name:
                lora_params.append(param)
            # Refine head detection: check if param belongs to the last module if it's Linear
            elif name.startswith(head_layer_name_part) and isinstance(model[-1], torch.nn.Linear):
                 head_params.append(param)

    # Fallback/Verification if needed (similar to before)
    if not head_params and isinstance(model[-1], torch.nn.Linear):
         print(f"WARN: Couldn't find head params by name '{head_layer_name_part}'. Using params from last module.")
         head_params = list(model[-1].parameters())

    if not lora_params: print("ERROR: No LoRA parameters found!"); return None
    if not head_params: print("ERROR: No Head parameters found!"); return None

    param_groups = [
        {'params': lora_params, 'lr': 1e-2, 'weight_decay': 0.0},
        {'params': head_params, 'lr': 1e-3, 'weight_decay': 0.01}
    ]
    # print(f"Parameter groups created: Group 0 (LoRA: {len(lora_params)} params), Group 1 (Head: {len(head_params)} params)")


    # Create lora_param_map for the wrapper
    lora_param_map = {}
    if use_wrapper:
        lora_A_in_group = [p for p in lora_params if hasattr(p, 'shape') and len(p.shape) > 1 and p.shape[0] == lora_rank]
        lora_B_in_group = [p for p in lora_params if hasattr(p, 'shape') and len(p.shape) > 1 and p.shape[1] == lora_rank]
        if len(lora_A_in_group) == len(lora_B_in_group) and len(lora_A_in_group) > 0:
             lora_param_map[0] = {'A': lora_A_in_group, 'B': lora_B_in_group}
             # print("Manually created lora_param_map for Group 0.")
        else: print("ERROR: Could not create lora_param_map."); return None


    # Instantiate Base Optimizer (always recreate for fair comparison)
    try:
        if use_bnb:
            # For 8-bit optimizers, assume they take param_groups and use internal lr if specified
            if base_optimizer_cls is Came8bit and came_8bit_available:
                # Came8bit specific defaults based on its own __init__
                base_optimizer = Came8bit(param_groups[0]['params'] + param_groups[1]['params'], 
                                          lr=1e-3, betas=(0.9, 0.99, 0.999), 
                                          eps1=1e-6, eps2=1e-6, clipping_threshold=1.0, weight_decay=0)
            elif base_optimizer_cls is Adafactor8bit and adafactor_8bit_available:
                combined_params = []
                for group in param_groups: combined_params.extend(group['params'])
                # Adafactor8bit specific defaults, lr=None for internal schedule
                base_optimizer = Adafactor8bit(combined_params, lr=None, eps=(1e-30, 1e-3),
                                             clip_threshold=1.0, beta2_decay=-0.8, beta1=None, 
                                             weight_decay=0.0, scale_parameter=True, 
                                             relative_step=True, warmup_init=False)
            else: # Other bnb optimizers if any
                 base_optimizer = base_optimizer_cls(param_groups)
        elif base_optimizer_cls is CAME and came_available: # Original CAME
            base_optimizer = CAME(param_groups[0]['params'] + param_groups[1]['params'], 
                                  lr=1e-3, betas=(0.9, 0.99, 0.999), 
                                  eps1=1e-6, eps2=1e-6, clipping_threshold=1.0, weight_decay=0)
        elif base_optimizer_cls is Adafactor and adafactor_available: # Original Adafactor
            combined_params = []
            for group in param_groups: combined_params.extend(group['params'])
            base_optimizer = Adafactor(combined_params, lr=None, eps=(1e-30, 1e-3),
                                         clip_threshold=1.0, beta2_decay=-0.8, beta1=None, 
                                         weight_decay=0.0, scale_parameter=True, 
                                         relative_step=True, warmup_init=False)
        else: # Standard PyTorch optimizers
            base_optimizer = base_optimizer_cls(param_groups)
    except Exception as e:
        if use_bnb and not is_bnb_available: print(f"Skip: bnb optim init failed on CPU ({e})"); return None
        else: print(f"ERROR: Failed base optim init: {e}"); return None

    # Instantiate Optimizer (Wrapper or Base)
    optimizer: torch.optim.Optimizer
    if use_wrapper:
        try:
            optimizer = LoRAProOptimizerWrapper(base_optimizer, lora_alpha, lora_rank, lora_param_map, lorapro_eps)
            # print("LoRAProOptimizerWrapper instantiated.")
        except Exception as e: print(f"ERROR: Wrapper init failed: {e}"); return None
    else:
        optimizer = base_optimizer
        # print("Using base optimizer directly (baseline).")


    # --- Training Loop (Modified for DataLoader) ---
    criterion = nn.CrossEntropyLoss() # Use CrossEntropyLoss for classification
    loss_history = []
    step_times = []
    data_iterator = itertools.cycle(dataloader) # Cycle through data if steps > dataset size

    print(f"Starting training loop ({num_steps} steps)...")
    initial_lora_params_vals = {id(p): p.data.clone() for p in lora_params}
    initial_head_params_vals = {id(p): p.data.clone() for p in head_params}

    loop_start_time = time.perf_counter()

    # Setup tqdm iterator
    if tqdm_available:
        loop_iterator = tqdm(range(num_steps), desc=f"{test_name} ({wrapper_status})", leave=False, unit="step")
    else:
        loop_iterator = range(num_steps)

    for i in loop_iterator:
        # --- Get Data ---
        try:
            inputs, targets = next(data_iterator)
            inputs, targets = inputs.to(device), targets.to(device)
            # Ensure input dtype matches model dtype (often float32 for models)
            model_param_dtype = next(model.parameters()).dtype
            inputs = inputs.to(model_param_dtype)
        except StopIteration:
            print("WARN: DataLoader exhausted unexpectedly (should cycle).")
            break # Stop if iterator fails
        except Exception as e:
            print(f"ERROR fetching data at step {i+1}: {e}"); return None

        # --- Zero Grad, Fwd/Bwd ---
        try:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            current_loss = criterion(outputs, targets)
            loss_history.append(current_loss.item())
            if i == 0: print(f"  Initial Loss: {current_loss.item():.6f}")
            if torch.isnan(current_loss) or torch.isinf(current_loss): print(f"ERROR: Loss NaN/Inf at step {i+1}."); return None
            current_loss.backward()
            grads_ok = all(p.grad is not None and not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any() for p in lora_params + head_params if p.requires_grad) # Check only trainable
            if not grads_ok: print(f"ERROR: Gradients invalid at step {i+1}."); return None
        except Exception as e: print(f"ERROR during fwd/bwd step {i+1}: {e}"); return None

        # --- Store pre-step grads ---
        pre_step_lora_grads = {}
        if use_wrapper:
            for p in lora_params: pre_step_lora_grads[p] = p.grad.clone() if p.grad is not None else None

        # --- Optimizer Step & Timing ---
        try:
            start_step_time = time.perf_counter()
            optimizer.step()
            end_step_time = time.perf_counter()
            step_times.append(end_step_time - start_step_time)
        except Exception as e: print(f"ERROR during step {i+1}: {e}"); import traceback; traceback.print_exc(); return None

        # --- Post-step Grad Check ---
        if use_wrapper:
            grads_restored = True
            for p in lora_params:
                 orig_grad = pre_step_lora_grads.get(p); curr_grad = p.grad
                 if (orig_grad is None and curr_grad is not None) or \
                    (orig_grad is not None and curr_grad is None) or \
                    (orig_grad is not None and not torch.equal(orig_grad, curr_grad)):
                     grads_restored = False; break
            if not grads_restored: print(f"ERROR: LoRA grad not restored after step {i+1}."); return None

    loop_end_time = time.perf_counter()
    total_loop_time = loop_end_time - loop_start_time

    # Final Checks
    final_loss = loss_history[-1] if loss_history else float('nan')
    lora_params_changed = any(not torch.equal(initial_lora_params_vals[id(p)], p.data) for p in lora_params)
    head_params_changed = any(not torch.equal(initial_head_params_vals[id(p)], p.data) for p in head_params)

    print(f"  Initial Loss: {loss_history[0]:.6f}" if loss_history else "N/A")
    print(f"  Final Loss: {final_loss:.6f}")
    if not lora_params_changed: print("WARN: LoRA parameters did not change!")
    if not head_params_changed: print("WARN: Head parameters did not change!")

    # Timing and Memory Results (same as before)
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    print(f"    Average step() time: {avg_step_time:.6f} seconds")
    print(f"    Total loop time ({num_steps} steps): {total_loop_time:.3f} seconds")
    end_mem_mb = get_memory_usage_mb()
    if start_mem_mb is not None and end_mem_mb is not None:
        mem_increase = end_mem_mb - start_mem_mb
        print(f"    Final Memory (RSS): {end_mem_mb:.2f} MB (Increase: {mem_increase:.2f} MB)")

    print(f"--- Test Completed: {test_name} ({wrapper_status}) ---")
    return loss_history, lora_params_changed, head_params_changed # Return flags too

def main():
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f" Running LoRA-Pro Wrapper Tests ({device.type.upper()}) - MNIST (500 steps)")
    print(f"==================================================")

    # --- Hyperparameters & Setup ---
    num_training_steps = 500 # Define number of steps here
    batch_size = 64
    all_passed = True
    results = {}
    change_status = {}

    print("\nLoading MNIST dataset...")
    # MNIST Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific
    ])
    try:
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print("MNIST dataset loaded.")
    except Exception as e:
        print(f"Failed to download MNIST: {e}")
        print("Please ensure you have an internet connection or download the dataset manually to ./data")
        return # Exit main if dataset fails to load

    # Create and save an initial model state
    print("\nCreating initial model (MNIST compatible)...")
    initial_model = create_dummy_model(device=device) # Pass device
    # Save the initial state of the model (deep copy)
    initial_model_state = copy.deepcopy(initial_model.state_dict())
    print("Initial model created and state saved.")

    # --- Optimizer Test Configurations ---
    optimizer_test_configs = []
    # AdamW
    if torch.optim.AdamW is not None:
        optimizer_test_configs.append({"cls": torch.optim.AdamW, "name": "AdamW", "use_bnb": False, "is_8bit": False})
    # Adam
    if torch.optim.Adam is not None:
        optimizer_test_configs.append({"cls": torch.optim.Adam, "name": "Adam", "use_bnb": False, "is_8bit": False})
    
    # CAME (Original and 8bit)
    if came_available and CAME is not None:
        optimizer_test_configs.append({"cls": CAME, "name": "CAME", "use_bnb": False, "is_8bit": False})
    if came_8bit_available and Came8bit is not None:
        optimizer_test_configs.append({"cls": Came8bit, "name": "Came8bit", "use_bnb": True, "is_8bit": True})

    # Adafactor (Original and 8bit)
    if adafactor_available and Adafactor is not None:
        optimizer_test_configs.append({"cls": Adafactor, "name": "Adafactor", "use_bnb": False, "is_8bit": False})
    if adafactor_8bit_available and Adafactor8bit is not None:
        optimizer_test_configs.append({"cls": Adafactor8bit, "name": "Adafactor8bit", "use_bnb": True, "is_8bit": True})

    # --- Run All Tests ---
    for config in optimizer_test_configs:
        base_optimizer_cls = config["cls"]
        opt_base_name = config["name"]
        use_bnb_flag = config["use_bnb"]
        is_8bit_flag = config["is_8bit"]

        if use_bnb_flag and not is_bnb_available:
            print(f"Skipping {opt_base_name} (8bit) as CUDA is not available for bitsandbytes.")
            continue
        if base_optimizer_cls is None: # Check if optimizer class itself is None (failed import)
            print(f"Skipping {opt_base_name} as the optimizer class is not available (None).")
            continue

        print(f"\n" + "="*20 + f" {opt_base_name} Tests " + "="*20)

        # --- Baseline Test (Optimizer without LoRA-Pro Wrapper) ---
        current_test_name_baseline = f"{opt_base_name}_Baseline"
        model = create_dummy_model(device=device)
        baseline_run_results = run_test(model=model, initial_model_state=initial_model_state,
                                        dataloader=train_loader, num_steps=num_training_steps,
                                        use_wrapper=False, base_optimizer_cls=base_optimizer_cls,
                                        device=device, use_bnb=use_bnb_flag, test_name=current_test_name_baseline)
        if baseline_run_results:
            results[current_test_name_baseline] = baseline_run_results[0]
            change_status[current_test_name_baseline] = (baseline_run_results[1], baseline_run_results[2])
        else:
            all_passed = False; change_status[current_test_name_baseline] = (None, None)

        # --- LoRA-Pro Wrapper Test ---
        current_test_name_wrapper = f"{opt_base_name}_Wrapper"
        model_w = create_dummy_model(device=device)
        wrapper_run_results = run_test(model=model_w, initial_model_state=initial_model_state,
                                       dataloader=train_loader, num_steps=num_training_steps,
                                       use_wrapper=True, base_optimizer_cls=base_optimizer_cls,
                                       device=device, use_bnb=use_bnb_flag, test_name=current_test_name_wrapper)
        if wrapper_run_results:
            results[current_test_name_wrapper] = wrapper_run_results[0]
            change_status[current_test_name_wrapper] = (wrapper_run_results[1], wrapper_run_results[2])
        else:
            all_passed = False; change_status[current_test_name_wrapper] = (None, None)

        print(f"\n--- {opt_base_name} Parameter Change Summary ---")
        bl_res = change_status.get(current_test_name_baseline, (None,None))
        wr_res = change_status.get(current_test_name_wrapper, (None,None))
        print(f"  Baseline: LoRA Changed={bl_res[0]}, Head Changed={bl_res[1]}")
        print(f"  Wrapper : LoRA Changed={wr_res[0]}, Head Changed={wr_res[1]}")

    # --- Plotting ---
    if matplotlib_available:
        print("\n" + "="*20 + f" Generating Plot (MNIST on {device.type.upper()}) " + "="*20)
        
        optimizer_names_for_plot = []
        # AdamW, Adam are standard
        if 'AdamW_Baseline' in results or 'AdamW_Wrapper' in results: optimizer_names_for_plot.append("AdamW")
        if 'Adam_Baseline' in results or 'Adam_Wrapper' in results: optimizer_names_for_plot.append("Adam")
        # CAME (Original and 8bit) - plot them together if both exist, or individually
        if came_available and ('CAME_Baseline' in results or 'CAME_Wrapper' in results): 
            optimizer_names_for_plot.append("CAME") # This will plot original CAME (Baseline vs Wrapper)
        if came_8bit_available and ('Came8bit_Baseline' in results or 'Came8bit_Wrapper' in results):
            optimizer_names_for_plot.append("Came8bit") # This will plot Came8bit (Baseline vs Wrapper)
        
        # Adafactor (Original and 8bit)
        if adafactor_available and ('Adafactor_Baseline' in results or 'Adafactor_Wrapper' in results):
            optimizer_names_for_plot.append("Adafactor")
        if adafactor_8bit_available and ('Adafactor8bit_Baseline' in results or 'Adafactor8bit_Wrapper' in results):
            optimizer_names_for_plot.append("Adafactor8bit")
        
        num_optimizers_to_plot = len(optimizer_names_for_plot)
        if num_optimizers_to_plot == 0:
            print("WARN: No results found to plot.")
            return

        # Determine grid size (2 columns)
        cols = 2
        rows = (num_optimizers_to_plot + cols - 1) // cols # Calculate rows needed
        
        fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
        fig.suptitle(f'MNIST Training Loss Comparison ({device.type.upper()}) - LoRA-Pro vs Baseline', fontsize=16)
        
        plot_idx = 0
        smoothing_window = 25 # Window size for smoothing

        for i in range(rows):
            for j in range(cols):
                if plot_idx < num_optimizers_to_plot:
                    opt_name_for_subplot = optimizer_names_for_plot[plot_idx]
                    ax = axs[i, j]
                    
                    # Keys for fetching results for this specific optimizer (e.g., AdamW, CAME, Came8bit)
                    baseline_key = f'{opt_name_for_subplot}_Baseline'
                    wrapper_key = f'{opt_name_for_subplot}_Wrapper'
                    
                    loss_baseline = results.get(baseline_key)
                    loss_wrapper = results.get(wrapper_key)
                    
                    if loss_baseline:
                        steps = list(range(len(loss_baseline)))
                        smoothed_baseline = smooth_losses(loss_baseline, smoothing_window)
                        ax.plot(steps, loss_baseline, alpha=0.3, color='blue', label='_nolegend_') # Thin raw loss
                        ax.plot(steps, smoothed_baseline, color='blue', linestyle='--', label=f'{opt_name_for_subplot} Baseline (Smoothed)')
                    
                    if loss_wrapper:
                        steps = list(range(len(loss_wrapper)))
                        smoothed_wrapper = smooth_losses(loss_wrapper, smoothing_window)
                        ax.plot(steps, loss_wrapper, alpha=0.3, color='orange', label='_nolegend_') # Thin raw loss
                        ax.plot(steps, smoothed_wrapper, color='orange', linestyle='-', label=f'{opt_name_for_subplot} LoRA-Pro (Smoothed)')

                    ax.set_title(f'{opt_name_for_subplot} Performance')
                    ax.set_xlabel("Training Steps")
                    ax.set_ylabel("Loss (Cross-Entropy)")
                    ax.legend()
                    ax.grid(True)
                    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0

                    plot_idx += 1
                else:
                    axs[i, j].axis('off') # Turn off unused subplots

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        plot_filename = f"mnist_loss_comparison_subplots_{device.type.lower()}.png"
        try:
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
        except Exception as e:
            print(f"ERROR during plotting: {e}")
    else:
        print("Plotting skipped as matplotlib is not available.")

    # --- Final Summary ---
    print("\n" + "="*50)
    print(f" Overall Test Summary ({device.type.upper()} MNIST Tests) ")
    print("="*50)
    if all_passed:
        print(f"✅ All {device.type.upper()} tests ran successfully (check individual results for pass/fail based on your criteria).")
    else:
        print(f"❌ Some {device.type.upper()} tests may have been skipped or encountered issues.")
    print("Review the logs and plots for detailed performance.")

# --- Main Execution ---
if __name__ == "__main__":
    main() 