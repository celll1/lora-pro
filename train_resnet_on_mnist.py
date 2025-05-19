import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import os
import warnings

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
    warnings.warn("psutil not found. Memory usage monitoring will be skipped. Install with: pip install psutil")

# Attempt matplotlib import
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    warnings.warn("matplotlib not found. Plotting will be skipped. Install with: pip install matplotlib")

# Import custom optimizers
try:
    from came import CAME
    came_available = True
except ImportError:
    warnings.warn("Original CAME optimizer (came.py) not found. Skipping tests involving CAME.")
    CAME = None
    came_available = False

try:
    from bnb_came_8bit import Came8bit
    came_8bit_available = True
except ImportError:
    warnings.warn("Came8bit optimizer not found. Skipping tests involving Came8bit.")
    Came8bit = None
    came_8bit_available = False

try:
    from adafactor import Adafactor
    adafactor_available = True
except ImportError:
    warnings.warn("Original Adafactor optimizer (adafactor.py) not found. Skipping tests involving Adafactor.")
    Adafactor = None
    adafactor_available = False

try:
    from bnb_adafactor_8bit import Adafactor8bit
    adafactor_8bit_available = True
except ImportError:
    warnings.warn("Adafactor8bit optimizer not found. Skipping tests involving Adafactor8bit.")
    Adafactor8bit = None
    adafactor_8bit_available = False

# --- Helper for Memory Usage ---
def get_memory_usage_mb():
    """Returns current process RSS memory usage in MB."""
    if not psutil_available:
        return None
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) # Convert bytes to MB

# --- Model Definition ---
def get_resnet18_for_mnist(pretrained=False, num_classes=10):
    """
    Loads a ResNet18 model and adapts it for MNIST.
    - Changes the first convolutional layer to accept 1 input channel (grayscale).
    - Changes the final fully connected layer to output num_classes.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

    # Modify the first convolutional layer for 1 input channel (MNIST is grayscale)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=model.conv1.kernel_size,
                            stride=model.conv1.stride, padding=model.conv1.padding,
                            bias=False)

    # Modify the final fully connected layer for the number of classes in MNIST
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- DataLoader ---
def get_mnist_dataloader(batch_size=64, data_root='./data'):
    """Creates DataLoader for MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific normalization
    ])
    train_dataset = torchvision.datasets.MNIST(root=data_root, train=True,
                                               download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2 if psutil_available else 0, pin_memory=True) # Added num_workers and pin_memory

    test_dataset = torchvision.datasets.MNIST(root=data_root, train=False,
                                              download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, # Larger batch for testing
                             shuffle=False, num_workers=2 if psutil_available else 0, pin_memory=True)
    return train_loader, test_loader

# --- Training and Evaluation Loop ---
def run_training_test(optimizer_cls,
                      optimizer_name,
                      model_to_train,
                      train_loader,
                      test_loader,
                      device,
                      num_epochs=5, # Reduced default epochs for quicker tests
                      is_bnb_optimizer=False):
    print(f"\n--- Running Test: {optimizer_name} ---")
    start_time = time.time()
    model = model_to_train # Use the passed model, don't recreate here
    model.to(device) # Ensure model is on the correct device

    # Re-initialize model weights for a fair comparison for each optimizer test
    # This requires a function or a saved initial state if we want true weight re-init.
    # For now, assume the passed model_to_train is fresh or its state is handled outside.
    # A better approach: deepcopy the initial model state and load it.
    # For simplicity here, we will reinstantiate the model if not passed one that is already reset.
    # However, to ensure the SAME initial weights for all optimizers, we should pass a fresh model
    # OR pass an initial_state_dict and load it.
    # Let's create a fresh model instance here for each test for now, assuming get_resnet18_for_mnist gives consistent init.
    current_model = get_resnet18_for_mnist().to(device)

    if is_bnb_optimizer and not torch.cuda.is_available():
        print(f"Skipping {optimizer_name}: BitsAndBytes optimizer requires CUDA.")
        return None

    try:
        if optimizer_name == "Adafactor" or optimizer_name == "Adafactor8bit":
             # Adafactor specific defaults, lr=None for internal schedule
            optimizer = optimizer_cls(current_model.parameters(), lr=None, eps=(1e-30, 1e-3),
                                     clip_threshold=1.0, beta2_decay=-0.8, beta1=None, 
                                     weight_decay=0.0, scale_parameter=True, 
                                     relative_step=True, warmup_init=False)
        elif optimizer_name == "CAME" or optimizer_name == "Came8bit":
             optimizer = optimizer_cls(current_model.parameters(), lr=1e-3, weight_decay=0.01) # Simplified CAME params for now
        else: # Standard PyTorch optimizers like Adam, AdamW
            optimizer = optimizer_cls(current_model.parameters(), lr=1e-3)
    except Exception as e:
        print(f"ERROR: Failed to initialize optimizer {optimizer_name}: {e}")
        return None

    criterion = nn.CrossEntropyLoss()
    loss_history = []
    acc_history = []
    epoch_times = [] 
    vram_usage_bytes = [] # Store (allocated, peak_allocated) tuples

    initial_rss_mb = get_memory_usage_mb()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        initial_vram_allocated = torch.cuda.memory_allocated(device)
        vram_usage_bytes.append((initial_vram_allocated, torch.cuda.max_memory_allocated(device)))

    print(f"  Initial RSS: {initial_rss_mb:.2f} MB" if initial_rss_mb else "RSS N/A")
    if device.type == 'cuda':
        print(f"  Initial VRAM: {initial_vram_allocated/1024**2:.2f} MB")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        current_model.train()
        running_loss = 0.0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} ({optimizer_name})", leave=False) if tqdm_available else train_loader

        for i, (inputs, labels) in enumerate(iterator):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = current_model(inputs)
            loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR: Loss became NaN/Inf with {optimizer_name} at epoch {epoch+1}, batch {i}. Stopping test.")
                return None # Stop test if loss is invalid
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        # Evaluation
        current_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = current_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_acc = 100 * correct / total
        acc_history.append(epoch_acc)
        epoch_end_time = time.time()
        current_epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(current_epoch_time)

        if device.type == 'cuda':
            vram_usage_bytes.append((torch.cuda.memory_allocated(device), torch.cuda.max_memory_allocated(device)))

        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Time: {current_epoch_time:.2f}s")

    final_rss_mb = get_memory_usage_mb()
    total_training_time = time.time() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    print(f"  Finished Training - Optimizer: {optimizer_name}")
    print(f"    Total Time: {total_training_time:.2f}s, Avg Epoch Time: {avg_epoch_time:.2f}s")
    if initial_rss_mb and final_rss_mb:
        print(f"    RSS Memory: {initial_rss_mb:.2f} MB -> {final_rss_mb:.2f} MB (Delta: {final_rss_mb - initial_rss_mb:.2f} MB)")
    if device.type == 'cuda' and vram_usage_bytes:
        initial_vram_alloc_mb = vram_usage_bytes[0][0] / 1024**2
        final_vram_alloc_mb = vram_usage_bytes[-1][0] / 1024**2
        peak_vram_throughout_mb = max(peak for _, peak in vram_usage_bytes) / 1024**2
        print(f"    VRAM Allocated: {initial_vram_alloc_mb:.2f} MB -> {final_vram_alloc_mb:.2f} MB")
        print(f"    Peak VRAM Allocated (during test): {peak_vram_throughout_mb:.2f} MB")
        
    return {
        "optimizer_name": optimizer_name,
        "loss_history": loss_history,
        "acc_history": acc_history,
        "total_time_s": total_training_time,
        "avg_epoch_time_s": avg_epoch_time,
        "initial_rss_mb": initial_rss_mb,
        "final_rss_mb": final_rss_mb,
        "vram_usage_bytes_per_epoch": vram_usage_bytes # List of (allocated, peak_allocated) after each epoch (+ initial)
    }

def plot_results(results_list, num_epochs, device_type):
    if not matplotlib_available or not results_list:
        print("Plotting skipped: matplotlib not available or no results.")
        return

    num_optimizers = len(results_list)
    if num_optimizers == 0: return

    # Create plots in a grid
    # Plot 1: Loss vs Epochs
    # Plot 2: Accuracy vs Epochs
    # Plot 3: Peak VRAM Usage (if CUDA)
    # Plot 4: Total Training Time

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'ResNet18 MNIST Training Comparison on {device_type.upper()} ({num_epochs} Epochs)', fontsize=16)

    # Plot 1: Loss
    ax1 = axs[0, 0]
    for res in results_list:
        if res: ax1.plot(range(1, num_epochs + 1), res['loss_history'], label=res['optimizer_name'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Loss vs. Epochs')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Accuracy
    ax2 = axs[0, 1]
    for res in results_list:
        if res: ax2.plot(range(1, num_epochs + 1), res['acc_history'], label=res['optimizer_name'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Peak VRAM (only if CUDA and data available)
    ax3 = axs[1, 0]
    vram_plot_data = []
    if device_type == 'cuda':
        for res in results_list:
            if res and res.get('vram_usage_bytes_per_epoch'):
                # Use the max peak VRAM recorded during the *entire* test for this optimizer
                peak_vram_for_opt = max(peak for alloc, peak in res['vram_usage_bytes_per_epoch']) / (1024**2)
                vram_plot_data.append((res['optimizer_name'], peak_vram_for_opt))
    
    if vram_plot_data:
        opt_names = [x[0] for x in vram_plot_data]
        vram_values = [x[1] for x in vram_plot_data]
        ax3.bar(opt_names, vram_values)
        ax3.set_ylabel('Peak VRAM Allocated (MB)')
        ax3.set_title('Peak VRAM Usage')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'VRAM data not available or not on CUDA', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        ax3.set_title('Peak VRAM Usage')
    ax3.grid(True, axis='y')

    # Plot 4: Total Training Time
    ax4 = axs[1, 1]
    time_plot_data = []
    for res in results_list:
        if res: time_plot_data.append((res['optimizer_name'], res['total_time_s']))
    
    if time_plot_data:
        opt_names = [x[0] for x in time_plot_data]
        time_values = [x[1] for x in time_plot_data]
        ax4.bar(opt_names, time_values)
        ax4.set_ylabel('Total Training Time (s)')
        ax4.set_title('Total Training Time')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Time data not available', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
        ax4.set_title('Total Training Time')
    ax4.grid(True, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = f"resnet18_mnist_comparison_{device_type.lower()}.png"
    try:
        plt.savefig(plot_filename)
        print(f"\nPlot saved to {plot_filename}")
    except Exception as e:
        print(f"ERROR during plotting: {e}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")
    num_epochs_to_run = 5 # Define number of epochs for the tests

    # Prepare DataLoaders
    train_loader, test_loader = get_mnist_dataloader()
    print(f"MNIST DataLoaders created. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    optimizer_configs = []
    # if optim.Adam is not None: optimizer_configs.append({"cls": optim.Adam, "name": "Adam", "is_bnb": False})
    # if optim.AdamW is not None: optimizer_configs.append({"cls": optim.AdamW, "name": "AdamW", "is_bnb": False})
    if came_available and CAME is not None: optimizer_configs.append({"cls": CAME, "name": "CAME", "is_bnb": False})
    if came_8bit_available and Came8bit is not None: optimizer_configs.append({"cls": Came8bit, "name": "Came8bit", "is_bnb": True})
    if adafactor_available and Adafactor is not None: optimizer_configs.append({"cls": Adafactor, "name": "Adafactor", "is_bnb": False})
    if adafactor_8bit_available and Adafactor8bit is not None: optimizer_configs.append({"cls": Adafactor8bit, "name": "Adafactor8bit", "is_bnb": True})

    all_results = []

    for config in optimizer_configs:
        optimizer_cls = config["cls"]
        optimizer_name = config["name"]
        is_bnb = config["is_bnb"]
        
        # Create a fresh model instance for each optimizer to ensure fair comparison of initial state
        # Note: get_resnet18_for_mnist() should ideally have fixed random seeds for layers if not pretrained for true reproducibility
        # or we should save and load an initial state_dict.
        # For this test, default PyTorch initialization is used each time.
        model_instance = get_resnet18_for_mnist() 
                                        
        results = run_training_test(optimizer_cls=optimizer_cls,
                                    optimizer_name=optimizer_name,
                                    model_to_train=model_instance, # Pass the model instance
                                    train_loader=train_loader,
                                    test_loader=test_loader,
                                    device=device,
                                    num_epochs=num_epochs_to_run,
                                    is_bnb_optimizer=is_bnb)
        if results:
            all_results.append(results)
    
    if all_results:
        plot_results(all_results, num_epochs_to_run, device.type)
    else:
        print("No results to plot.")

    print("\nAll tests completed.") 