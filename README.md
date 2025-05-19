# LoRA-Pro Optimizer Wrapper and Custom Optimizer Tests

This project implements and tests the LoRA-Pro optimizer wrapper, along with various base optimizers including custom implementations of CAME (Confidence-guided Adaptive Memory Efficient Optimization) and Adafactor, as well as their 8-bit quantized versions using bitsandbytes.

The primary goal is to evaluate the performance of these optimizers, both with and without the LoRA-Pro wrapper, on a simple MNIST image classification task.

## Features

*   Implementation of the LoRA-Pro optimizer wrapper.
*   Custom implementations of CAME and Adafactor optimizers.
*   8-bit quantized versions of CAME and Adafactor using `bitsandbytes`.
*   A test script (`test_wrapper.py`) to:
    *   Train a simple LoRA-equipped model on MNIST.
    *   Compare the performance (loss curves) of different optimizers (AdamW, Adam, CAME, Adafactor, and their 8-bit counterparts where applicable).
    *   Compare each optimizer with and without the LoRA-Pro wrapper.
    *   Generate plots siswa_loss_comparison_subplots_cuda.pngng the results.

## Project Structure

*   `lora_pro_optimizer_wrapper.py`: Contains the `LoRAProOptimizerWrapper` class and related utilities.
*   `came.py`: Implementation of the CAME optimizer.
*   `adafactor.py`: Implementation of the Adafactor optimizer.
*   `bnb_came_8bit.py`: 8-bit CAME implementation using bitsandbytes.
*   `bnb_adafactor_8bit.py`: 8-bit Adafactor implementation using bitsandbytes.
*   `test_wrapper.py`: The main script for running tests and generating plots.
*   `requirements.txt`: Python dependencies for the project.

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Ensure you have a compatible PyTorch version installed, especially if using CUDA features with bitsandbytes.

4.  **Run the tests:**
    ```bash
    python test_wrapper.py
    ```
    This will execute the training and evaluation for all configured optimizers and generate a plot named `mnist_loss_comparison_subplots_<device>.png`.

## Acknowledgements and Licenses

This project utilizes and is inspired by the following open-source projects. We are grateful to their authors and contributors.

*   **LoRA-Pro**: The LoRA-Pro optimizer wrapper implementation is based on the work presented in "[LoRA-Pro: Are Low-Rank Adapters Properly Optimized?](https://arxiv.org/abs/2407.18242)" by Wang et al. and the associated [official LoRA-Pro repository](https://github.com/mrflogs/LoRA-Pro) (MIT License).
*   **CAME Optimizer**: The implementation of the CAME optimizer is based on the paper "[CAME: Confidence-guided Adaptive Memory Efficient Optimization](https://arxiv.org/abs/2307.02047)" by Luo et al. and the [official CAME repository](https://github.com/yangluo7/CAME) (MIT License).
*   **PyTorch**: This project is built using [PyTorch](https://pytorch.org/). PyTorch is licensed under a [BSD-style license](https://github.com/pytorch/pytorch/blob/main/LICENSE).
*   **bitsandbytes**: The 8-bit optimizers leverage the [bitsandbytes library](https://github.com/TimDettmers/bitsandbytes) by Tim Dettmers et al. (MIT License).

Please refer to the respective repositories for full license details. 