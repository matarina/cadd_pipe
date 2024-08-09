
## Resolving Deprecated API Usage in `torch_geometric`

When you encounter a situation where a deprecated API in `torch_geometric` is used in a project, and this API is only available in older versions (2.3 or earlier), you can resolve this by following these steps:

1. **Create a new Conda environment**: Start by creating an empty Conda environment.

2. **Install `torch_geometric` version 2.3**:
   ```bash
   pip install torch_geometric==2.3.*

3. **Install PyTorch with a specific version and CUDA support:
   ```bash
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

4. **Install CUDA: Use the official NVIDIA channel to install CUDA, as the Conda-Forge channel may not work correctly:
   ```bash
    conda install nvidia::cuda=11.6

Optional dependencies may cause errors. For guidance on installing them, refer to the [PyTorch Geometric installation page](https://pytorch-geometric.readthedocs.io/en/2.3.1/install/installation.html).

