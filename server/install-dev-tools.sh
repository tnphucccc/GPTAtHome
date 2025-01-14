#!/usr/bin/env bash
# Run this command from the PyTorch directory after cloning the source code using the “Get the PyTorch Source“ section below
pip install -r /workspaces/server/requirements.txt


# This takes some time
make setup-lint

# Add CMAKE_PREFIX_PATH to bashrc
echo 'export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}' >> ~/.bashrc
# Add linker path so that cuda-related libraries can be found
echo 'export LDFLAGS="-L${CONDA_PREFIX}/lib/ $LDFLAGS"' >> ~/.bashrc