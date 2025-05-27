# DistFuser
A Role-Disaggregated Inference Engine for Diffusion Transformers (DiTs) based on [xDiT](https://github.com/xdit-project/xDiT) and [verl](https://github.com/volcengine/verl).

## Table of Contents
- [DistFuser](#distfuser)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Supported DiTs](#supported-dits)
  - [Installation](#installation)
    - [Build from source](#build-from-source)
    - [Requirement: hi\_diffusers (Optional)](#requirement-hi_diffusers-optional)
  - [Usage](#usage)
  - [License](#license)

## Overview


## Supported DiTs
|                       Model Name                       |       CFG        |  SP   |  PP   | Performance Report |
| :----------------------------------------------------: | :--------------: | :---: | :---: | :----------------: |
| [HiDream-I1](https://github.com/HiDream-ai/HiDream-I1) | ✔️(for full type) |   ✔️   |   ❎   |        WIP         |


## Installation
### Build from source
```bash
git clone https://github.com/ZDJeffrey/DistFuser.git
cd DistFuser
pip install -e . # or pip install -e .[flash-attn,sage-attn] for more features
```
### Requirement: hi_diffusers (Optional)
To use HiDream models, you need to install the `hi_diffusers` package:
```bash
git clone https://github.com/HiDream-ai/HiDream-I1.git
echo '/path/to/HiDream-I1' > /path/to/site-packages/hi_diffusers.pth
# Note: the path to site-packages can be found by running:
# python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
# Remove the `hi_diffusers.pth` file if you want to uninstall the package.
```

> You have to install `hi_diffusers` manually because it does not support the `pip install` command yet.
> This requirement will be deprecated when `diffusers` supports HiDream models in the future releases.

## Usage
We porvide examples demonstrating how to run models with DistFuser in the [examples](./examples/) directory.
You can easily modify the options in the `./examples/*.sh` script to run the supported DiT models.
```bash
bash examples/run_hidream.sh
```

## License
This project is licensed under the [Apache 2.0 License](LICENSE).
