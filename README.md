# Face Avatar Reconstruction (FAR)
FAR allows the creation of facial avatars by fitting a morphable model to images. It is a python wrapper around [EOS](https://github.com/patrikhuber/eos) 1.1.2 that adds some extra functionality:
- support fitting the 3d morphable models FexMM, FLAME, FaceGen and BFM2017
- use barycentric landmarks
- fitting multiple images of same person (requires [this fork](https://github.com/kanonet/eos))
- create a complete texture (only with FexMM)


# Installation

## 1. Install Conda and build tools
Install Conda, a compiler and build tools for your operating system. If you need more details, you may check [this guide](https://github.com/mgrewe/ovmf/blob/main/INSTALLATION.md#21-install-compiler-and-build-tools) section 2.1 and 2.2.

## 2. Setup Conda environment and clone FAR repository
Create a Conda environment for FAR and activate it:

    conda create -n far
    conda activate far

Clone [FAR](https://github.com/kanonet/face-avatar-reconstruction) repository:

    git clone --recurse-submodules https://github.com/kanonet/face-avatar-reconstruction.git
    cd face-avatar-reconstruction

Setup Conda environment:

    conda env update -f environment.yml

## 3. Install EOS
FAR comes with a [fork](https://github.com/kanonet/eos) of [EOS](https://github.com/patrikhuber/eos). The necessary functionality can be found on devel branch. Build and install the EOS python wrapper:

    cd contrib/eos
    python setup.py install

## 4. Convert morphable models
Convert the 3d morphable models to the EOS/FAR model format.  
Converter scripts for FexMM, FaceGen and FLAME can be found in the directory `face-avatar-reconstruction/converter`.  
For BFM use the converter script in the directory `face-avatar-reconstruction/contrib/eos/share/scripts`. (To use BFM you will need to copy some additional files from EOS repository to FAR.)  
Place the converted models inside their respective directories in `face-avatar-reconstruction/data/`.  
To receive the FexMM follow [this instruction](https://github.com/mgrewe/ovmf#fexmm-avatars).

# USAGE
Once your installation is complete and you converted all required models, you can start using FAR.  
Have a look at the examples: `examples/fit_single_person.py` (simple) and `examples/fit_and_evaluate_multiple_persons.py` (more complex).