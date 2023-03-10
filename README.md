# Face Avatar Reconstruction (FAR)
FAR allows the creation of facial avatars by fitting a morphable model to images. It is a python wrapper around [EOS](https://github.com/patrikhuber/eos) 1.1.2 that adds some extra functionality:
- support fitting the 3d morphable models FexMM, FLAME, FaceGen and BFM2017 (best results with FexMM)
- use barycentric landmarks
- fitting multiple images of same person (requires [this fork](https://github.com/kanonet/eos))
- create a complete texture (only tested with FexMM)


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
For BFM use the converter script in the directory `face-avatar-reconstruction/contrib/eos/share/scripts`.  
Place the converted models inside their respective directories in `face-avatar-reconstruction/data/`. To use image registration in texture reconstruction, you also need to provide a `reference_texture.png` which belongs to the morphable model.  
To receive the FexMM follow [this instruction](https://github.com/mgrewe/ovmf#fexmm-avatars).

# USAGE
Once your installation is complete and you have converted all the necessary models, you can finally start using FAR.  
To create an avatar, you need to provide a series of photographs of a person. Each picture must show the person well-lit and with a neutral expression. You should photograph the person from different angles, with the expression and lighting kept unchanged.  
You also have to identify [68 facial landmarks](https://ibug.doc.ic.ac.uk/media/uploads/images/300-w/figure_1_68.jpg) in each image. This can be done using [OpenFace](https://github.com/mgrewe/OpenFace), for example. Store the landmarks in an array of shape (68, 2) and save this in a CSV file named `landmark_coordinates_{image name}.csv`.  
Have a look at the examples: [fit_single_person.py (simple)](examples/fit_single_person.py) and [fit_and_evaluate_multiple_persons.py (more complex)](examples/fit_and_evaluate_multiple_persons.py). You will probably have to adapt the code to your own needs.  
When constructing your instance of `far.FaceAvatarReconstruction`, you have to specify which morphable model you want to use by setting the appropriate `far.ModelType`.
