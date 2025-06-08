# imaris_surface_parser

## Introduction
This library contains a script to voxelize the mesh output exporting surfaces from the [Imaris software](https://imaris.oxinst.com/), which the data is a (often) very large .wrl file.  This program was initially written to convert human-labeled alzheimer's plaques in the form of mesh data into a binary mask in order for the data to be suitable for machine learning.

### Export Surfaces from Imaris
1.  First, load up your image and open it in Surpass.
2.  In the Properties section, select the surface of interest.

![image](https://github.com/user-attachments/assets/6464222c-84f8-4ef3-8ac8-f6bbf7b85c7d)

3.  Go to the 3D View dropdown menu and click 'Export Selected Objects'.  This will allow you to save the surface as a .wrl file.

![image](https://github.com/user-attachments/assets/40eddf86-9908-4825-bd2f-709f6d5c5d55)


## Running
This program can be run in Command Prompt (Windows).  Not yet tested for Linux.

First, clone this repository and cd into it.
```
> python parse_wrl.py -i <input.wrl> -o <output> [ARGS]
```

### Arguments
* `--input`, `-i` (required, string): File path to input .wrl file containing the mesh data.  The .wrl, if opened, should contain all the coordinates and normals in single arrays.  (e.g., the first shape object will have the Coordinate and Normal arrays.  All the other shapes should use the same array.)
* `--output`, `-o` (required, string): File path to output directory where the binary mask image will be stored.  Ideally this directory should be non-existant or empty.  Directories with contents at the start will be deleted.
* `--num_threads`, `-n` (optional, int, default=1): Number of threads used for assembling the final output image.
* `--dx`, `-dx` (required, float): Voxel size in the x-axis
* `--dy`, `-dy` (required, float): Voxel size in the y-axis
* `--dz`, `-dz` (required, float): Voxel size in the z-axis
* `--x`, `-x` (required, int, 2 args): Minimum and maximum of the x-axis, respectively
* `--y`, `-y` (required, int, 2 args): Minimum and maximum of the y-axis, respectively
* `--z`, `-z` (required, int, 2 args): Minimum and maximum of the z-axis, respectively
* `--skip_to`, `-st` (optional, int, default=0): Start executing from a given step.
  * If skip_to == 0, the program will start executing from the start of the program
  * If skip_to == 1, the program will start executing from extracting normal vectors from the .wrl file
  * If skip_to == 2, the program will start executing from extracting triangles from the .wrl file
  * If skip_to == 3, the program will start executing from converting the triangle (mesh) data into voxel outputs
  * If skip_to == 4, the program will start executing from assembling the final image
  * Starting from each step assumes the previous steps have been completed and is meant for debugging purposes or if the program is interrupted.  Skipping steps may result in errors.
* `--flip_x`, `-fx` (optional): Including this argument will flip the final image over the x-axis.
* `--flip_y`, `-fy` (optional): Including this argument will flip the final image over the y-axis.
* `--flip_z`, `-fz` (optional): Including this argument will flip the final image over the z-axis.

## Output
The output will be located in the output directory provided, under the `image` folder as a series of 2-dimensional .tif files.  If other formats (e.g., .ims, .fnt) are needed, convert them using an external program.  See [image-processing-pipeline](https://github.com/ucla-brain/image-preprocessing-pipeline).

## Dependencies
* argparse, multiprocessing, numpy, open3d, pathlib, scipy, shutil, tqdm, tifffile

All of these can be found via the Package Installer for Python (pip)
```
pip install argparse multiprocessing numpy open3d pathlib scipy pytest-shutil tqdm tifffile
```
If also using [image-processing-pipeline](https://github.com/ucla-brain/image-preprocessing-pipeline), the `stitching` environment contains all the libraries necessary.

### Development

Developed by Aidan Jan, Hongwei Dong Lab (B.R.A.I.N) @ UCLA, 2025

### Disclaimer
This tool was developed independently from Imaris.  It is not affiliated nor endorsed by Bitplane or Oxford Instruments.
