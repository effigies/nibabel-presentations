# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # NiBabel
#
# <div style="float: right"><img src="https://nipy.org/nibabel/_static/reggie.png"></div>
#
# ## Neuroimaging data and file structures in Python
#
# ###### Christopher J Markiewicz
#
# ###### NeuroHackademy 2020

# %% [markdown] slideshow={"slide_type": "notes"}
# The goal of this presentation is to become familiar with loading, modifying, saving and visualizing neuroimaging data in Python.

# %% [markdown] slideshow={"slide_type": "slide"}
# # NiBabel
#
# NiBabel is a low-level Python library that gives access to a variety of imaging formats, with a particular focus on providing a common interface to the various volumetric formats produced by scanners and used in common neuroimaging toolkits:
#
# | | | |
# |:---: |:---: |:---:|
# | NIfTI-1 | NIfTI-2 | MGH |
# | MINC 1.0 | MINC 2.0 | AFNI BRIK/HEAD |
# | ANALYZE | SPM99 ANALYZE | SPM2 ANALYZE |
# | DICOM | PAR/REC | ECAT | 
#
# It also supports surface file formats:
#
# | | |
# |:--:|:--:|
# | GIFTI | FreeSurfer (FS) geometry |
# |FS labels | FS annotations |
#
# Tractography files:
#
# | | |
# |:--:|:--:|
# | TrackVis (TRK) | MRtrix (TCK) |
#
# As well as the CIFTI-2 format for composite volume/surface data.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Installation
#
# NiBabel is available on [PyPI](https://pypi.org/project/nibabel/):
#
# ```Shell
# pip install nibabel
# ```
#
# And [conda-forge](https://anaconda.org/conda-forge/nibabel):
#
# ```Shell
# conda install -c conda-forge nibabel
# ```

# %% slideshow={"slide_type": "subslide"}
import nibabel as nb

# %% slideshow={"slide_type": "fragment"}
# Some additional, useful imports

import numpy as np
import nilearn as nl
import nilearn.plotting
from matplotlib import pyplot as plt
import transforms3d

# %matplotlib inline

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Agenda
#
# 1. Basic input/output: loading and saving images
# 1. Image types
#    1. The `SpatialImage` API: volumetric images with affines (NIfTI and friends)
#    1. Surfaces and surface-sampled data (GIFTI and FreeSurfer geometry)
#    1. CIFTI-2
#    1. Tractography
# 1. The `DataobjImage` and `ArrayProxy` APIs: data scaling and memory management

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Basic I/O
#
# ### Loading

# %%
t1w = nb.load('/data/bids/openneuro/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz')
bold = nb.load('/data/bids/openneuro/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz')

# %% slideshow={"slide_type": "fragment"}
print(t1w)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Saving
#
# All NiBabel images have a `.to_filename()` method:

# %% slideshow={"slide_type": "-"}
t1w.to_filename('/tmp/img.nii.gz')

# %% [markdown] slideshow={"slide_type": "fragment"}
# `nibabel.save` will attempt to convert to a reasonable image type, if the extension doesn't match:

# %%
nb.save(t1w, '/tmp/img.mgz')

# %% [markdown] slideshow={"slide_type": "fragment"}
# Some image types separate header and data into two separate images. Saving to either filename will generate both.

# %%
nb.save(t1w, '/tmp/img.img')
print(nb.load('/tmp/img.hdr'))

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Serialization
#
# Some NiBabel images can be serialized to and deserialized from byte strings, for performing stream-based I/O.

# %%
bstr = t1w.to_bytes()
print(bstr[:100])

# %%
new_t1w = nb.Nifti1Image.from_bytes(bstr)

# %% [markdown] slideshow={"slide_type": "fragment"}
# Images that save to single files can generally be serialized. NIfTI-1/2, GIFTI and MGH are currently supported.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Spatial Images
#
# Spatial images contain three things:
#
# 1. The image *data* array: a 3D or 4D array of image data
# 1. An *affine* matrix: 4x4 array relating voxel coordinates and "world" coordinates
# 1. Image *metadata*: usually a format-specific header
#
# Many file types encode this basic structure. NiBabel will read any of ANALYZE (plain, SPM99, SPM2 and later), NIfTI-1, NIfTI-2, MINC 1.0, MINC 2.0, AFNI BRIK/HEAD, MGH, ECAT, DICOM and Philips PAR/REC, and expose a simple API:

# %%
data = t1w.get_fdata()
affine = t1w.affine
header = t1w.header

# %% [markdown] slideshow={"slide_type": "fragment"}
# ###### Aside
# Why not just `t1w.data`? Working with neuroimages can use a lot of memory, so nibabel works hard to be memory efficient. If it can read some data while leaving the rest on disk, it will. `t1w.get_fdata()` reflects that it's doing some work behind the scenes.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### The data array
#
# The data is a simple numpy array. It has a shape, it can be sliced and generally manipulated as you would any array.

# %%
print(data.shape)
i, j, k = np.array(data.shape) // 2
fig, axes = plt.subplots(1, 3)
axes[0].imshow(data[i,:,:].T, cmap='Greys_r', origin='lower')
axes[1].imshow(data[:,j,:].T, cmap='Greys_r', origin='lower')
_ = axes[2].imshow(data[:,:,k].T, cmap='Greys_r', origin='lower')

# %% [markdown]
# Each location in the image data array is a *voxel* (pixel with a volume), and can be referred to with *voxel coordinates* (array indices).
#
# This is a natural way to describe a block of data, but is practically meaningless with regard to anatomy.

# %% [markdown] slideshow={"slide_type": "subslide"}
# NiBabel has a basic viewer that scales voxels to reflect their size and labels orientations.

# %% slideshow={"slide_type": "-"}
_ = t1w.orthoview()

# %% [markdown]
# The crosshair is focused at the origin $(0, 0, 0)$.
#
# All of this information is encoded in the affine:

# %%
print(affine)

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Affine transforms
#
# The affine is a 4 x 4 numpy array. This describes the transformation from the voxel space (indices $(i, j, k)$) to a *reference* space (coordinates $(x, y, z)$). These coordinates are, by convention, distance in mm *right*, *anterior* and *superior* of an origin.
#
# $$
#     \begin{bmatrix}
#     x\\
#     y\\
#     z\\
#     1\\
#     \end{bmatrix} =
#     \mathbf A
#     \begin{bmatrix}
#     i\\
#     j\\
#     k\\
#     1\\
#     \end{bmatrix} =\begin{bmatrix}
#     m_{1,1} & m_{1,2} & m_{1,3} & a \\
#     m_{2,1} & m_{2,2} & m_{2,3} & b \\
#     m_{3,1} & m_{3,2} & m_{3,3} & c \\
#     0 & 0 & 0 & 1 \\
#     \end{bmatrix}
#     \begin{bmatrix}
#     i\\
#     j\\
#     k\\
#     1\\
#     \end{bmatrix}
# $$
#
# For an unmodified image, this reference space typically refers to an origin in the isocenter of the imaging magnet, and the directions right, anterior and superior are defined assuming a subject is lying in the scanner, face up.
#
# ![](https://nipy.org/nibabel/_images/localizer.png)
#
#

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### The affine as a series of transformations
#
# <!--
# <div style="float: left">
#     <img src="https://nipy.org/nibabel/_images/illustrating_affine.png">
# </div>
# -->
#
# ![](https://nipy.org/nibabel/_images/illustrating_affine.png)
#
# An affine transformation can be decomposed into translation, rotation, scaling (zooms) and shear transformations, applied right-to-left.

# %%
T, R, Z, S = transforms3d.affines.decompose44(affine)  # External library
print(f"Translation: {T}\nRotation:\n{R}\nZooms: {Z}\nMax shear: {np.abs(S).max()}")

# %%
Tm, Rm, Zm, Sm = [np.eye(4) for _ in range(4)]
Tm[:3, 3] = T
Rm[:3, :3] = R
Zm[:3, :3] = np.diag(Z)
Sm[[0, 0, 1], [1, 2, 2]] = S
np.allclose(Tm @ Rm @ Zm @ Sm, affine)

# %% [markdown] slideshow={"slide_type": "subslide"}
# NiBabel provides functions for extracting information from affines:
#
# * Orientation (or axis codes) indicates the direction an axis encodes. If increasing index along an axis moves to the right or left, the axis is coded "R" or "L", respectively.
# * Voxel sizes (or zooms)
# * Obliquity measures the amount of rotation from "canonical" axes.

# %%
print("Orientation:", nb.aff2axcodes(affine))
print("Zooms:", nb.affines.voxel_sizes(affine))
print("Obliquity:", nb.affines.obliquity(affine))

# %% [markdown] slideshow={"slide_type": "fragment"}
# You can also use it to answer specific questions. For instance, the inverse affine allows you to calculate indices from RAS coordinates and look up the image intensity value at that location.

# %%
i, j, k, _ = np.linalg.pinv(affine) @ [0, 0, 0, 1]
print(f"Center: ({int(i)}, {int(j)}, {int(k)})")
print(f"Value: ", data[int(i), int(j), int(k)])

# %% [markdown] slideshow={"slide_type": "skip"}
# `rescale_affine` modifies an affine to fit an image with different zooms or field-of-view while preserving rotations and the world coordinate of the central voxel:

# %% slideshow={"slide_type": "skip"}
print(nb.affines.rescale_affine(affine, t1w.shape, (1, 1, 1), (256, 256, 256)))

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### Don't Panic
#
# <div style="float: right"><img src="https://nipy.org/nibabel/_static/reggie.png"></div>
#
# If you didn't follow all of the above, that's okay. Here are the important points:
#
# 1. Affines provide the spatial interpretation of the data.
# 2. NiBabel has some useful methods for working with them.
#
# You'll go over this again with Noah Benson in [Introduction to the Geometry and Structure of the Human Brain](https://neurohackademy.org/course/introduction-to-the-geometry-and-structure-of-the-human-brain/).
#
# Matthew Brett's [Coordinate systems and affines](https://nipy.org/nibabel/coordinate_systems.html) tutorial is an excellent resource.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Image headers
#
# The image header is specific to each file format. Minimally, it contains information to construct the data and affine arrays, but some methods are useful beyond that. 
#
# * `get_zooms()` method returns voxel sizes. If the data is 4D, then repetition time is included as a temporal zoom.
# * `get_data_dtype()` returns the numpy data type in which the image data is stored (or will be stored when the image is saved).

# %%
print(bold.header.get_zooms())
print(bold.header.get_data_dtype())

# %% [markdown] slideshow={"slide_type": "subslide"}
# For images with fixed header structures, header fields are exposed through a `dict`-like interface.

# %%
print(header)

# %%
print(header['descrip'])

# %% [markdown] slideshow={"slide_type": "subslide"}
# The MGH header is similarly accessible, but its structure is quite different:

# %%
mghheader = nb.load('/tmp/img.mgz').header
print(mghheader)

# %%
print(mghheader['Pxyz_c'])
print(mghheader.get_zooms())
print(mghheader.get_data_dtype())


# %% [markdown]
# Other formats use text key-value pairs (PAR/REC, BRIK/HEAD) or keys in NetCDF (MINC 1.0) or HDF5 (MINC 2.0) containers, and their values are accessible by other means.
#
# Often, we're not particularly interested in the header, or even the affine. But it's important to know they're there and, especially, to remember to copy them when making new images, so that derivatives stay aligned with the original image.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Creating new images
#
# Reading data will only take you so far. In many cases, you will want to compute something on an image and save a new image containing the result.
#
# ```Python
# def my_function(image):
#     orig_data = image.get_fdata()
#     new_data = some_operation_on(orig_data)
#     return image_type(new_data, affine=image.affine, header=image.header)
# ```
#
# For example, perhaps we want to save space and rescale our T1w image to an unsigned byte:

# %%
def rescale(img):
    data = img.get_fdata()
    rescaled = ((data - data.min()) * 255. / (data.max() - data.min())).astype(np.uint8)
    return nb.Nifti1Image(rescaled, affine=img.affine, header=img.header)

rescaled_t1w = rescale(t1w)
rescaled_t1w.to_filename('/tmp/rescaled.nii.gz')

# %% [markdown]
# *Note*: Do not use this function. We will revisit it after discussing data types.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Data types and scaling factors
#
#

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Surfaces and surface-sampled data
#
# Surface data does not have the intrinsic geometry of a 3D array, so different structures are used.
#
# 1. The surface *mesh*
#    1. Vertices: a list of coordinates in world space
#    1. Triangles: a list of 3-tuples of indices into the coordinate list
# 2. The *data* array: a 1D or 2D array of values or vectors at each vertex
#
# Unlike spatial images, these components are frequently kept in separate files.

# %% [markdown] slideshow={"slide_type": "fragment"}
# ### Image-specific APIs
#
# The two supported surface formats at present are GIFTI and FreeSurfer geometry files.
#
# FreeSurfer encodes a great deal of information in its directory structure and file names.

# %%

# %% [markdown]
# ## Memory management
#
# Images that expose the data-header (`DataobjImage`) API have two ways of accessing data.
#
# `get_fdata()` returns the data 

# %%

# %%
