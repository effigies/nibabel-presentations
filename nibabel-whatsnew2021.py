# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # NiBabel - What's New(ish)
#
# <div style="float: right"><img src="https://nipy.org/nibabel/_static/reggie.png"></div>
#
# ###### Christopher J Markiewicz
#
# ###### Nilearn Dev Days 2021

# %% [markdown] slideshow={"slide_type": "notes"}
# The goal of this presentation is to provide a summary of recent changes in NiBabel. NiBabel is a pretty slow-moving library, so I'm going to cover the range of NiBabel 2.4-3.2, roughly 2019-present.
#
# This document is intended to be viewed as a [RISE](https://rise.readthedocs.io/) presentation. It works fine as a notebook, but blocks with images may look strange because they are formatted to work as slides.

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
#
# *Note*: This notebook assumes NiBabel 3+, which requires a minimum Python version of 3.5.

# %% slideshow={"slide_type": "subslide"}
import nibabel as nb
print(nb.__version__)

# %% slideshow={"slide_type": "fragment"}
# Some additional, useful imports
from pathlib import Path              # Combine path elements with /
from pprint import pprint             # Pretty-printing

import numpy as np                    # Numeric Python
from matplotlib import pyplot as plt  # Matlab-ish plotting commands
from nilearn import plotting as nlp   # Nice neuroimage plotting
import transforms3d                   # Work with affine algebra
from scipy import ndimage as ndi      # Operate on N-dimensional images
import nibabel.testing                # For fetching test data

rng = np.random.default_rng()

# %pylab inline

# %%
data_dir = Path('/data/bids')

# %% [markdown] slideshow={"slide_type": "skip"}
# ## Learning objectives
#
# 1. Be able to load and save different types of files in NiBabel
# 1. Become familiar with the `SpatialImage` API and identify its components
# 1. Understand the differences between array and proxy images
# 1. Acquire a passing familiarity with the structures of surface images, CIFTI-2 files, and tractograms

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Basic I/O
#
# ### Loading (`pathlib.Path` support in `nibabel >=3`)

# %%
t1w = nb.load(data_dir / 'openneuro/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz')
bold = nb.load(data_dir / 'openneuro/ds000114/sub-01/ses-test/func/sub-01_ses-test_task-fingerfootlips_bold.nii.gz')

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
# ### Serialization (`nibabel >=2.5`)
#
# Some NiBabel images can be serialized to and deserialized from byte strings, for performing stream-based I/O.

# %%
bstr = t1w.to_bytes()
print(bstr[:100])

# %%
new_t1w = nb.Nifti1Image.from_bytes(bstr)

# %% [markdown] slideshow={"slide_type": "fragment"}
# Images that save to single files can generally be serialized. NIfTI-1/2, GIFTI, CIFTI-2, and MGH are currently supported.

# %% slideshow={"slide_type": "fragment"} tags=["raises-exception"]
nb.Nifti2Image.from_bytes(bstr)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Spatial Images
#
# For MRI studies, neuroimaging data is typically acquired as one or more *volumes*, a regular grid of values. NiBabel represents these data as *spatial images*, a structure containing three things:
#
# 1. The image *data* array: a 3D or 4D array of image data
# 1. An *affine* matrix: 4x4 array relating voxel coordinates and "world" coordinates
# 1. Image *metadata*: usually a format-specific header
#
# Many file types encode this basic structure. NiBabel will read any of ANALYZE (plain, SPM99, SPM2 and later), NIfTI-1, NIfTI-2, MINC 1.0, MINC 2.0, AFNI BRIK/HEAD, MGH, ECAT, DICOM and Philips PAR/REC, and expose a simple interface:

# %%
data = bold.get_fdata()
affine = bold.affine
header = bold.header

# %% [markdown] slideshow={"slide_type": "fragment"}
# Get *f*data? What happened to `get_data()`?

# %%
olddata = bold.get_data()

# %% [markdown] slideshow={"slide_type": "slide"}
# ### `get_data()` has been deprecated (`nibabel >=3.0`)
#
# `get_data()` does not have a consistent return type, due to image scaling. Scaling allows floating point data to be stored in an integer with fewer bits of precision.

# %%
random_data = rng.uniform(500, 2000, (2, 3, 4))
float_img = nb.Nifti1Image(random_data, affine=np.diag([2, 2, 2, 1]))
float_img.header.set_data_dtype(np.uint16)
float_img.to_filename('/tmp/float_img.nii')
nb.Nifti1Image(np.uint16(random_data), affine=np.diag([2, 2, 2, 1])).to_filename('/tmp/int_img.nii')


# %% [markdown] slideshow={"slide_type": "fragment"}
# The type depends on the on-disk type (`get_data_dtype()`), the slope and the intercept:

# %%
def show_img(fname):
    img = nb.load(fname)
    print(f"{fname}:\n{img.get_data_dtype()=} {img.dataobj.slope=}, {img.dataobj.inter=}\n{img.get_data().dtype=}")

with nb.testing.suppress_warnings():
    show_img("/tmp/float_img.nii")
    show_img("/tmp/int_img.nii")

# %% [markdown]
# A common gotcha with `get_data()` was when an algorithm written on datasets where `get_data()` returned `float`s hit overflow or rounding issues on a dataset that stored unscaled `int`s.

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### How to use `get_fdata()` and `dataobj`
#
# 1. When in doubt, use `img.get_fdata()`. It will fetch all of the data, and it will always be a float. You can control the size of the float.

# %%
float_img = nb.load('/tmp/float_img.nii')
print(float_img.get_fdata().dtype)
print(float_img.get_fdata(dtype=np.float16).dtype)

# %% [markdown] slideshow={"slide_type": "fragment"}
# 2. `img.dataobj` exists if you want to load only some data or control the data type. When you create an image, this is an `array`. If you load from disk, it's an `ArrayProxy`.

# %%
float_img.dataobj[:, :, 0]  # Just one slice

# %%
np.uint32(float_img.dataobj)  # I like ints!

# %%
np.asanyarray(float_img.dataobj)  # Surprise me!

# %% [markdown] slideshow={"slide_type": "fragment"}
# 3. Both methods transparently scale data when scale factors are present.

# %% slideshow={"slide_type": "slide"}

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Slicing images (`nibabel >=2.3`... docs in 3.0)
#
# Slicing array proxies is nice. Wouldn't it be nicer to keep track of the affine and header?
#
# The `slicer` attribute provides an interface that allows you to apply slices to an image, and updates the affine to ensure that the spatial information matches.
#
# Consider the T1-weighted image from earlier:

# %%
_ = t1w.orthoview()

# %% [markdown] slideshow={"slide_type": "subslide"}
# We can use the slicer to crop unneeded voxels in the left-right and inferior-superior directions:

# %%
cropped = t1w.slicer[40:216, :, 50:226]
cropped.orthoview()

# %% [markdown]
# Note the origin crosshair points to the same structure. The affines now differ in translation:

# %%
print(cropped.affine - t1w.affine)

# %% [markdown] slideshow={"slide_type": "subslide"}
# You can even downsample an image, and the zooms will reflect the increased distance between voxels.

# %%
cheap_downsample = cropped.slicer[2::4, 2::4, 2::4]
print(cheap_downsample.header.get_zooms())
cheap_downsample.orthoview()


# %% [markdown]
# Note that this is a bad idea in *most* circumstances because it induces aliasing.

# %% [markdown] slideshow={"slide_type": "subslide"}
# The better approach would be to anti-alias and then slice:

# %%
def blur(img, sigma):  # Isotropic in voxel space, not world space
    return img.__class__(ndi.gaussian_filter(img.dataobj, sigma), img.affine, img.header)

better_downsample = blur(cropped, sigma=1.5).slicer[2::4, 2::4, 2::4]
better_downsample.orthoview()

# %% [markdown] slideshow={"slide_type": "subslide"}
# For non-spatial dimensions, slices or indices may be used to select one or more volumes.

# %% slideshow={"slide_type": "-"}
tp15 = bold.slicer[..., :5]
tp1 = bold.slicer[..., 0]
print(f"BOLD shape: {bold.shape}; Zooms: {bold.header.get_zooms()}")
print(f"Time pts 1-5 shape: {tp15.shape}; Zooms: {tp15.header.get_zooms()}")
print(f"Time pt 1 shape: {tp1.shape}; Zooms: {tp1.header.get_zooms()}")
np.array_equal(tp15.get_fdata(), bold.dataobj[..., :5])

# %% [markdown]
# Aliasing considerations apply to time series as well, so be careful with down-sampling here, too.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## CIFTI-2
#
# <div style="float: right">
#     <div>
#         <img src="https://www.ncbi.nlm.nih.gov/pmc/articles/instance/6172654/bin/nihms-990058-f0001.jpg"><br/>
#         <span style="font-size: small">From Glasser, et al., 2016. doi:<a href="https://doi.org/10.1038/nn.4361">10.1038/nn.4361</a></span>
#     </div>
# </div>
#
# CIFTI-2 is a file format intended to cover many use cases for connectivity analysis.
#
# Files have 2-3 dimensions and each dimension is described by one of 5 types of axis.
#
# * Brain models: each row/column is a voxel or vertex
# * Parcels: each row/column is a group of voxels and/or vertices
# * Scalars: each row/column has a unique name
# * Labels: each row/column has a unique name and label table
# * Series: each row/column is a point in a series (typically time series), which increases monotonically
#
# For example, a "parcellated dense connectivity" CIFTI-2 file has two dimensions, indexed by a brain models axis and a parcels axis, respectively. The interpretation is "connectivity from parcels to vertices and/or voxels".

# %% [markdown] slideshow={"slide_type": "subslide"}
# <div style="float: right">
#     <img src="images/cifti-xml.png">
# </div>
#
# On disk, the file is a NIfTI-2 file with an alternative XML header as an extension, schematized here.
#
# NiBabel loads a header that closely mirrors this structure, and makes the NIfTI-2 header accessible as a `nifti_header` attribute.

# %%
cifti = nb.load('/data/out/ds000005-fmriprep/fmriprep/sub-01/func/sub-01_task-mixedgamblestask_run-1_space-fsLR_den-91k_bold.dtseries.nii')
cifti_data = cifti.get_fdata(dtype=np.float32)
cifti_hdr = cifti.header
nifti_hdr = cifti.nifti_header

# %% [markdown] slideshow={"slide_type": "fragment"}
# The `Cifti2Header` is useful if you're familiar with the XML structure and need to fetch an exact value or have fine control over the header that is written.

# %%
bm0 = next(cifti_hdr.matrix[1].brain_models)
print(bm0.voxel_indices_ijk)
print(list(bm0.vertex_indices)[:20])

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### The `Axis` API (`nibabel >=2.4`)
#
# Most of the time, the `Axis` format will be more useful:

# %%
axes = [cifti_hdr.get_axis(i) for i in range(cifti.ndim)]
axes

# %% slideshow={"slide_type": "subslide"}
t_axis, bm_axis = axes
print(f"{t_axis.size=}, {t_axis.start=}, {t_axis.step=}, {t_axis.unit}")
print(t_axis.time[:10])

# %%
pprint(list(bm_axis.iter_structures())[:5])
