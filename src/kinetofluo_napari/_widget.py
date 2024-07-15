import napari
from napari import Viewer
from napari.layers import Image, Labels
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker

from magicgui import magic_factory

import os
import pathlib
import datetime

import numpy as np

from scipy import ndimage as ndi
from scipy import stats
from scipy import signal

from skimage import filters
from skimage import morphology
from skimage import measure
from skimage import restoration
from skimage import segmentation

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas


@magic_factory(call_button='Preprocess stack',)
def stack_preprocessing(viewer: Viewer, img:Image,
                        median_filter:bool=False, median_kernel:int=3,  #gaussian_blur:bool=True, gaussian_sigma=0.75,
                        background_substraction:bool=True):
    if input is not None:
        def _save_ch(params):
            img = params[0]
            img_name = params[1]
            try: 
                viewer.layers[img_name].data = img
            except KeyError:
                viewer.add_image(img, name=img_name, colormap='turbo')

        @thread_worker(connect={'yielded':_save_ch})
        def _split_channels():
            def _preprocessing(ch_img, ch_suffix):
                if median_filter:
                    median_axis = lambda x,k: np.array([ndi.median_filter(f, size=k) for f in x], dtype=x.dtype)
                    ch_img = median_axis(ch_img, median_kernel)
                if background_substraction:
                    bc_p = lambda x: np.array([f - np.percentile(f, 0.5) for f in x]).clip(min=0).astype(dtype=x.dtype)
                    ch_img = bc_p(ch_img)
                return (ch_img, img.name+ch_suffix)

            if img.data.ndim == 4:
                show_info(f'{img.name}: Ch. split and preprocessing mode, shape {img.data.shape}')
                for i in range(0,img.data.shape[1]):
                    show_info(f'{img.name}: Ch. {i} preprocessing')
                    if i == 0:
                        yield (img.data[:,i,...], f'{img.name}_ch{i}')
                    else:
                        yield _preprocessing(ch_img=img.data[:,i,...], ch_suffix=f'_ch{i}')
            elif img.data.ndim == 3:
                show_info(f'{img.name}: Image already has 3 dimensions, preprocessing only mode')
                yield _preprocessing(ch_img=img.data, ch_suffix=f'_ch0')
            else:
                raise ValueError('Input image should have 3 or 4 dimensions!')       
        
        _split_channels()


@magic_factory(call_button='Detect cells',)
def cell_detector(viewer: Viewer, trans_img:Image, fluo_img:Image,
                  trans_filtering_kernel:int=4,
                  fluo_filtering_kernel:int=5):
    if input is not None:
        def _save_mask(params):
            img = params[0]
            img_name = params[1]
            try: 
                viewer.layers[img_name].data = img
            except KeyError:
                viewer.add_labels(img, name=img_name, opacity=0.75)
    
    @thread_worker(connect={'yielded':_save_mask})
    def _cell_detector():

        img_trans_filt = filters.gaussian(np.sum(trans_img.data, axis=0, dtype=np.uint32), sigma=2)

        img_trans_filt = img_trans_filt / np.max(np.abs(img_trans_filt))
        img_trans_filt = filters.rank.gradient(img_trans_filt, morphology.disk(3))

        mask = img_trans_filt > filters.threshold_otsu(img_trans_filt)
        mask = ndi.binary_fill_holes(mask)
        mask = segmentation.clear_border(mask)

        mask = morphology.erosion(mask, footprint=morphology.disk(trans_filtering_kernel+1))
        mask = morphology.dilation(mask, footprint=morphology.disk(trans_filtering_kernel))

        labels, labels_num = ndi.label(mask)

        # debris filtering
        ctrl_fluo_img = np.sum(fluo_img.data, axis=0, dtype=np.uint32)
        ctrl_fluo_mask = ctrl_fluo_img > filters.threshold_otsu(ctrl_fluo_img)
        ctrl_fluo_mask = morphology.opening(ctrl_fluo_mask,
                                            footprint=morphology.disk(fluo_filtering_kernel))

        sums = ndi.sum(ctrl_fluo_mask, labels, np.arange(labels_num+1))
        connected = sums > 0
        debris_mask = connected[labels]

        # final mask filtering
        fin_mask = np.copy(mask)
        fin_mask[~debris_mask] = 0  # debris rejection
        fin_mask[ctrl_fluo_mask] = 1  # holes filling with DAPI+DRAQ mask
        fin_mask = segmentation.clear_border(fin_mask)  # borders cleaning
        fin_mask = morphology.opening(fin_mask, footprint=morphology.disk(5))  # rejection of DAPI+DRAW mask artifacts
        fin_mask = ndi.binary_fill_holes(fin_mask)

        cells_labels, cells_num = ndi.label(fin_mask)
        show_info(f'{trans_img.name}: Detected {cells_num} cells')

        yield (cells_labels, f'{trans_img.name}_cell_mask')

    _cell_detector()

