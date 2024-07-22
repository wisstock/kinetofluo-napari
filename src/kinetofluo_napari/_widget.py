import napari
from napari import Viewer
from napari.layers import Image, Labels, Points
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker

from magicgui import magic_factory
from magicgui.widgets import ComboBox, Container

import os
import pathlib
import datetime

import numpy as np
from numpy import ma

import pandas as pd

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
                        background_substraction:bool=True,
                        drop_slices:bool=True,
                        slice_range:list=[1, 6]):
    if input is not None:
        def _save_ch(params):
            img = params[0]
            img_name = params[1]
            if img_name.split('_')[-1] == 'ch0':
                colormap_name = 'gray'
            else:
                colormap_name = 'turbo'
            try: 
                viewer.layers[img_name].data = img
            except KeyError:
                viewer.add_image(img, name=img_name, colormap=colormap_name)

        @thread_worker(connect={'yielded':_save_ch})
        def _split_channels():
            def _preprocessing(ch_img, ch_suffix):
                if drop_slices:
                    ch_img = ch_img[slice_range[0]:slice_range[-1],...]
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


@magic_factory(call_button='Detect cells',
               detection_method={"choices": ['volumetric', 'intensity']},)
def cell_detector(viewer: Viewer, trans_img:Image, DAPI_img:Image,
                  detection_method:str='intensity',
                  trans_filtering_kernel:int=5,
                  DAPI_intensity_filtering:int=10,
                  DAPI_volumetric_threshold:int=4):
    if input is not None:
        def _save_mask(params):
            img = params[0]
            img_name = params[1]
            try: 
                viewer.layers[img_name].data = img
            except KeyError:
                viewer.add_labels(img, name=img_name, opacity=0.5)
    
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

        if detection_method == 'intensity':
            # intensity debris filtering
            ctrl_fluo_img = np.sum(DAPI_img.data, axis=0, dtype=np.uint32)
            ctrl_fluo_mask = ctrl_fluo_img > filters.threshold_otsu(ctrl_fluo_img)
            ctrl_fluo_mask = morphology.opening(ctrl_fluo_mask,
                                                footprint=morphology.disk(DAPI_intensity_filtering))
        elif detection_method == 'volumetric':
            # volumetric debris filtering
            ctrl_fluo_volumetric_mask = DAPI_img.data > filters.threshold_otsu(DAPI_img.data)
            # ctrl_fluo_volumetric_mask = morphology.opening(ctrl_fluo_volumetric_mask,
            #                                                footprint=morphology.disk(2)) 
            ctrl_fluo_volumetric_img = np.sum(ctrl_fluo_volumetric_mask, axis=0)
            ctrl_fluo_mask = ctrl_fluo_volumetric_img > DAPI_volumetric_threshold
            ctrl_fluo_mask = morphology.dilation(ctrl_fluo_mask,
                                                 footprint=morphology.disk(2)) 

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


@magic_factory(call_button='Detect nucleus',)
def nucl_detector(viewer: Viewer, DAPI_img:Image, cell_mask:Labels,
                  nucleus_filtering_footprint:int=1):
    if input is not None:
        def _save_mask(params):
            img = params[0]
            img_name = params[1]
            try: 
                viewer.layers[img_name].data = img
            except KeyError:
                viewer.add_labels(img, name=img_name, opacity=0.5)
    
    @thread_worker(connect={'yielded':_save_mask})
    def _nucl_detector():
        dapi_img = DAPI_img.data
        if dapi_img.ndim == 3:
            filtering_mask = cell_mask.data != 0

            # volumetric mask
            dapi_img_pre_masked = ma.masked_where(np.broadcast_to(filtering_mask, dapi_img.shape), dapi_img)
            dapi_3d_vol_mask = dapi_img > filters.threshold_otsu(dapi_img_pre_masked.compressed())
            dapi_vol_img = np.sum(dapi_3d_vol_mask, axis=0)
            
            # signal density
            dapi_sum = np.sum(dapi_img, axis=0, dtype=np.uint32)
            dapi_int_to_vol = (dapi_sum / dapi_vol_img) * dapi_sum
            dapi_int_to_vol[dapi_int_to_vol == np.inf] = 0

            # masking loop
            nucl_mask = np.zeros_like(dapi_sum)
            print(cell_mask.data.shape)
            print(nucl_mask.shape)
            for cell_region in measure.regionprops(cell_mask.data):
                one_cell_box = cell_region.bbox
                one_cell_int_to_vol = dapi_int_to_vol[one_cell_box[0]:one_cell_box[2],one_cell_box[1]:one_cell_box[3]]
                one_cell_mask = one_cell_int_to_vol > filters.threshold_otsu(one_cell_int_to_vol)
                nucl_mask[one_cell_box[0]:one_cell_box[2],one_cell_box[1]:one_cell_box[3]] = one_cell_mask

            nucl_mask = morphology.opening(nucl_mask,
                                           footprint=morphology.disk(nucleus_filtering_footprint))
            nucl_labels, nucl_num = ndi.label(nucl_mask)
            show_info(f'{DAPI_img.name}: Detected {nucl_num} nucleus')

            yield (nucl_labels, f'{DAPI_img.name}_nucleus_mask')
        else:
            raise ValueError('Incorrect DAPI image shape!') 

    _nucl_detector()


@magic_factory(call_button='Mark species',)
def species_annotator(viewer: Viewer, base_img: Image):
    COLOR_CYCLE = ['#FF0000',
                   '#008000',
                   '#0000FF',
                   '#FFFF00',
                   '#FF00FF']
    
    labels = ['R','G','B','Y','F']

    points_layer = viewer.add_points(name=f'{base_img.name}_sp_points',
        ndim=2,
        property_choices={'label': labels},
        edge_color='label',
        edge_color_cycle=COLOR_CYCLE,
        symbol='o',
        face_color='label',
        face_color_cycle=COLOR_CYCLE,
        edge_width=0.05,  # fraction of point size
        size=15,
    )
    points_layer.edge_color_mode = 'cycle'
    points_layer.mode = 'add'

    def next_on_click(layer, event):
        """Mouse click binding to advance the label when a point is added"""
        if layer.mode == 'add':
            # By default, napari selects the point that was just added.
            # Disable that behavior, as the highlight gets in the way
            # and also causes next_label to change the color of the
            # point that was just added.
            layer.selected_data = set()

    @viewer.bind_key('.')
    def next_label(event):
        """Keybinding to advance to the next label with wraparound"""
        current_properties = points_layer.current_properties
        current_label = current_properties['label'][0]
        ind = list(labels).index(current_label)
        new_ind = (ind + 1) % len(labels)
        new_label = labels[new_ind]
        current_properties['label'] = np.array([new_label])
        points_layer.current_properties = current_properties
        points_layer.refresh_colors()

    @viewer.bind_key(',')
    def prev_label(event):
        """Keybinding to decrement to the previous label with wraparound"""
        current_properties = points_layer.current_properties
        current_label = current_properties['label'][0]
        ind = list(labels).index(current_label)
        n_labels = len(labels)
        new_ind = ((ind - 1) + n_labels) % n_labels
        new_label = labels[new_ind]
        current_properties['label'] = np.array([new_label])
        points_layer.current_properties = current_properties
        points_layer.refresh_colors()

    # points_layer.border_color_mode = 'cycle'
    points_layer.mouse_drag_callbacks.append(next_on_click)


@magic_factory(call_button='Save nucleus data',)
def save_nucl_df(nucleus_img:Image,
                      cell_mask: Labels, nucleus_mask:Labels,
                      sp_markers:Points,
                      saving_path:pathlib.Path = os.getcwd()):
    output_data_frame = pd.DataFrame({'id':[],
                                      'cell':[],
                                      'sp':[],
                                      'int':[]})

    sp_list = sp_markers.properties['label']
    sp_coord = sp_markers.data

    img_data = np.sum(nucleus_img.data, axis=0)
    c_mask = cell_mask.data
    n_mask = nucleus_mask.data != 0

    for cell_region in measure.regionprops(c_mask):
        one_cell_box = cell_region.bbox

        for sp_i in range(len(sp_list)):
            one_point_coord = sp_coord[sp_i]
            if one_cell_box[0] < one_point_coord[0] < one_cell_box[2] and one_cell_box[1] < one_point_coord[1] < one_cell_box[3]:
                one_cell_sp = sp_list[sp_i]
            else:
                one_cell_sp = 'NA'

        one_cell_mask = c_mask == cell_region.label
        one_nucl_mask = np.copy(n_mask)
        one_nucl_mask[~one_cell_mask] = 0
        one_cell_int = np.sum(img_data, where=one_nucl_mask)

        cell_row = [nucleus_img.name, cell_region.label, one_cell_sp, one_cell_int]
        output_data_frame.loc[len(output_data_frame.index)] = cell_row

        output_data_frame.to_csv(os.path.join(saving_path, f'{nucleus_img.name}_nucl_df.csv'))


@magic_factory(call_button='Print fetures',)
def print_annotator(viewer: Viewer, point:Points):
    print(point.properties)
    print(point.data)
    print(point.features)
