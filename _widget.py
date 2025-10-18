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

from sklearn import preprocessing

from skimage import filters
from skimage import morphology
from skimage import measure
from skimage import restoration
from skimage import segmentation
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.transform import resize

# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvas


@magic_factory(call_button='Preprocess stack',)
def stack_preprocessing(viewer: Viewer, img:Image,
                        median_filter:bool=False, median_kernel:int=3,
                        background_substraction:bool=True,
                        drop_slices:bool=False,
                        slice_range:list=[0, 10]):
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
                  DAPI_intensity_filtering:int=15,
                  DAPI_volumetric_threshold:int=4,
                  cell_extension_footprint:int=0):
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

        if cell_extension_footprint != 0:
            fin_mask = morphology.dilation(fin_mask, footprint=morphology.disk(cell_extension_footprint))

        cells_labels, cells_num = ndi.label(fin_mask)
        show_info(f'{trans_img.name}: Detected {cells_num} cells')

        yield (cells_labels, f'{trans_img.name}_cell_mask')

    _cell_detector()


@magic_factory(call_button='Detect nucleus',)
def nucl_detector(viewer: Viewer, DRAQ_img:Image, DAPI_img:Image, cell_mask:Labels,
                  nucleus_filtering_footprint:int=0,
                  nucleus_extension_footprint:int=0):
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
        dapi_img = np.add(DAPI_img.data, DRAQ_img.data)
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

            if nucleus_filtering_footprint != 0 :
                nucl_mask = morphology.opening(nucl_mask,
                                            footprint=morphology.disk(nucleus_filtering_footprint))
            if nucleus_extension_footprint != 0:
                nucl_mask = morphology.dilation(nucl_mask, footprint=morphology.disk(nucleus_extension_footprint))

            nucl_labels, nucl_num = ndi.label(nucl_mask)
            show_info(f'{DAPI_img.name}: Detected {nucl_num} nucleus')

            yield (nucl_labels, f'{DAPI_img.name}_nucleus_mask')
        else:
            raise ValueError('Incorrect DAPI image shape!') 

    _nucl_detector()


@magic_factory(call_button='Segment compartments',)
def comp_detector(viewer: Viewer, DRAQ_img:Image, DAPI_img:Image, cell_mask:Labels, perfect_species:bool = False,
                  n_class:int=2):
    if input is not None:
        def _save_comp_masks(params):
            img_dapi_l = params[0]
            img_dapi_h = params[1]
            img_name = params[2]
            try: 
                viewer.layers[f'{img_name}_low'].data = img_dapi_l
            except KeyError:
                viewer.add_labels(img_dapi_l, name=f'{img_name}_low', opacity=0.5)
            try: 
                viewer.layers[f'{img_name}_high'].data = img_dapi_h
            except KeyError:
                viewer.add_labels(img_dapi_h, name=f'{img_name}_high', opacity=0.5)

        @thread_worker(connect={'yielded':_save_comp_masks})
        def _comp_detector():
            dapi_img_input = np.sum(DAPI_img.data, axis=0, dtype=np.float64)
            draq_img_input = np.sum(DRAQ_img.data, axis=0, dtype=np.float64)

            dapi_l_lab = np.zeros_like(dapi_img_input, dtype=bool)
            dapi_h_lab = np.zeros_like(dapi_img_input, dtype=bool)

            for cell_region in measure.regionprops(cell_mask.data):
                one_cell_box = cell_region.bbox
                dapi_img = dapi_img_input[one_cell_box[0]:one_cell_box[2],one_cell_box[1]:one_cell_box[3]]
                draq_img = draq_img_input[one_cell_box[0]:one_cell_box[2],one_cell_box[1]:one_cell_box[3]]
                one_cell_mask = cell_mask.data[one_cell_box[0]:one_cell_box[2],one_cell_box[1]:one_cell_box[3]] != 0

                dapi_img = dapi_img - np.mean(dapi_img, where=~one_cell_mask)  # background extraction
                dapi_img[~one_cell_mask] = np.mean(dapi_img, where=~one_cell_mask)  # extracellular px masking
                
                draq_img = draq_img - np.mean(draq_img, where=~one_cell_mask)  # background extraction
                draq_img[~one_cell_mask] = np.mean(draq_img, where=~one_cell_mask)  # extracellular px masking

                scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), clip=True)  # naive 0-1 scaler, sensitive to outliers
                # scaler = preprocessing.RobustScaler(with_centering=True, with_scaling=True, quantile_range=(2.0, 98.0))

                dapi_norm = scaler.fit_transform(dapi_img.reshape(-1,1)).reshape(dapi_img.shape)
                draq_norm = scaler.fit_transform(draq_img.reshape(-1,1)).reshape(draq_img.shape)   

                qd_rel = np.divide(draq_norm, dapi_norm, out=np.zeros_like(draq_norm), where=dapi_norm!=0)
                qd_rel = filters.median(qd_rel)
                qd = draq_img * qd_rel
                dq_rel = np.divide(dapi_norm, draq_norm, out=np.zeros_like(dapi_norm), where=draq_norm!=0)
                dq_rel = filters.median(dq_rel)
                dq = dapi_img * dq_rel

                qd_norm = scaler.fit_transform(qd.reshape(-1,1)).reshape(qd.shape)
                dq_norm = scaler.fit_transform(dq.reshape(-1,1)).reshape(dq.shape)

                # qd_mask = morphology.opening(qd_norm > filters.threshold_otsu(qd_norm))
                # dq_mask = morphology.opening(dq_norm > filters.threshold_otsu(dq_norm))
                qd_mask = qd_norm > np.max(filters.threshold_multiotsu(qd_norm,classes = n_class))
                dq_mask = dq_norm > np.max(filters.threshold_multiotsu(dq_norm,classes = n_class))

                q_mask = np.copy(qd_mask)
                q_mask[dq_mask] = False
                d_mask = np.copy(dq_mask)
                d_mask[qd_mask] = False
                # overlap_mask = (qd_mask & dq_mask)

                overlap_percent = np.sum((qd_mask & dq_mask)) / np.sum((qd_mask | dq_mask))
                #####for 13G


                show_info(f'Cell {cell_region.label}: compartments overlap {overlap_percent}%')

                dapi_l_lab[one_cell_box[0]:one_cell_box[2],one_cell_box[1]:one_cell_box[3]] = dq_mask
                dapi_h_lab[one_cell_box[0]:one_cell_box[2],one_cell_box[1]:one_cell_box[3]] = qd_mask


                if perfect_species :

                    ## from https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html#sphx-glr-auto-examples-segmentation-plot-watershed-py
                    in_img = qd_mask
                    distance = ndi.distance_transform_edt(in_img)

                    #in_img = in_img.astype(np.uint8)

                    coords = peak_local_max(distance, labels=in_img)

                    mask = np.zeros(distance.shape, dtype=bool)
                    mask[tuple(coords.T)] = True
                    markers, _ = ndi.label(mask)
                    labels = watershed(-distance, markers, mask=in_img)


                    for compartment in measure.regionprops(labels, intensity_image=dq_norm):

                        abs_max = np.max(compartment.intensity_image)
                        local_max = np.max(compartment.intensity_image[~compartment.image])

                        compartment_shape = compartment.image.shape
    
    # Extract the region of the dapi_l_lab or dapi_h_lab using the bounding box (one_cell_box)
                        region_l = dapi_l_lab[one_cell_box[0]:one_cell_box[2], one_cell_box[1]:one_cell_box[3]]
                        region_h = dapi_h_lab[one_cell_box[0]:one_cell_box[2], one_cell_box[1]:one_cell_box[3]]

    # Resize compartment.image to match the region size
                        compartment_resized = resize(compartment.image, (region_l.shape[0], region_l.shape[1]), mode='constant')

    # Compare abs_max and local_max, and assign to the appropriate label region
                        if abs_max > local_max:
                            region_l[:] = compartment_resized  # Assign the resized image to the region in dapi_l_lab
                        else:
                            region_h[:] = compartment_resized  # Assign the resized image to the region in dapi_h_lab


                 # overlap_mask = (qd_mask & dq_mask)

                    overlap_percent = np.sum((qd_mask & dq_mask)) / np.sum((qd_mask | dq_mask))
   

                #####for 13G





                #nucl_labels, nucl_num = ndi.label(nucl_mask)


            yield (dapi_l_lab, dapi_h_lab, f'{DAPI_img.name}_compartments_mask')

        _comp_detector()


@magic_factory(call_button='Mark species',)
def species_annotator(viewer: Viewer, base_img: Image, sp_labels:list=['Sros','Capsa','Yeast', 'Theca', 'Diplo', 'KIN', 'KIN']):
    COLOR_CYCLE = ['#FF0000',
                   '#008000',
                   '#0000FF',
                   '#FFFF00',
                   '#FF00FF']
    
    if len(sp_labels) > 20:
        raise ValueError('Too many species, 20 or less is recommended!')
    else:
        labels = sp_labels

    points_layer = viewer.add_points(name=f'{base_img.name}_sp_points',
                                     ndim=2,
                                     property_choices={'label': labels},
                                     edge_color='label',
                                     text={'string':'label', 'size':20, 'color':'black'},
                                     edge_color_cycle=COLOR_CYCLE,
                                     symbol='o',
                                     face_color='label',
                                     face_color_cycle=COLOR_CYCLE,
                                     edge_width=0.05,
                                     size=15)
    points_layer.edge_color_mode = 'cycle'
    points_layer.mode = 'add'

    def next_on_click(layer, event):
        if layer.mode == 'add':
            layer.selected_data = set()

    @viewer.bind_key('.', overwrite=True)
    def next_label(event):
        current_properties = points_layer.current_properties
        current_label = current_properties['label'][0]
        ind = list(labels).index(current_label)
        new_ind = (ind + 1) % len(labels)
        new_label = labels[new_ind]
        current_properties['label'] = np.array([new_label])
        points_layer.current_properties = current_properties
        points_layer.refresh_colors()
        points_layer.refresh_text()

    @viewer.bind_key(',', overwrite=True)
    def prev_label(event):
        current_properties = points_layer.current_properties
        current_label = current_properties['label'][0]
        ind = list(labels).index(current_label)
        n_labels = len(labels)
        new_ind = ((ind - 1) + n_labels) % n_labels
        new_label = labels[new_ind]
        current_properties['label'] = np.array([new_label])
        points_layer.current_properties = current_properties
        points_layer.refresh_colors()
        points_layer.refresh_text()

    points_layer.mouse_drag_callbacks.append(next_on_click)


@magic_factory(call_button='Save nucleus data',)
def save_nucl_df(nucleus_img:Image,
                      cell_mask: Labels, nucleus_mask:Labels,
                    #####For compartment measurements
                      kDNA_mask:Labels,
                      nuclDNA_mask:Labels,
                    #####For compartment measurements
                      sp_markers:Points,
                      saving_path:pathlib.Path = os.getcwd()):
    output_data_frame = pd.DataFrame({'id':[],
                                      'cell':[],
                                      'sp':[],
                                      'cell_coord':[],
                                      'nucl_sum_int':[],
                                      'cyto_mean_int':[],
                                      'nucl_sum_int_corr':[],
                                      #####For compartment measurements
                                      'one_kDNA_int':[],
                                      'one_nuclDNA_int':[],
                                      'one_kDNA_int_corr':[],
                                      'one_nuclDNA_int_corr':[]})

    sp_list = sp_markers.properties['label']
    sp_coord = sp_markers.data

    img_data = np.sum(nucleus_img.data, axis=0)
    c_mask = cell_mask.data
    n_mask = nucleus_mask.data != 0

    #####For compartment measurements
    kDNA_mask = kDNA_mask.data != 0
    nuclDNA_mask = nuclDNA_mask.data != 0
    #####For compartment measurements


    if c_mask.ndim != 2 or n_mask.ndim != 2:
        raise ValueError('Incorrect mask shape!')
    else:
        for cell_region in measure.regionprops(c_mask):
            one_cell_box = cell_region.bbox

            for sp_i in range(len(sp_list)):
                one_point_coord = sp_coord[sp_i]
                if one_cell_box[0] < one_point_coord[0] < one_cell_box[2] and one_cell_box[1] < one_point_coord[1] < one_cell_box[3]:
                    one_cell_sp = sp_list[sp_i]
                    one_cell_coord = str([int(one_point_coord[0]), int(one_point_coord[-1])])
                    print(sp_i, one_cell_sp)
                    break
                one_cell_sp = 'NA'
                one_cell_coord = 'NA'

            one_cell_mask = c_mask == cell_region.label

            one_nucl_mask = np.copy(n_mask)
            one_nucl_mask[~one_cell_mask] = 0
            one_nucl_int = np.sum(img_data, where=one_nucl_mask)

            one_cytoplasm_mask = np.copy(one_cell_mask)
            one_cytoplasm_mask[one_nucl_mask] = 0
            one_cytoplasm_int = np.mean(img_data, where=one_cytoplasm_mask, dtype=type(one_nucl_int))

            one_nucl_int_corr = one_nucl_int - one_cytoplasm_int

            #####For compartment measurements

            one_kDNA_mask = np.copy(kDNA_mask)
            one_kDNA_mask[~one_cell_mask] = 0
            one_kDNA_int = np.sum(img_data, where=one_kDNA_mask)
            one_kDNA_int_corr = one_kDNA_int - one_cytoplasm_int

            one_nuclDNA_mask = np.copy(nuclDNA_mask)
            one_nuclDNA_mask[~one_cell_mask] = 0
            one_nuclDNA_int = np.sum(img_data, where=one_nuclDNA_mask)
            one_nuclDNA_int_corr = one_nuclDNA_int - one_cytoplasm_int

            #####For compartment measurements

            cell_row = [nucleus_img.name, cell_region.label, one_cell_sp, one_cell_coord, one_nucl_int, one_cytoplasm_int, one_nucl_int_corr, one_kDNA_int, one_nuclDNA_int, one_kDNA_int_corr, one_nuclDNA_int_corr]
            output_data_frame.loc[len(output_data_frame.index)] = cell_row

        output_data_frame.to_csv(os.path.join(saving_path, f'{nucleus_img.name}_nucl_df.csv'))
        show_info(f'{nucleus_img.name}: nucleus int data frame saved')

@magic_factory(call_button='Save kDNA data',)
def save_kDNA_df(nucleus_img:Image,
                      cell_mask: Labels, nucleus_mask:Labels, kDNA_mask:Labels,
                    #####For compartment measurements
                      
                      #nuclDNA_mask:Labels,
                    #####For compartment measurements
                      sp_markers:Points,
                      saving_path:pathlib.Path = os.getcwd()):
    output_data_frame = pd.DataFrame({'id':[],
                                      'cell':[],
                                      'sp':[],
                                      'cell_coord':[],
                                      'cyto_mean_int':[],
                                    
                                      #####For compartment measurements
                                      'one_kDNA_int':[],
                                      #'one_nuclDNA_int':[],
                                      'one_kDNA_int_corr':[]#,
                                      #'one_nuclDNA_int_corr':[]
                                      })

    sp_list = sp_markers.properties['label']
    sp_coord = sp_markers.data

    img_data = np.sum(nucleus_img.data, axis=0)
    c_mask = cell_mask.data
    n_mask = nucleus_mask.data != 0


    #####For compartment measurements
    kDNA_mask = kDNA_mask.data != 0

    #####For compartment measurements


    if c_mask.ndim != 2 or kDNA_mask.ndim != 2:
        raise ValueError('Incorrect mask shape!')
    else:
        for cell_region in measure.regionprops(c_mask):
            one_cell_box = cell_region.bbox

            for sp_i in range(len(sp_list)):
                if 'KIN' in sp_list[sp_i] :
                    one_point_coord = sp_coord[sp_i]
                    if one_cell_box[0] < one_point_coord[0] < one_cell_box[2] and one_cell_box[1] < one_point_coord[1] < one_cell_box[3]:
                        one_cell_sp = sp_list[sp_i]
                        one_cell_coord = str([int(one_point_coord[0]), int(one_point_coord[-1])])
                        print(sp_i, one_cell_sp)
                        break
                    one_cell_sp = 'NA'
                    one_cell_coord = 'NA'
                #else : next

            one_cell_mask = c_mask == cell_region.label

            one_nucl_mask = np.copy(n_mask)
            one_nucl_mask[~one_cell_mask] = 0
            one_nucl_int = np.sum(img_data, where=one_nucl_mask)


            one_cytoplasm_mask = np.copy(one_cell_mask)
            one_cytoplasm_mask[one_nucl_mask] = 0
            one_cytoplasm_int = np.mean(img_data, where=one_cytoplasm_mask, dtype=type(one_nucl_int))

            #####For compartment measurements

            one_kDNA_mask = np.copy(kDNA_mask)
            one_kDNA_mask[~one_cell_mask] = 0
            one_kDNA_int = np.sum(img_data, where=one_kDNA_mask)
            one_kDNA_int_corr = one_kDNA_int - one_cytoplasm_int


            #####For compartment measurements

            cell_row = [nucleus_img.name, cell_region.label, one_cell_sp, one_cell_coord, one_cytoplasm_int, one_kDNA_int, one_kDNA_int_corr]
            output_data_frame.loc[len(output_data_frame.index)] = cell_row

        output_data_frame.to_csv(os.path.join(saving_path, f'{nucleus_img.name}_kDNA_df.csv'))
        show_info(f'{nucleus_img.name}: kDNA int data frame saved')


@magic_factory(call_button='Save kinet nuclDNA data',)
def save_KINnuclDNA_df(nucleus_img:Image,
                      cell_mask: Labels, nucleus_mask:Labels,
                    #####For compartment measurements
                      
                      nuclDNA_mask:Labels,
                    #####For compartment measurements
                      sp_markers:Points,
                      saving_path:pathlib.Path = os.getcwd()):
    output_data_frame = pd.DataFrame({'id':[],
                                      'cell':[],
                                      'sp':[],
                                      'cell_coord':[],
                                      'cyto_mean_int':[],
                                    
                                      #####For compartment measurements
                                      
                                      'one_nuclDNA_int':[],
                                      
                                      'one_nuclDNA_int_corr':[]
                                      })

    sp_list = sp_markers.properties['label']
    sp_coord = sp_markers.data

    img_data = np.sum(nucleus_img.data, axis=0)
    c_mask = cell_mask.data
    n_mask = nucleus_mask.data != 0



    #####For compartment measurements
    nuclDNA_mask = nuclDNA_mask.data != 0

    #####For compartment measurements


    if c_mask.ndim != 2 or nuclDNA_mask.ndim != 2:
        raise ValueError('Incorrect mask shape!')
    else:
        for cell_region in measure.regionprops(c_mask):
            one_cell_box = cell_region.bbox

            for sp_i in range(len(sp_list)):
                if 'KIN' in sp_list[sp_i] :
                    one_point_coord = sp_coord[sp_i]
                    if one_cell_box[0] < one_point_coord[0] < one_cell_box[2] and one_cell_box[1] < one_point_coord[1] < one_cell_box[3]:
                        one_cell_sp = sp_list[sp_i]
                        one_cell_coord = str([int(one_point_coord[0]), int(one_point_coord[-1])])
                        print(sp_i, one_cell_sp)
                        break
                    one_cell_sp = 'NA'
                    one_cell_coord = 'NA'
                #else : next

            one_cell_mask = c_mask == cell_region.label

            one_nucl_mask = np.copy(n_mask)
            one_nucl_mask[~one_cell_mask] = 0
            one_nucl_int = np.sum(img_data, where=one_nucl_mask)

            one_cytoplasm_mask = np.copy(one_cell_mask)
            one_cytoplasm_mask[one_nucl_mask] = 0
            one_cytoplasm_int = np.mean(img_data, where=one_cytoplasm_mask, dtype=type(one_nucl_int))

            #####For compartment measurements

            one_nuclDNA_mask = np.copy(nuclDNA_mask)
            one_nuclDNA_mask[~one_cell_mask] = 0
            one_nuclDNA_int = np.sum(img_data, where=one_nuclDNA_mask)
            one_nuclDNA_int_corr = one_nuclDNA_int - one_cytoplasm_int


            #####For compartment measurements

            cell_row = [nucleus_img.name, cell_region.label, one_cell_sp, one_cell_coord, one_cytoplasm_int, one_nuclDNA_int, one_nuclDNA_int_corr]
            output_data_frame.loc[len(output_data_frame.index)] = cell_row

        output_data_frame.to_csv(os.path.join(saving_path, f'{nucleus_img.name}_KIN_nuclDNA_df.csv'))
        show_info(f'{nucleus_img.name}: KIN_nuclDNA int data frame saved')



@magic_factory(call_button='Print fetures',)
def print_annotator(viewer: Viewer, point:Points):
    print(point.data)
    print(point.features)
