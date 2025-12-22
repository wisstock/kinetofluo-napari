# kinetofluo-napari

`kinetofluo-napari` is a [napari](https://napari.org/) plugin designed to semi-automatically detect and analyze intracellular DNA-containing compartments. 

## Data Analysis Pipeline Overview

The standard analysis workflow with `kinetofluo-napari` typically consists of the following steps:

1.  **Preprocessing:** Multi-channel stacks are split into individual layers, filtered, and background subtraction is applied.
2.  **Cell detection:** Hybrid segmentation using fluorescence and brightfield channel images.
3.  **Nucleus detection:** Segmentation of the nuclei of the control species within identified cell boundaries using intensity-to-volume ratios.
4.  **Compartment segmentation:** Segmentation of DNA-rich compartments in the species of interest within identified cell boundaries based on A-content of the DNA.
5.  **Species annotation:** Manual labeling of cells for classification.
6.  **Data export:** Calculation of intensity metrics with local background correction and export to tidy data frame in CSV format.

---

## Widget and Parameter Description

### 1. Preprocess stack
Prepares raw microscope data for analysis by splitting channels and applying filters.

- **img**: The input image layer. Supports 3D (Z-stack) or 4D (Time/Z-stack) data.
- **median_filter** (bool): If enabled, applies a median filter to each slice.
- **median_kernel** (int): The size of the neighborhood for the median filter. A larger kernel removes more noise but may blur fine details. Default is 3.
- **background_substraction** (bool): When active, the widget estimates background intensity by calculating the 0.5th percentile of intensity for each slice and subtracts it.
- **drop_slices** (bool): Enables Z-depth cropping. Useful if the top or bottom of the stack contains only out-of-focus blur.
- **slice_range** (list): A list of two integers `[start, end]` defining the Z-range to keep (e.g., `[0, 10]`).

### 2. Detect cells
Identifies individual cell boundaries using a combination of brightfield/transmission and fluorescence signals.

- **trans_img / DAPI_img**: The respective image layers for brightfield (e.g., DIC, Phase Contrast) and fluorescence (DNA marker).
- **detection_method**:
    - `intensity`: Uses a 2D projection of the fluorescence channel for debris filtering.
    - `volumetric`: Analyzes the 3D connectivity of the fluorescence signal. More robust against small fluorescent artifacts.
- **trans_filtering_kernel** (int): Controls the radius of morphological erosion and dilation applied to the brightfield image mask. Increasing this helps smooth irregular cell boundaries and merge fragmented detections.
- **DAPI_intensity_filtering** (int): Radius for the morphological opening on the fluorescence mask. Used to eliminate small, bright debris (bacteria).
- **DAPI_volumetric_threshold** (int): Specifically for the `volumetric` method. Defines the minimum number of Z-slices a fluorescent object must span to be considered as the cell.
- **cell_extension_footprint** (int): A final dilation radius applied to the cell mask.

### 3. Detect nucleus
Performs nucleus segmentation restricted to the areas identified as cells. Useful for nucleus detection of the control species. This widget calculates an intensity to volume ratio: `(sum_intensity / volume) * sum_intensity`. This amplification factor helps distinguish true nuclei from low-level background signal.

- **DRAQ_img / DAPI_img**: Usually the same channels used for control species cell detection or specific high-quality nuclear stains.
- **cell_mask**: The labels layer representing individual cells. This acts as a spatial constraint—the detector only looks for nuclei inside these regions.
- **nucleus_filtering_footprint** (int): Radius for morphological opening. Removes small nuclear fragments or micronuclei that should be ignored in the primary count.
- **nucleus_extension_footprint** (int): Dilation radius for the final nucleus mask. Important for ensuring the mask covers the entire high-intensity region.

### 4. Segment compartments
Multi-class segmentation for identifying zones of different DNA density within the nucleus (e.g., separating kDNA from nuclear DNA). The plugin applies `MinMaxScaler` per cell to normalize intensities, then uses multi-Otsu threshold to find optimal thresholds for the specified number of classes. It also calculates compartment overlap percentage which is displayed in the info bar.

- **DRAQ_img / DAPI_img**: Fluorescent channels representing DNA.
- **cell_mask**: Used for per-cell normalization.
- **perfect_species** (bool): Enables an advanced watershed-based refinement step to split overlapped compartments.
- **n_class** (int): The number of classes for the Multi-Otsu thresholding algorithm. 
    - At `n_class=2`, it separates background from foreground DNA.
    - Increasing this allows for distinguishing between low-density and high-density regions.

### 5. Mark species
Interactive tool for manual classification of cells. Automatically assigns unique colors from a predefined cycle to each species label.

- **base_img**: The background image used for visual context.
- **sp_labels** (list): A list of category names. Defaults include `['Sros','Capsa','Yeast', 'Theca', 'Diplo', 'KIN']`.
- **Interactive Controls**:
    - **Add Mode**: Direct clicking on the image adds a point with the currently selected label.
    - **Label Switching**:
        - Use the `.` (period) key to cycle to the next label in the list.
        - Use the `,` (comma) key to cycle to the previous label.

### 6. Save nucleus data
Extracts quantitative data and saves it for statistical analysis.

- **nucleus_img**: The channel from which intensity measurements are taken.
- **cell_mask / nucleus_mask**: The definitive segmentation results.
- **kDNA_mask / nuclDNA_mask**: (Optional) Masks for specific DNA compartments (e.g., kinetoplast or nuclear DNA) to include in the metrics.
- **sp_markers**: The points layer containing manual annotations.
- **saving_path**: Directory where the resulting CSV will be stored.
- **CSV Column Descriptions**:
    - *id*: Name of the nucleus layer used for measurements.
    - *cell*: Cell identifier (label) from the segmentation mask.
    - *sp*: Species name, derived from the `sp_markers` annotation layer by finding the point within the cell's bounding box.
    - *cell_coord*: Coordinates of the species annotation point.
    - *nucl_sum_int*: Sum intensity of the signal within the nucleus mask (sum projection along Z-axis).
    - *cyto_mean_int*: Mean intensity of the signal in the cytoplasm (cell area minus nucleus area).
    - *nucl_sum_int_corr*: Corrected nucleus intensity, calculated as the difference between sum nucleus intensity and mean cytoplasm intensity (`nucl_sum_int - cyto_mean_int`).
    - *one_kDNA_int*: Sum intensity of the signal within the kDNA (kinetoplast) mask.
    - *one_nuclDNA_int*: Sum intensity of the signal within the nuclear DNA mask.
    - *one_kDNA_int_corr*: Corrected kDNA intensity (`one_kDNA_int - cyto_mean_int`).
    - *one_nuclDNA_int_corr*: Corrected nuclear DNA intensity (`one_nuclDNA_int - cyto_mean_int`).

### 7. Save kDNA data
Specialized export tool for kinetoplast DNA (kDNA).

- **Input**: Requires `nucleus_img`, `cell_mask`, `nucleus_mask`, and `kDNA_mask`.
- **Nuance**: This widget specifically filters for cells annotated with "KIN" (part of the label string) in the species markers. Cells without this tag are essentially ignored or handled differently in the iteration.
- **CSV Column Descriptions**:
    - *id*: Name of the nucleus layer.
    - *cell*: Cell identifier.
    - *sp*: Species name (only those containing "KIN" are filtered).
    - *cell_coord*: Coordinates of the annotation point.
    - *cyto_mean_int*: Mean intensity of the signal in the cytoplasm.
    - *one_kDNA_int*: Sum intensity of the signal within the kDNA mask.
    - *one_kDNA_int_corr*: Corrected kDNA intensity (`one_kDNA_int - cyto_mean_int`).

### 8. Save kinet nuclDNA data
Specialized export tool for nuclear DNA of the kinetoplast species. Similar to the kDNA tool, this tool specifically targets cells annotated as `KIN`.

- **Input**: Requires `nuclDNA_mask` in addition to standard images and cell masks.
- **CSV Column Descriptions**:
    - *id*: Name of the nucleus layer.
    - *cell*: Cell identifier.
    - *sp*: Species name (only those containing "KIN" are filtered).
    - *cell_coord*: Coordinates of the annotation point.
    - *cyto_mean_int*: Mean intensity of the signal in the cytoplasm.
    - *one_nuclDNA_int*: Sum intensity of the signal within the nuclear DNA mask.
    - *one_nuclDNA_int_corr*: Corrected nuclear DNA intensity (`one_nuclDNA_int - cyto_mean_int`).