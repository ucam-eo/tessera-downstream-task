
# DOWNSTREAM TASK: BIOMASSTERS

This folder contains code for training a UNet regression model on TESSERA and AlphaEarth embeddings. 

## UNet model on TESSERA embeddings

The UNet regressor can be trained on the TESSERA embeddings by running ***unet_regressor.py*** with the option **efm=False**. Edit the path variables near the top of ***unet_regressor.py*** so that **root_dir** and **ground_truth_dir** point to correct folders. The code expects the following file structure:

```text
<root_dir>/
└── tessera/
    └── representations/
        ├── representations_train_all/
        ├── representations_test/
        ├── scales_train_all/
        └── scales_test/

<ground_truth_dir>/
├── train_agbm_all/
└── test_agbm/
```


The ground truth files are expected to be in **.npy** format. Code for converting from **.tif** to **.npy** is provided in the file ***run_converting_from_tif_to_npy.py***. A function for clipping the ground truth data to a threshold can be found in the file ***clipping.py***.


## Obtaining AlphaEarth embeddings

The AlphaEarth representations can be obtained by running ***alphaearth_preprocessing.ipynb*** for the chosen ROI and year. The code assumes the ROIs to be saved to Google Cloud Storage. 

The regressor expects the files to be under **root_dir/efm/data/** in subfolders **train_representations_all** and **test_representations** in **.npy** format. 

After this, the UNet model can be trained by running ***unet_regressor.py*** with the option **efm=True**.

## Plotting

The code for plotting the figures in the TESSERA paper can be found in ***plotting.ipynb*** and the precalculated data for the figures under **data/** directory.
