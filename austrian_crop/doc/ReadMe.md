Please download the data here: https://drive.google.com/drive/folders/1Yg4TuD22LJl2Gt6AOswG1WTmpVhK55hY?usp=sharing

And then put：
austrian_crop_gse_embedding_downsample_100.npy
austrian_crop_presto_embeddings_downsample_100.npy
austrian_crop_tessera_embedding_downsample_100.npy
bands_downsample_100.npy
fieldid_downsample_100.npy
fieldtype_17classes_downsample_100.npy
sar_ascending_downsample_100.npy
sar_descending_downsample_100.npy
updated_fielddata.csv

in austrian_crop/data

then do: `cd austrian_crop/`

finally you can run: `python src/pixel_wise_fieldid_eval_FM.py`