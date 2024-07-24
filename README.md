# StackMFF
End-to-end Multi-Focus Image Stack Fusion Network

## Prepare datasets
### 1. 4D Light Field
Download [4D-Light-Field](https://lightfield-analysis.uni-konstanz.de/) dataset
Put full_data.zip under ./data/4D-Light-Field_Gen
Run the following command under ./data/4D-Light-Field_Gen

```
unzip full_data.zip
python LF2hdf5.py --base_dir ./full_data --output_dir ./LF
python FS_gen.py --LF_path ./LF/HCI_LF_trainval.h5 --output_dir ./FS
python save_AiF.py ./FS/HCI_FS_trainval.h5 ./FS
python save_stack.py ./FS/HCI_FS_trainval.h5 ./FS
```

### 2. FlyingThings3D Dataset
Download [FlyingThings3D_FS](https://drive.google.com/file/d/19n3QGhg-IViwt0aqQ4rR8J3sO60PoWgL/view?usp=sharing) under ./data/FlyingThings3D/
Unzip the dataset

### 3. Middlebury Dataset

Download [Middlebury_FS](https://drive.google.com/file/d/1FDXf47Qp1-dT_C7bo30ZySvvPAgJf5FU/view?usp=sharing) under ./data/Middlebury/
Unzip the dataaset
Mobile Depth Dataset

### 4. Mobile Depth Dataset
Download [Mobile Depth](https://www.supasorn.com/dffdownload.html) dataset under ./data/Mobile_Depth_Gen
Run the following command under ./data/Mobile_Depth_Gen
```
mkdir Photos_Calibration_Results
mv depth_from_focus_data2.zip Photos_Calibration_Results
cd Photos_Calibration_Results
unzip ./depth_from_focus_data2.zip
mv calibration/metal calibration/metals
mv calibration/GT calibration/zeromotion
mv calibration/GTSmall calibration/smallmotion
mv calibration/GTLarge calibration/largemotion
cd ..
unzip depth_from_focus_data3.zip
```
