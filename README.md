# StackMFF
End-to-end Multi-Focus Image Stack Fusion Network

## Prepare datasets for evaluation
We have prepared all the evaluation datasets for you, which can be downloaded [here](https://pan.baidu.com/s/1n68SffCOg5RpzRgCIuuy4g?pwd=cite).
Put Datasets_StackMFF to data/Datasets_StackMFF
If you want to make your own evaluation dataset, please refer to the following:

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


## Inference
### If you want to inference datasets, run:
```
python predict_dataset.py --model_path checkpoint/checkpoint.pth --stack_basedir_path data/Datasets_StackMFF/4D-Light-Field/image stack
```
### If you want to inference a image stack, run:
```
python predict.py --model_path checkpoint/checkpoint.pth --stack_path data/Datasets_StackMFF/4D-Light-Field/image stack/boxes
```

## Train
### 1. Download the validation set of the original dataset [Open Images V7](https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer) used to make the training dataset, and put all images to data/OpenImagesV7.
### 2. Split the validation set of the original dataset [Open Images V7](https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer) into 2 training sets and validation sets by running the following command:
```
python split_dataset.py
```
### 3. Using [Metric3D](https://github.com/YvanYin/Metric3D) to get depth maps (8bit, range from 0 to 255) for all images, and put all depth maps to data/OpenImagesV7/train_depth and data/OpenImagesV7/val_depth, respectively.
### 4. Using depth-adapted multi-Focus simulation (DAMS) to get multi-focus image stacks, run:
```
python make_dataset.py
```

### 5. Train StackMFF
The training code will be released after the article is accepted.
