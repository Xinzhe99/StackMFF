# StackMFF
End-to-end Multi-Focus Image Stack Fusion Network

## Prepare datasets
### 4D Light Field
Go to this [website](https://lightfield-analysis.uni-konstanz.de/) to request for the 4D-Light-Field dataset
Download full_data.zip under ./data/4D-Light-Field_Gen
Run the following command under ./data/4D-Light-Field_Gen

```
unzip full_data.zip
python LF2hdf5.py --base_dir ./full_data --output_dir ./LF
python FS_gen.py --LF_path ./LF/HCI_LF_trainval.h5 --output_dir ./FS
python save_AiF.py ./FS/HCI_FS_trainval.h5 ./FS
python save_stack.py ./FS/HCI_FS_trainval.h5 ./FS
```
