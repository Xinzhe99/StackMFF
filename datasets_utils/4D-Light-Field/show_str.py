import h5py

def print_hdf5_structure(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        def print_structure(name, obj):
            print(name)
        hdf5_file.visititems(print_structure)

if __name__ == '__main__':
    hdf5_path = r'/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/project_depth_from_focus/AiFDepthNet/data/4D-Light-Field_Gen/FS/HCI_FS_trainval.h5'
    print_hdf5_structure(hdf5_path)
