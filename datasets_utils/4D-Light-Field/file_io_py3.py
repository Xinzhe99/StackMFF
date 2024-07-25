#####################################################################
# This file is part of the 4D Light Field Benchmark.                #
#                                                                   #
# This work is licensed under the Creative Commons                  #
# Attribution-NonCommercial-ShareAlike 4.0 International License.   #
# To view a copy of this license,                                   #
# visit http://creativecommons.org/licenses/by-nc-sa/4.0/.          #
#####################################################################

import configparser
import os
import sys
import numpy as np
import cv2


def read_lightfield(data_folder):
    params = read_parameters(data_folder)
    light_field = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"], 3), dtype=np.uint8)

    views = sorted([f for f in os.listdir(data_folder) if f.startswith("input_") and f.endswith(".png")])

    for idx, view in enumerate(views):
        fpath = os.path.join(data_folder, view)
        try:
            img = read_img(fpath)
            light_field[idx // params["num_cams_x"], idx % params["num_cams_y"], :, :, :] = img
        except IOError:
            print(f"Could not read input file: {fpath}")
            sys.exit()

    return light_field


def read_parameters(data_folder):
    params = dict()

    with open(os.path.join(data_folder, "parameters.cfg"), "r") as f:
        parser = configparser.ConfigParser()
        parser.read_file(f)

        section = "intrinsics"
        params["width"] = parser.getint(section, 'image_resolution_x_px')
        params["height"] = parser.getint(section, 'image_resolution_y_px')
        params["focal_length_mm"] = parser.getfloat(section, 'focal_length_mm')
        params["sensor_size_mm"] = parser.getfloat(section, 'sensor_size_mm')
        params["fstop"] = parser.getfloat(section, 'fstop')

        section = "extrinsics"
        params["num_cams_x"] = parser.getint(section, 'num_cams_x')
        params["num_cams_y"] = parser.getint(section, 'num_cams_y')
        params["baseline_mm"] = parser.getfloat(section, 'baseline_mm')
        params["focus_distance_m"] = parser.getfloat(section, 'focus_distance_m')
        params["center_cam_x_m"] = parser.getfloat(section, 'center_cam_x_m')
        params["center_cam_y_m"] = parser.getfloat(section, 'center_cam_y_m')
        params["center_cam_z_m"] = parser.getfloat(section, 'center_cam_z_m')
        params["center_cam_rx_rad"] = parser.getfloat(section, 'center_cam_rx_rad')
        params["center_cam_ry_rad"] = parser.getfloat(section, 'center_cam_ry_rad')
        params["center_cam_rz_rad"] = parser.getfloat(section, 'center_cam_rz_rad')

        section = "meta"
        params["disp_min"] = parser.getfloat(section, 'disp_min')
        params["disp_max"] = parser.getfloat(section, 'disp_max')
        params["frustum_disp_min"] = parser.getfloat(section, 'frustum_disp_min')
        params["frustum_disp_max"] = parser.getfloat(section, 'frustum_disp_max')
        params["depth_map_scale"] = parser.getfloat(section, 'depth_map_scale')

        params["scene"] = parser.get(section, 'scene')
        params["category"] = parser.get(section, 'category')
        params["date"] = parser.get(section, 'date')
        params["version"] = parser.get(section, 'version')
        params["authors"] = parser.get(section, 'authors').split(", ")
        params["contact"] = parser.get(section, 'contact')

    return params


def read_depth(data_folder, highres=False):
    fpath = os.path.join(data_folder, f"gt_depth_{'highres' if highres else 'lowres'}.pfm")
    try:
        data = read_pfm(fpath)
    except IOError:
        print(f"Could not read depth file: {fpath}")
        sys.exit()
    return data


def read_disparity(data_folder, highres=False):
    fpath = os.path.join(data_folder, f"gt_disp_{'highres' if highres else 'lowres'}.pfm")
    try:
        data = read_pfm(fpath)
    except IOError:
        print(f"Could not read disparity file: {fpath}")
        sys.exit()
    return data


def read_img(fpath):
    img = cv2.imread(fpath)
    if img is None:
        raise IOError(f"Could not read image file: {fpath}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_hdf5(data, fpath):
    import h5py
    with h5py.File(fpath, 'w') as h:
        for key, value in data.items():
            h.create_dataset(key, data=value)


def write_pfm(data, fpath, scale=1, file_identifier="Pf", dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = data.shape[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder
    print(endianess)

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write(f"{file_identifier}\n".encode())
        file.write(f"{width} {height}\n".encode())
        file.write(f"{scale}\n".encode())
        file.write(values.tobytes())


def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception(f'Unknown identifier. Expected: "{expected_identifier}", got: "{identifier}".')

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split()
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception(f'Could not parse dimensions: "{line_dimensions}". '
                            'Expected "width height", e.g. "512 512".')

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            endianness = "<" if scale < 0 else ">"
        except:
            raise Exception(f'Could not parse max value / endianess information: "{line_scale}". '
                            'Should be a non-zero number.')

        try:
            data = np.fromfile(f, f"{endianness}f")
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception(f'Invalid binary values. Could not create {height}x{width} array from input.')

        return data


def _get_next_line(f):
    next_line = f.readline().rstrip().decode('utf-8')
    # ignore comments
    while next_line.startswith('#'):
        next_line = f.readline().rstrip().decode('utf-8')
    return next_line