from utils import load_numpy
from pathlib import Path
from numpy import zeros, asarray, uint8, swapaxes
from tqdm import tqdm
from tifffile import imwrite

# AD6 brain ims: (622.555 GB)
# Image size:       X: 10559    Y: 7589     Z: 4171

# micrometers       X:          Y:          Z:
# Voxel size:       1.80        1.80        1.80
# Min:              0.00        -1.37e4     -7508
# Max:              1.90e4      0.00        0.00
# X:\3D_stitched_LS\20231010_FM230407_07_LS_15x_800z_AD1\Plaques\LS\FM230407_07_LS_6x_1000z_deconvolved_crop_LS.ims

def read_mesh_index(mesh_filepath: Path):
    d = {}
    for dir in mesh_filepath.iterdir():
        index_file = dir / "index.txt"
        folder = Path(dir).name
        with index_file.open('r') as f:
            for line in f:
                s = line.strip().split(',')
                # s[0]: mesh index.  s[1], s[2], s[3]: x, y, z coords, respectively
                d[s[0]] = {"min": [int(float(s[1])), int(float(s[2])), int(float(s[3]))],
                           "max": [int(float(s[4])), int(float(s[5])), int(float(s[6]))],
                           "folder": folder}
    return d


def save_image(data, output_file):
    output_file.mkdir(parents=True, exist_ok=True)
    _dtype = uint8
    print("Saving images...")
    for z in tqdm(range(data.shape[2])):
        layer = data[:, :, z].swapaxes(0, 1)
        path = output_file / (str(z) + ".tif")
        imwrite(path, layer.astype(_dtype), dtype=_dtype)


def paste_array(img, arr, position):
    x, y, z = position
    paste_region = img[x : x + arr.shape[0], y : y + arr.shape[1], z : z + arr.shape[2]]
    paste_region[arr] = True


# TODO: make building image work for cases where coordinates go from -n -> 0 instead of 0 -> n.
#       (Maybe detect that and just flip image after?)
def build_image(d, mins, maxs, meshes, output):
    image_size = tuple([maxs[i] - mins[i] + 1 for i in range(len(mins))])
    image = zeros(image_size, dtype=bool)

    count = 0
    for name, data in tqdm(d.items()):
        data_min = asarray(data["min"])
        adjusted_min = data_min - mins
        folder = data["folder"]
        arr = load_numpy(meshes / folder / (name + ".npy"))
        paste_array(image, arr, adjusted_min)
        count += 1

    save_image(image, output)


if __name__ == '__main__':
    image_mins = [0, -7588, -4171]
    image_maxs = [10559, 0, 0]
    mesh_path = Path(r'E:\Aidan\testing_extraction\np_meshes')
    output_path = Path(r'E:\Aidan\testing_extraction\np_meshes_out')

    mesh_index = read_mesh_index(mesh_path)
    # build_image(mesh_index, image_mins, image_maxs, mesh_path, output_path)

    print("COMPLETE")

    # img = zeros((8, 8, 8), dtype=bool)
    # paste = asarray([[[False, True], [True, False]], [[True, False], [False, True]]])
    # paste2 = asarray([[[True, False], [False, True]], [[False, True], [True, False]]])
    #
    # paste_array(img, paste, ((2, 3, 1)))
    # paste_array(img, paste2, ((2, 3, 1)))
    #
    # print("TEST")

    # test_path = Path(r'E:\Aidan\plaques_extracted\testing')
    # img = zeros((80, 70, 100), dtype=bool)
    # save_image(img, test_path)

