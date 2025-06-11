from utils import load_numpy
from pathlib import Path
from numpy import zeros, asarray, uint8, rot90
from tqdm import tqdm
from tifffile import imwrite


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


def save_image(data, output_file, flips, dtype=uint8):
    output_file.mkdir(parents=True, exist_ok=True)
    print("Saving images...")
    file_num = 0
    if flips[2]: # if flip z
        for z in tqdm(range(data.shape[2] - 1, -1, -1)):
            layer = rot90(data[::-1, ::-1, z]) if flips[0] and flips[1] else \
                    rot90(data[::-1, :, z]) if flips[0] and not flips[1] else \
                    rot90(data[:, ::-1, z]) if flips[0] and flips[1] else \
                    rot90(data[:, :, z])

            path = output_file / (str(file_num) + ".tif")
            imwrite(path, layer.astype(dtype), dtype=dtype)
            file_num += 1
    else:
        for z in tqdm(range(data.shape[2])):
            layer = rot90(data[::-1, ::-1, z]) if flips[0] and flips[1] else \
                    rot90(data[::-1, :, z]) if flips[0] and not flips[1] else \
                    rot90(data[:, ::-1, z]) if not flips[0] and flips[1] else \
                    rot90(data[:, :, z])
            path = output_file / (str(file_num) + ".tif")
            imwrite(path, layer.astype(dtype), dtype=dtype)
            file_num += 1


def paste_array(img, arr, position):
    x, y, z = position
    paste_region = img[x : x + arr.shape[0], y : y + arr.shape[1], z : z + arr.shape[2]]
    paste_region[arr] = True


def build_image(d, mins, maxs, meshes, output, flips):
    image_size = tuple([maxs[i] - mins[i] for i in range(len(mins))])  # may need to add 1 if image is cropped. e.g., maxs[i] - mins[i] + 1 TODO: fix this bug.
    image = zeros(image_size, dtype=bool)

    count = 0
    for name, data in tqdm(d.items()):
        data_min = asarray(data["min"])
        adjusted_min = data_min - mins
        folder = data["folder"]
        arr = load_numpy(meshes / folder / (name + ".npy"))
        paste_array(image, arr, adjusted_min)
        count += 1

    save_image(image, output, flips)


if __name__ == '__main__':
    image_mins = [0, -7588, -4171]
    image_maxs = [10559, 0, 0]
    mesh_path = Path(r'E:\Aidan\testing_extraction\np_meshes')
    output_path = Path(r'E:\Aidan\testing_extraction\np_meshes_out')

    mesh_index = read_mesh_index(mesh_path)

    print("COMPLETE")



