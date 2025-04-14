from pathlib import Path
from numpy import save, load, concatenate, asarray, min, max
from tqdm import tqdm


def read_index_file(filepath: Path):
    d = {}
    with filepath.open('r') as f:
        obj = None
        raw_arr = []
        overall_min = 9999999999
        overall_max = -1
        for line in f:
            if line.strip()[0:3] == 'DEF':
                if obj:
                    arr = []
                    temp = []
                    for num in raw_arr:
                        if num == -1 and temp:
                            arr.append(temp)
                            temp = []
                        else:
                            temp.append(num)
                    if temp:
                        arr.append(temp)
                    d[obj] = arr
                obj = line.strip().split(' ')[1]
                raw_arr = []
            else:
                nums = line.strip().split(',')
                temp = [int(x) for x in nums if x]
                raw_arr += temp
                min_temp = 9999999999
                for i in temp:
                    if 0 <= i < min_temp:
                        min_temp = i
                if min_temp < overall_min:
                    overall_min = min_temp
                if max(temp) > overall_max:
                    overall_max = max(temp)
        if obj:
            arr = []
            temp = []
            for num in raw_arr:
                if num == -1 and temp:
                    arr.append(temp)
                    temp = []
                else:
                    temp.append(num)
            if temp:
                arr.append(temp)
            d[obj] = arr
    return d, overall_min, overall_max


def convert_output_indices_to_npy():
    input_files = Path(r'E:\Aidan\plaques_extracted\output_indices')
    output_files = Path(r'E:\Aidan\plaques_extracted\np_indices')
    output_files.mkdir(exist_ok=True)
    (output_files / "index.txt").touch()

    with (output_files / "index.txt").open('w') as o:
        for file in tqdm(input_files.iterdir()):
            d, _min, _max = read_index_file(file)
            o.write(file.name + ", " + str(_min) + ", " + str(_max) + "\n")
            save(output_files / file.name, d)
        return


def convert_coords_to_npy_voxels(size):
    input_files = Path(r'E:\Aidan\plaques_extracted\output_coords')
    output_files = Path(r'E:\Aidan\plaques_extracted\np_coords')

    for file in tqdm(input_files.iterdir()):
        l = []
        with file.open('r') as f:
            for line in f:
                point = [float(x) for x in line.strip().split(' ')]
                if len(point) == 3:
                    l.append([int(c // size) for c in point])
        save(output_files / file.name, l)


def load_numpy(filepath: Path, is_dict=False):
    ret = load(filepath, allow_pickle=True)
    if is_dict:
        return ret.item()
    else:
        return ret


# loads multiple numpy files and concatenates the lists
def load_multi_numpy(filepaths: list[Path]):
    all_lists = []
    for file in filepaths:
        all_lists.append(load(file, allow_pickle=True))
    return concatenate(all_lists)


def save_numpy(filepath: Path, data):
    save(filepath, data)


if __name__ == '__main__':
    pass