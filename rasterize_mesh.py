from numpy import stack, zeros, asarray, ndarray, concatenate
from utils import load_numpy, load_multi_numpy, save_numpy
from open3d import geometry, utility
from scipy.ndimage import binary_fill_holes
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


def rasterize_single_mesh(indices: ndarray, coordinates, coords_offset):
    # create mesh
    mesh = geometry.TriangleMesh()

    if coords_offset != 0:
        indices -= coords_offset

    mesh.vertices = utility.Vector3dVector(coordinates)
    mesh.triangles = utility.Vector3iVector(indices)
    mesh.remove_unreferenced_vertices()

    # get bounds
    mins = mesh.get_min_bound()
    maxs = mesh.get_max_bound()

    # voxelize mesh
    voxel_grid = geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1)

    # visualization.draw_geometries([voxel_grid])

    voxel_indices = stack([voxel.grid_index for voxel in voxel_grid.get_voxels()])

    # convert to binary mask
    image = zeros(tuple([int(maxs[i] - mins[i] + 1) for i in range(len(mins))]), dtype=bool)

    for index in voxel_indices:
        image[index[0], index[1], index[2]] = True

    image = binary_fill_holes(image)
    return image, mins, maxs


# input_path should be a path to a numpy file, output_path should be a directory
# coords is a list containing coordinates.  coords_offset is the index of the first value in coords.
def rasterize_file(input_path: Path, coords: list[list[float]], coords_offset: int, output_path: Path):
    print("rasterizing file", str(input_path.absolute), len(coords), coords_offset, str(output_path.absolute))
    index_file = load_numpy(input_path, True)
    output_path.mkdir(exist_ok=True, parents=True)
    index_path = (output_path / "index.txt")
    index_path.touch(exist_ok=True)

    with index_path.open('w') as index:
        for key in index_file.keys():
            image, mins, maxs = rasterize_single_mesh(asarray(index_file[key]), coords, coords_offset)
            index.write(key + ", " + ", ".join(map(str, concatenate((mins, maxs)))) + "\n")
            save_numpy(output_path / key, image)

    return


def splitter(args: tuple):
    rasterize_file(*args)
    return


def rasterize_all_indices(indices_folder: Path, coords_folder: Path, output_filepath: Path, threads=8):
    print("Loading coordinates")
    coords_filepaths = sorted(coords_folder.glob('*.npy'), key=lambda path: int(path.stem.split('.')[0].rsplit('_')[1]))
    coords = load_multi_numpy(coords_filepaths)

    print("Loading indices")
    indices_files = list(indices_folder.iterdir())
    indices_index = indices_files[-1]  # grab the '_index.npy' file
    indices_files = indices_files[:-1]

    index_offsets = load_numpy(indices_index, True)

    # make parameters
    params = []

    for file in indices_files:
        temp_input_path = file
        temp_offsets = index_offsets[file.stem]
        temp_coords = coords[int(temp_offsets[0]):int(temp_offsets[1]) + 1]
        params.append(tuple([temp_input_path, temp_coords, int(temp_offsets[0]), output_filepath / file.stem.split('.')[0]]))

    if threads == 1:
        list(tqdm(map(splitter, params), total=len(indices_files)))
    else:
        pool = Pool(processes=threads)
        try:
            # need to convert to list so the tqdm iterator is consumed; otherwise progress bar doesn't update.
            list(tqdm(pool.imap_unordered(splitter, params), total=len(indices_files)))
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected, terminating thread pool...")
            pool.terminate()
            pool.join()
        except Exception as e:  # this one ideally should never occur...
            print("Thread pool for aligning large images encountered an error, terminating...")
            print(e)
            pool.terminate()
            pool.join()
        else:
            pool.close()
            pool.join()

    print("COMPLETE")
    return


if __name__ == '__main__':
    indices_filepath = Path(r'E:\Aidan\plaques_extracted\np_indices')
    coords_folder = Path(r'E:\Aidan\plaques_extracted\np_coords')
    coords_filepath = sorted(coords_folder.glob('*.npy'), key=lambda path: int(path.stem.split('.')[0].rsplit('_')[1]))

    output_filepath = Path(r'E:\Aidan\plaques_extracted\meshes')

    rasterize_all_indices(indices_filepath, coords_folder, output_filepath, 48)
