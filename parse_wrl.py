from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree
from utils import save_numpy

from rasterize_mesh import rasterize_all_indices
from build_image import read_mesh_index, build_image

from numba import jit


COORDS_SUBFOLDER = "np_coords"
INDICES_SUBFOLDER = "np_indices"
NORMALS_SUBFOLDER = "np_normals"
MESH_SUBFOLDER = "np_meshes"
IMAGE_SUBFOLDER = "image"


# voxel_sizes is a list [x, y, z]
# skip_to = 0: start from coords, skip_to = 1: start from normals, skip_to = 2: start from triangles
def read_wrl(filepath: Path, output_path: Path, voxel_sizes: list, skip_to: int = 0):
    with open(filepath, 'r', encoding='utf-8') as infile:
        file_count = 0
        in_coords = False
        in_normals = False
        in_triangles = False
        line = ""
        while not in_coords:
            line = infile.readline()
            if "Coordinate" in line:
                in_coords = True
        if not in_coords:
            print("ERROR: bad wrl format - cannot find coordinates.")
            return -1

        coords_path = output_path / COORDS_SUBFOLDER
        coords_path.mkdir(exist_ok=True)

        # loop through coords
        print("Parsing coordinates")
        if skip_to >= 1:
            line = infile.readline()
            print("Skipping parsing coordinates")
            while ']' not in line:
                line = infile.readline()
            in_coords = False
        while in_coords:
            coords = []
            line_count = 0
            while line_count < 1000000:
                line = infile.readline()
                if ']' not in line:
                    coord = (line.split(',')[0]).split(' ')[-3:]
                    coord = [int(c // voxel_sizes[i]) for i, c in enumerate(map(float, coord))]
                    coords.append(coord)
                    line_count += 1
                else:
                    coord = (line.split(']')[0]).strip().split(' ')[-3:]
                    coord = [int(c // voxel_sizes[i]) for i, c in enumerate(map(float, coord))]
                    coords.append(coord)
                    in_coords = False
                    break
            save_numpy(coords_path / f"coord_{file_count}.npy", coords)
            file_count += 1

        while not in_normals:
            line = infile.readline()
            if "{" in line:
                in_normals = True
        if not in_normals:
            print("ERROR: bad wrl format - cannot find normals.")
            return -1

        normals_path = output_path / NORMALS_SUBFOLDER
        normals_path.mkdir(exist_ok=True)
        file_count = 0

        # loop through normals
        print("Parsing normals")
        if skip_to >= 2:
            line = infile.readline()
            print("Skipping parsing normals")
            while ']' not in line:
                line = infile.readline()
            in_normals = False
        while in_normals:
            normals = []
            line_count = 0
            while line_count < 1000000:
                line = infile.readline()
                if ']' not in line:
                    normal = (line.split(',')[0]).split(' ')[-3:]
                    normal = list(map(float, normal))
                    normals.append(normal)
                    line_count += 1
                else:
                    normal = (line.split(']')[0]).strip().split(' ')[-3:]
                    normal = list(map(float, normal))
                    normals.append(normal)
                    in_normals = False
                    break
            save_numpy(normals_path / f"normal_{file_count}.npy", normals)
            file_count += 1

        # loop through triangles
        print("Parsing triangles")
        raw_arr = []
        while not in_triangles:
            line = infile.readline()
            if "coordIndex" in line:
                in_triangles = True
                nums = [int(n) for n in (line.split('[')[1]).strip().split(',') if n]
                raw_arr += nums
        if not in_triangles:
            print("ERROR: bad wrl format - cannot find triangles.")
            return -1

        triangles_path = output_path / INDICES_SUBFOLDER
        triangles_path.mkdir(exist_ok=True)
        triangle_index = {}
        file_count = 0
        cur_def = "DEF _0"
        tri_done = False
        while in_triangles:
            d = {}
            overall_min = 9999999999
            overall_max = -1
            def_count = 0
            while def_count < 1000:
                line = infile.readline()
                while ']' not in line:
                    nums = [int(n) for n in line.strip().split(',') if n]
                    raw_arr += nums
                    line = infile.readline()
                nums = [int(n) for n in (line.split(']')[0]).strip().split(',') if n]
                raw_arr += nums

                arr = []
                temp = []
                for num in raw_arr:
                    if num == -1 and temp:
                        arr.append(temp)
                        temp = []
                    else:
                        if num < overall_min:
                            overall_min = num
                        if num > overall_max:
                            overall_max = num
                        temp.append(num)
                if temp:
                    arr.append(temp)
                d[cur_def.split(' ')[1]] = arr

                raw_arr = []
                while 'coordIndex' not in line:
                    line = infile.readline()
                    if "DEF _" in line:
                        cur_def = " ".join(line.strip().split(' ')[:2])
                        def_count += 1
                    if line == '':
                        in_triangles = False
                        tri_done = True
                        break
                if tri_done:
                    break
                if 'coordIndex' in line:
                    nums = [int(n) for n in (line.split('[')[1]).strip().split(',') if n]
                    raw_arr += nums
            save_numpy(triangles_path / f"indices_{file_count}.npy", d)
            triangle_index[f'indices_{file_count}'] = [overall_min, overall_max]
            file_count += 1
        save_numpy(triangles_path / "_index.npy", triangle_index)
        return 0


def parse_wrl(args):
    input_path = Path(args.input)
    output_root = Path(args.output)

    # directory checks
    if not input_path.exists():
        print("Input file does not exist.  Terminating...")
        return

    if not output_root.exists():
        print(f"Creating output directory: {output_root.absolute()}")
        output_root.mkdir(parents=True)
    elif args.skip_to == 0:
        c = ''
        while not c or c[0] not in {'y', 'n', 'Y', 'N'}:
            print(f"Output directory: {output_root.absolute()}")
            c = input("Output directory exists.  Continuing operation will destroy all contents.  Proceed?  [y/n]")
        if c[0] in {'n', 'N'}:
            print("Terminating process.")
            return
        print(f"Output directory exists, burning all contents...")
        rmtree(output_root)
        output_root.mkdir(parents=True)

    if args.skip_to <= 2:
        # get voxel sizes
        voxel_sizes = [args.dx, args.dy, args.dz]

        # read wrl file and get coordinates, normals, and triangles
        read_wrl(input_path, output_root, voxel_sizes, skip_to=args.skip_to)

        print("wrl converted to index, normal, and coordinate files successfully.")
    if args.skip_to <= 3:
        print("Voxelizing meshes.")

        # rasterize the mesh
        rasterize_all_indices(indices_folder=(output_root / INDICES_SUBFOLDER),
                              coords_folder=(output_root / COORDS_SUBFOLDER),
                              output_filepath=(output_root / MESH_SUBFOLDER),
                              threads=args.num_threads)
        print("Voxelization successful.")
    if args.skip_to <= 4:
        print("Building image.")
        # load mesh indices
        mesh_index = read_mesh_index(output_root / MESH_SUBFOLDER)

        # construct final image
        image_mins = [min(pair) for pair in [args.x] + [args.y] + [args.z]]
        image_maxs = [max(pair) for pair in [args.x] + [args.y] + [args.z]]

        build_image(d=mesh_index,
                    mins=image_mins,
                    maxs=image_maxs,
                    meshes=(output_root / MESH_SUBFOLDER),
                    output=(output_root / IMAGE_SUBFOLDER),
                    flips=[args.flip_x, args.flip_y, args.flip_z]
                    )
        print("COMPLETE")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--input', '-i', type=str, required=True,
                        help="Path to input .wrl file")
    parser.add_argument('--output', '-o', type=str, required=True,
                        help="Path to output directory.  EXISTING CONTENTS IN THE OUTPUT DIRECTORY WILL BE DELETED.  Non-existant directories will be created.")
    parser.add_argument('--force_restart', '-fr', action='store_true',
                        help="Since the process may take long, the program will automatically continue from the last saved point if it crashed.  Include this argument to force a restart.")
    parser.add_argument('--num_threads', '-n', type=int, default=1,
                        help="Number of threads used for assembling final output image.")
    parser.add_argument('--dx', '-dx', type=float, required=True,
                        help="Voxel size in x-axis")
    parser.add_argument('--dy', '-dy', type=float, required=True,
                        help="Voxel size in y-axis")
    parser.add_argument('--dz', '-dz', type=float, required=True,
                        help="Voxel size in z-axis")
    parser.add_argument('--x', '-x', nargs=2, type=int, required=True,
                        help="Minimum and maximum of the x-axis.")
    parser.add_argument('--y', '-y', nargs=2, type=int, required=True,
                        help="Minimum and maximum of the y-axis.")
    parser.add_argument('--z', '-z', nargs=2, type=int, required=True,
                        help="Minimum and maximum of the y-axis.")
    parser.add_argument('--skip_to', '-st', type=int, default=0,
                        help="Skip to certain save point to avoid recomputing in case of crash.  Mutually exclusive with --force_restart")
    parser.add_argument("--flip_x", '-fx', action='store_true',
                        help="Flip images along x axis")
    parser.add_argument("--flip_y", '-fy', action='store_true',
                        help="Flip images along y axis")
    parser.add_argument("--flip_z", '-fz', action='store_true',
                        help="Flip images along z axis")
    parse_wrl(parser.parse_args())
