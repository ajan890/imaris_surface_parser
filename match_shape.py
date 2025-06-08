from argparse import ArgumentParser
from pathlib import Path
from tifstack import TifStack
from numpy import ndarray, zeros, uint8, pad
from tifffile import imwrite
from tqdm import tqdm


def pad_to_shape(pad_shape: tuple, arr: ndarray, mode='constant'):
    assert len(pad_shape) == len(arr.shape)
    if pad_shape == arr.shape: return arr
    pad_dim = [pad_shape[i] - arr.shape[i] for i in range(len(pad_shape))]
    pad0 = list(map(lambda x: (x // 2, (x + 1) // 2), pad_dim))
    return pad(arr, pad_width=pad0, mode=mode)


# resizes a ndarray to match the shape provided by target_shape
def resize_array(arr: ndarray, target_shape: tuple):
    output = zeros(target_shape, dtype=arr.dtype)
    min_shape = tuple(min(arr.shape[i], target_shape[i]) for i in range(3))

    slices_input = tuple(slice(0, min_shape[i]) for i in range(3))
    slices_output = tuple(slice(0, min_shape[i]) for i in range(3))

    output[slices_output] = arr[slices_input]
    return output


def main(args):
    input_path = Path(args.input)
    reference_path = Path(args.reference)
    output_path = Path(args.output)

    reference_tifstack = TifStack(reference_path)
    input_tif = TifStack(input_path).as_3d_numpy()

    target_shape = reference_tifstack.shape
    print(f"Current shape: {input_tif.shape}")
    print(f"Target shape: {target_shape}")
    output_tif = pad_to_shape(target_shape, input_tif)

    output_path.mkdir(parents=True, exist_ok=True)
    file_num = 0
    for z in tqdm(range(output_tif.shape[0])):
        path = output_path / (str(file_num) + ".tif")
        imwrite(path, output_tif[z, :, :].astype(uint8), dtype=uint8)
        file_num += 1

    print("COMPLETE")


# Takes in two paths pointing to .tif stacks, and pads the input image to match the size of the reference.
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--reference', '-r', required=True, help="Path to reference image tif directory")
    parser.add_argument('--input', '-i', required=True, help="Path to input image tif directory")
    parser.add_argument('--output', '-o', required=True, help="Path to file output")

    main(parser.parse_args())
