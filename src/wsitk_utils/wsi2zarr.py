from typing import Optional, Tuple, Union
from pathlib import Path
import configargparse as opt
from wsitk_core import WSI
from math import floor
import zarr
from tqdm import trange
import pyvips
import numpy as np

def wsi2zarr(
        wsi_path: Union[str|Path],
        dst_path: Union[str|Path],
        crop: Optional[Tuple[int,int,int,int]|bool],
        band_size: Optional[int]=1528,
) -> None:
    """
    Converts a WSI file to pyramidal ZARR format.

    :param wsi_path: source file path.
    :param dst_path: destination file path.
    :param crop: either bool to control auto-crop or (x0, y0, width, height) for the crop region
    :param band_size: band height for processed regions
    :return: None
    """
    if not isinstance(wsi_path, Path):
        wsi_path = Path(wsi_path)
    if not isinstance(dst_path, Path):
        dst_path = Path(dst_path)
        if not dst_path.exists():
            dst_path.mkdir(parents=True, exist_ok=True)

    wsi = WSI(wsi_path)

    # initially, whole image
    x0, y0, width, height = (0, 0, wsi.info["width"], wsi.info["height"])

    if isinstance(crop, bool):
        if crop and wsi.info['roi'] is not None:
            x0, y0, width, height = (wsi.info['roi']['x0'],
                                     wsi.info['roi']['y0'],
                                     wsi.info['roi']["width"],
                                     wsi.info['roi']["height"])
    else:
        if crop is not None:
            x0, y0, width, height = crop
            x0 = max(0, min(x0, wsi.info["width"]))
            y0 = max(0, min(y0, wsi.info["height"]))
            width = min(width, wsi.info["width"] - x0)
            height = min(height, wsi.info["height"] - y0)

    levels = np.zeros((2, wsi.level_count), dtype=np.int64)

    with (zarr.open_group(str(dst_path/'pyramid_0.zarr'), mode='w') as root):
        for i in trange(wsi.level_count, desc="Pyramid"):
            # copy levels from WSI, band by band...
            # -level i crop region:
            cx0 = int(floor(x0 / wsi.downsample_factor(i)))
            cy0 = int(floor(y0 / wsi.downsample_factor(i)))
            cw = int(floor(width / wsi.downsample_factor(i)))
            ch = int(floor(height / wsi.downsample_factor(i)))

            im = pyvips.Image.new_from_file(str(wsi_path), level=i, autocrop=False)
            im = im.crop(cx0, cy0, cw, ch)
            im = im.flatten()

            shape = (ch, cw, 3)  # YXC axes
            levels[:, i] = (cw, ch)

            arr = root.zeros('/'+str(i), shape=shape, chunks=(4096, 4096, None), dtype="uint8")
            n_bands = ch // band_size
            incomplete_band = shape[0] % band_size
            for j in trange(n_bands, desc=f"Level {i}"):  # by horizontal bands
                buf = im.crop(0, j * band_size, cw, band_size).numpy()
                arr[j * band_size : (j + 1) * band_size] = buf
                # arr[j * band_size:(j + 1) * band_size, ...] = \
                #     wsi.get_region_px(cx0, cy0+j*band_size, cw, band_size, as_type=np.uint8)

            if incomplete_band > 0:
                buf = im.crop(0, n_bands * band_size, cw, incomplete_band).numpy()
                arr[n_bands * band_size : n_bands * band_size + incomplete_band] = buf
                # arr[n_bands * band_size: n_bands * band_size + incomplete_band, ...] = \
                #     wsi.get_region_px(cx0, n_bands*band_size, cw, incomplete_band, as_type=np.uint8)
        root.attrs["max_level"] = wsi.level_count
        root.attrs["channel_names"] = ["R", "G", "B"]
        root.attrs["dimension_names"] = ["y", "x", "c"]
        root.attrs["mpp_x"] = wsi.info['mpp_x']
        root.attrs["mpp_y"] = wsi.info["mpp_y"]
        root.attrs["mag_step"] = int(wsi.info['magnification_step'])
        root.attrs["objective_power"] = wsi.info['objective_power']
        root.attrs["extent"] = levels.tolist()

    return


if __name__ == "__main__":
    p = opt.ArgumentParser(description="Convert an image from WSI format to ZARR with eventual auto/cropping.")
    p.add_argument("--input", action="store", help="whole slide image to process", required=True)
    p.add_argument("--output", action="store", help="destination file path", required=True)
    p.add_argument("--autocrop", action="store_true",
                   help="""try to crop the image to the bounding box of the tissue (if OpenSlide provides one!)"""
                        """If <autocrop> is provided, <crop> is ignored.""")
    p.add_argument("--crop", action="store", help="region to crop (x0, y0, width, height in level-0 coordinates)",
                   nargs=4, type=int, required=False, default=None)

    args = p.parse_args()

    wsi2zarr(args.input, args.output, crop=True if args.autocrop else args.crop)
# end
