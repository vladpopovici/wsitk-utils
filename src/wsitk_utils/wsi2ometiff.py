from typing import Optional, Tuple

import pyvips
from wsitk_core import WSI
from pathlib import Path
import configargparse as opt
from datetime import datetime


def build_omexml(w: WSI,
                 actual_width: Optional[int]=None,
                 actual_height: Optional[int]=None,
                 actual_bands: Optional[int]=3) -> str:
    """
    Extracts relevant metadata from WSI and builds a minimalistic OME-XML description.

    :param w: a whole slide image (WSI) object.
    :param actual_width: width of the region to be written (in case partial image is read)
    :param actual_height: height of the region to be written (in case partial image is read)
    :param actual_bands: number of bands
    :return: the OME-XML description as a string.
    """
    vendor = str(w._original_meta['openslide.vendor'])
    dt = datetime.strptime(
        w._original_meta[vendor + '.GENERAL.SLIDE_CREATIONDATETIME'],
        '%d/%m/%Y %I:%M:%S',
    )
    if actual_width is None:
        actual_width = w.info['width']
    if actual_height is None:
        actual_height = w.info['height']
    mag = round(w.info['objective_power'],1)
    o = f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
    <Instrument ID="Instrument:0">
        <Detector ID="Detector:0:0" 
                Model="{w._original_meta[vendor + '.GENERAL.CAMERA_TYPE']}"
                />
        <Objective ID="Objective:0:0" 
                Model="{w._original_meta[vendor + '.GENERAL.OBJECTIVE_NAME']}" 
                NominalMagnification="{mag}"
                />
    </Instrument>
    <Image ID="Image:0" Name="{w._original_meta[vendor + '.GENERAL.OBJECTIVE_MAGNIFICATION']}x">
        <AcquisitionDate>{dt.isoformat()}</AcquisitionDate>
        <Description>"{w._original_meta[vendor + '.GENERAL.SLIDE_ID']}"</Description>
        <InstrumentRef ID="Instrument:0"/>
        <ObjectiveSettings ID="Objective:0:0"/>
        <Pixels BigEndian="true" 
                DimensionOrder="XYZCT" 
                ID="Pixels:0" 
                Interleaved="false" 
                PhysicalSizeX="{w.info['mpp_x']}" 
                PhysicalSizeXUnit="µm" 
                PhysicalSizeY="{w.info['mpp_y']}" 
                PhysicalSizeYUnit="µm" 
                SignificantBits="8" 
                SizeC="{actual_bands}" 
                SizeT="1" 
                SizeX="{actual_width}" 
                SizeY="{actual_height}" 
                SizeZ="1" 
                Type="uint8">
            <Channel ID="Channel:0:0" SamplesPerPixel="1">
                <LightPath/>
            </Channel>
            <Channel ID="Channel:0:1" SamplesPerPixel="1">
                <LightPath/>
            </Channel>
            <Channel ID="Channel:0:2" SamplesPerPixel="1">
                <LightPath/>
            </Channel>
            <MetadataOnly/>
        </Pixels>
    </Image>
</OME>"""
    return o

def wsi2ometiff(wsi_path, tiff_path, crop: Optional[Tuple[int,int,int,int]|bool]) -> None:
    """
    Converts a WSI file to OME-TIFF format.

    :param wsi_path: source file path.
    :param tiff_path: destination file path.
    :param crop: either bool to control auto-crop or (x0, y0, width, height) for the crop region

    :return: None
    """
    wsi = WSI(Path(wsi_path))

    if isinstance(crop, bool):
        im = pyvips.Image.new_from_file(wsi_path, autocrop=crop)
    else:
        if crop is None:
            x0, y0, width, height = 0, 0, wsi.info["width"], wsi.info["height"]
        else:
            x0, y0, width, height = crop
            x0 = max(0, min(x0, wsi.info["width"]))
            y0 = max(0, min(y0, wsi.info["height"]))
            width = min(width, wsi.info["width"] - x0)
            height = min(height, wsi.info["height"] - y0)
        im = pyvips.Image.new_from_file(wsi_path, autocrop=False)
        im = im.crop(x0, y0, width, height)

    if im.hasalpha():
        # alpha channel in 4th band, use for masking
        im = im.flatten()

    image_height = im.height
    image_bands = im.bands
    # split to separate image planes and stack vertically ready for OME
    im = pyvips.Image.arrayjoin(im.bandsplit(), across=1)
    im = im.copy()
    im.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    im.set_type(pyvips.GValue.gstr_type, "image-description",
                build_omexml(wsi, actual_width=im.width, actual_height=image_height, actual_bands=image_bands))

    im.tiffsave(tiff_path, compression="jpeg", Q=89,
                tile=True, bigtiff=True, subifd=True,
                pyramid=True, tile_width=512, tile_height=512)

    return


if __name__ == "__main__":
    p = opt.ArgumentParser(description="Convert an image from WSI format to OME-TIFF with eventual auto/cropping.")
    p.add_argument("--input", action="store", help="whole slide image to process", required=True)
    p.add_argument("--output", action="store", help="destination file path", required=True)
    p.add_argument("--autocrop", action="store_true",
                   help="""try to crop the image to the bounding box of the tissue (if OpenSlide provides one!)"""
                        """If <autocrop> is provided, <crop> is ignored.""")
    p.add_argument("--crop", action="store", help="region to crop (x0, y0, width, height in level-0 coordinates)",
                   nargs=4, type=int, required=False, default=None)

    args = p.parse_args()

    wsi2ometiff(args.input, args.output,
                crop=True if args.autocrop else args.crop)
# end
