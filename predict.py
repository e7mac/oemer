import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from argparse import Namespace

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cog import BasePredictor, Input, Path as CogPath

from oemer import MODULE_PATH
from oemer import layers
from oemer.ete import extract, clear_data, CHECKPOINTS_URL
from oemer.draw_teaser import teaser


def download_file(title: str, url: str, save_path: str) -> None:
    """Download a file with progress."""
    resp = urllib.request.urlopen(url)
    length = int(resp.getheader("Content-Length", -1))
    chunk_size = 2**12
    total = 0
    with open(save_path, "wb") as out:
        while True:
            data = resp.read(chunk_size)
            if not data:
                break
            total += out.write(data)
            print(f"{title}: {total*100/length:.1f}%")
    print(f"{title}: Complete")


class Predictor(BasePredictor):
    def setup(self):
        """Download model checkpoints if they don't exist."""
        chk_path = os.path.join(MODULE_PATH, "checkpoints/unet_big/model.onnx")
        if not os.path.exists(chk_path):
            print("Downloading model checkpoints...")
            for idx, (title, url) in enumerate(CHECKPOINTS_URL.items()):
                print(f"Downloading checkpoint ({idx+1}/{len(CHECKPOINTS_URL)}): {title}")
                save_dir = "unet_big" if title.startswith("1st") else "seg_net"
                save_dir = os.path.join(MODULE_PATH, "checkpoints", save_dir)
                save_path = os.path.join(save_dir, title.split("_")[1])
                download_file(title, url, save_path)
            print("Checkpoints downloaded successfully.")

    def predict(
        self,
        image: CogPath = Input(description="Input sheet music image (JPG, PNG, or GIF)"),
        disable_deskew: bool = Input(
            description="Disable automatic deskewing. Set to True if your image is already properly aligned.",
            default=False
        ),
    ) -> list[CogPath]:
        """
        Run Optical Music Recognition on a sheet music image.

        Returns:
            - MusicXML file containing the transcribed music notation
            - Teaser image showing the detected musical elements
        """
        # Clear any previous layer data
        clear_data()

        # Create output directory
        output_dir = Path(tempfile.mkdtemp())

        # Prepare arguments namespace (matching CLI interface)
        args = Namespace(
            img_path=str(image),
            output_path=str(output_dir),
            use_tf=False,
            save_cache=False,
            without_deskew=disable_deskew
        )

        # Run the extraction pipeline
        mxl_path = extract(args)

        # Generate teaser visualization
        teaser_img = teaser()
        teaser_path = mxl_path.replace(".musicxml", "_teaser.png")
        teaser_img.save(teaser_path)

        return [CogPath(mxl_path), CogPath(teaser_path)]
