import gradio as gr
from pathlib import Path
import os
import shutil
from huggingface_hub import HfApi, snapshot_download


HF_TOKEN = os.environ["HF_TOKEN"]

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set")

api = HfApi()
username = api.whoami(token=HF_TOKEN)["name"]
repo = "sd_out"
user_repo = f"{username}/{repo}"


def refresh_images():
    """
    Refreshes the images by deleting the existing image directory and retrieving new images.

    Returns:
        A list of image files.
    """
    try:
        shutil.rmtree(os.environ["IMAGE_DIR"])
    except:
        pass

    image_dir = Path(
        snapshot_download(repo_id=user_repo, repo_type="dataset", token=HF_TOKEN)
    )

    os.environ["IMAGE_DIR"] = str(image_dir)

    image_files = list(Path(image_dir).rglob("*.[pjw]*[npjg]*[ge]*"))

    return image_files


with gr.Blocks(
    analytics_enabled=False, title="Image Gallery", theme="NoCrypt/miku"
) as demo:
    gr.HTML("""<center><h1>Image Gallery</h1></center>""")
    submit = gr.Button("Refresh", variant="primary")
    gallery = gr.Gallery(
        value="", columns=4, show_label=False, height=800, object_fit="contain"
    )

    submit.click(refresh_images, outputs=[gallery])


demo.launch(debug=True)
