import gradio as gr
from modules import shared, script_callbacks, paths, extensions
from huggingface_hub import HfApi, get_token_permission
from huggingface_hub.utils import HfHubHTTPError
from requests import HTTPError
from pathlib import Path

root_path = paths.script_path
api = HfApi()
username = ""
repo = "sd_out"
user_repo = ""
enabled = False
token = ""


def get_self_extension_path():
    """
    Returns the path of the active extension that contains the current file.

    This function checks if the global variable '__file__' exists, and if so,
    assigns its value to the 'filepath' variable. Otherwise, it imports the
    'inspect' module and uses the 'getfile' function to get the path of the
    current file.

    It then iterates over the active extensions and checks if the path of each
    extension is a substring of the 'filepath'. If a match is found, the path
    of the extension is returned.

    Returns:
        str: The path of the active extension that contains the current file.
    """
    if "__file__" in globals():
        filepath = __file__
    else:
        import inspect

        filepath = inspect.getfile(lambda: None)
    for ext in extensions.active():
        if ext.path in filepath:
            return ext.path


def on_image_saved(params):
    """
    Saves an image to a remote repository if the `enabled` flag is set to True and the `enable_hf_out` option is enabled.
    
    Parameters:
    - params: A dictionary containing information about the image to be saved.
    
    Returns:
    - None
    
    Raises:
    - HTTPError: If there was an error while uploading the image.
    """
    global username, api, user_repo, enabled, token

    if not enabled:
        return

    if not shared.opts.enable_hf_out:
        return
    
    print("[HF Out] Uploading Image...")
    try:
        api.upload_file(
            repo_id=user_repo,
            path_or_fileobj=Path(root_path) / params.filename,
            token=token,
            run_as_future=True,
        )
    except HTTPError as e:
        print("[HF Out] Failed to save image:", e)
        


def on_ui_settings():
    """
    A function to handle UI settings.

    This function adds an option to the shared `opts` object. The option is called `enable_hf_out` and it is of type `shared.OptionInfo`. The default value of the option is `True`. The option is used to enable the "Save to HF" feature, which requires the `hf-token-out` parameter to be provided. If the `hf-token-out` parameter is not provided, the option will be ignored.

    Parameters:
        None

    Return:
        None
    """
    shared.opts.add_option(
        "enable_hf_out",
        shared.OptionInfo(
            True,
            "Enable Save to HF (hf-token-out required else ignored)",
            section=("hf_out", "HF Out"),
        ),
    )


def on_app_started(_, __):
    """
    Initializes the extension when it starts.

    This function performs the following tasks:
    - Sets global variables: `username`, `api`, `user_repo`, `enabled`, and `token`.
    - Checks if a HF Token is provided. If not, it prints a message and disables HF Out.
    - Sets the `token` variable to the provided HF Token.
    - Checks the permission of the token. If it has "read" permission, it prints a message and returns.
    - Retrieves the username using the `api.whoami()` function and the provided HF Token.
    - Creates a Dataset Repo if it doesn't exist. It sets the `dataset_url` variable to the created repo URL.
    - Creates a Space Repo if it doesn't exist. It sets the `space_url` variable to the created repo URL.
    - Adds a secret key `HF_TOKEN` to the Space Repo.
    - Uploads a file `gallery_space.py` to the Space Repo.
    - Restarts the Space Repo.
    - Sets the `enabled` variable to `True`.
    - Prints a message that the extension is ready to roll.

    This function takes no parameters and has no return value.
    """
    global username, api, user_repo, enabled, token, user_repo, repo


    if not hasattr(shared.cmd_opts, "hf_token_out"):
        print("[HF Out] No HF Token provided. HF Out will be disabled.")
        return
    token = shared.cmd_opts.hf_token_out

    if get_token_permission(token) == "read":
        print(
            "[HF Out] Token permission was on 'READ'. Please provide a 'WRITE' token."
        )
        return

    username = api.whoami(token=shared.cmd_opts.hf_token_out)["name"]
    user_repo = f"{username}/{repo}"

    # Create Dataset Repo if haven't
    try:
        dataset_url = api.create_repo(
            repo_id=user_repo, private=True, repo_type="dataset"
        )
        print("[HF Out] Created Dataset Repo: ", dataset_url)

    except HfHubHTTPError as e:
        print("[HF Out] Dataset Repo already exists. Skipping.")

    # Create Space Repo if haven't
    try:
        space_url = api.create_repo(
            repo_id=user_repo + "_gallery", private=True, repo_type="space", space_sdk="gradio"
        )
        # if the repo doesnt exist, create it, and set it up
        api.add_space_secret(
            repo_id=user_repo + "_gallery", key="HF_TOKEN", value=token
        )
        api.upload_file(
            repo_id=user_repo + "_gallery",
            path_or_fileobj=get_self_extension_path() + "/gallery_space.py",
            path_in_repo="app.py",
            token=token,
            run_as_future=True,
        )
        api.restart_space(repo_id=user_repo + "_gallery", token=token)

        print("[HF Out] Created Space Repo: ", space_url)
    except HfHubHTTPError as e:
        print("[HF Out] Space Repo already exists. Skipping.")

    enabled = True
    print("[HF Out] Ready to roll!")


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_image_saved(on_image_saved)
