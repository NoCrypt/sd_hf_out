import launch

if not launch.is_installed("huggingface_hub"):
    launch.run_pip("install huggingface_hub", "huggingface_hub")