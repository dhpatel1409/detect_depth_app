import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libglib2.0-0", "libsm6", "libxext6", "libxrender1", "libgl1", "libxcb1")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/app")
)

app = modal.App("depth-detect-app")

@app.function(image=image, memory=4096, timeout=120)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.append("/app")
    from app.main import app as web_app
    return web_app