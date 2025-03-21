import os
from concurrent.futures import ThreadPoolExecutor
import requests
from rich.progress import Progress

url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

pretraineds_hifigan_list = [
    (
        "pretrained_v2/",
        [
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    )
]
models_list = [("predictors/", ["rmvpe.pt", "fcpe.pt"])]
embedders_list = [("embedders/contentvec/", ["pytorch_model.bin", "config.json"])]
executables_list = [
    ("", ["ffmpeg.exe", "ffprobe.exe"]),
]

folder_mapping_list = {
    "pretrained_v2/": "rvc/models/pretraineds/hifi-gan/",
    "embedders/contentvec/": "rvc/models/embedders/contentvec/",
    "predictors/": "rvc/models/predictors/",
    "formant/": "rvc/models/formant/",
}


def get_file_size_if_missing(file_list):
    """
    Calculate the total size of files to be downloaded only if they do not exist locally.
    """
    total_size = 0
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if not os.path.exists(destination_path):
                url = f"{url_base}/{remote_folder}{file}"
                response = requests.head(url)
                total_size += int(response.headers.get("content-length", 0))
    return total_size


def download_file(url, destination_path, progress, task_id):
    """
    Download a file from the given URL to the specified destination path,
    updating the rich progress bar as data is downloaded.
    """

    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    progress.update(task_id, total=total_size, description=f"Downloading {os.path.basename(destination_path)}")
    block_size = 1024
    downloaded = 0
    with open(destination_path, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
            downloaded += len(data)
            progress.update(task_id, advance=len(data))


def download_mapping_files(file_mapping_list, progress, task_id):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the rich progress bar as downloads progress.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                if not os.path.exists(destination_path):
                    url = f"{url_base}/{remote_folder}{file}"
                    futures.append(
                        executor.submit(
                            download_file, url, destination_path, progress, task_id
                        )
                    )
        for future in futures:
            future.result()


def split_pretraineds(pretrained_list):
    f0_list = []
    non_f0_list = []
    for folder, files in pretrained_list:
        f0_files = [f for f in files if f.startswith("f0")]
        non_f0_files = [f for f in files if not f.startswith("f0")]
        if f0_files:
            f0_list.append((folder, f0_files))
        if non_f0_files:
            non_f0_list.append((folder, non_f0_files))
    return f0_list, non_f0_list


pretraineds_hifigan_list, _ = split_pretraineds(pretraineds_hifigan_list)


def calculate_total_size(
    pretraineds_hifigan,
    models,
    exe,
):
    """
    Calculate the total size of all files to be downloaded based on selected categories.
    """
    total_size = 0
    if models:
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(embedders_list)
    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)
    total_size += get_file_size_if_missing(pretraineds_hifigan)
    return total_size


def prequisites_download_pipeline(
    pretraineds_hifigan,
    models,
    exe,
):
    """
    Manage the download pipeline for different categories of files using rich progress bar.
    """
    total_size = calculate_total_size(
        pretraineds_hifigan_list if pretraineds_hifigan else [],
        models,
        exe,
    )

    if total_size > 0:
        with Progress() as progress:
            task_id = progress.add_task("Downloading all files", total=total_size)
            if models:
                download_mapping_files(models_list, progress, task_id)
                download_mapping_files(embedders_list, progress, task_id)
            if exe:
                if os.name == "nt":
                    download_mapping_files(executables_list, progress, task_id)
                else:
                    print("No executables needed")
            if pretraineds_hifigan:
                download_mapping_files(pretraineds_hifigan_list, progress, task_id)
            progress.stop_task(task_id) # Ensure task is stopped after completion
            progress.update(task_id, description="[bold green]Download Complete!")

    else:
        print("All files are already downloaded.")


if __name__ == "__main__":
    # Example usage:
    prequisites_download_pipeline(
        pretraineds_hifigan=True,
        models=True,
        exe=True,
    )