import os

def rename_files(directory, prefix):
    i = 1
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            os.rename(os.path.join(directory, filename), os.path.join(directory, f"{prefix}{i}.mp4"))
            i += 1

rename_files("jumping into pool", "jumping_into_pool")