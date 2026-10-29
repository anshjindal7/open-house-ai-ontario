"""Put Open House on a Hugging Face Space (or update it).

    hf auth login                                         # once, with a write token from huggingface.co/settings/tokens
    python deploy/upload_space.py YOUR-HF-NAME/open-house-ai-ontario

Uploads the committed files the server needs, with a Space README in place of the GitHub one
(Hugging Face reads the Space's settings from it). The Space builds from the Dockerfile.
Then, in the Space's Settings -> Variables and secrets, add the secret TYPESAFE_API_KEY.
Optional, to keep feedback across restarts: HF_TOKEN (a write token) and FEEDBACK_DATASET
(for example YOUR-HF-NAME/open-house-feedback, created private on first use).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
GITHUB = "https://github.com/anshjindal7/open-house-ai-ontario"
# Not needed to run the app: training data, drafts, checks, tests, training logs.
SKIP = ("train/data/", "train/interpret/", "evals/", "tests/", "train/train_log", "train/compare")

SPACE_README = f"""---
title: Open House
emoji: 🏠
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Plain answers about renting in Ontario
---

# Open House (beta)

Plain answers about renting in Ontario, for tenants and landlords, quoted from the Residential
Tenancies Act, the Standard Form of Lease, LTB guidance and real 2026 LTB decisions.

General information, not legal advice. Code and details: {GITHUB}

Sources and credits: see NOTICE.md. Contains information licensed under the Open Government
Licence – Ontario.
"""


def main() -> None:
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python deploy/upload_space.py YOUR-HF-NAME/SPACE-NAME")
    space = sys.argv[1]
    from huggingface_hub import HfApi

    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    files = [f for f in files if f and not f.startswith(SKIP)]
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print("Note: you have uncommitted changes; the upload uses the files as they are on disk.")

    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            dest = Path(tmp) / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, dest)
        (Path(tmp) / "README.md").write_text(SPACE_README)
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space", commit_message="Update Open House")
    # Remove files an earlier upload sent that no longer exist (the Space's own .gitattributes stays).
    uploaded = set(files) | {"README.md"}
    stale = [f for f in api.list_repo_files(space, repo_type="space") if f not in uploaded and f != ".gitattributes"]
    if stale:
        api.delete_files(space, delete_patterns=stale, repo_type="space", commit_message="Remove old files")
    print(f"Uploaded {len(files)} files. Building at https://huggingface.co/spaces/{space}")
    print("If you haven't yet: Settings -> Variables and secrets -> New secret TYPESAFE_API_KEY.")


if __name__ == "__main__":
    main()
