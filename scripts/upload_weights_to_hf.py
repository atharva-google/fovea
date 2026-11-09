"""Upload your local weights/ folder to a Hugging Face model repo (one-time, by the maintainer).

Usage:
    huggingface-cli login            # or set HF_TOKEN
    python scripts/upload_weights_to_hf.py --repo <your-username>/fovea-weights

Then users get every model from one reliable place:
    FOVEA_HF_REPO=<your-username>/fovea-weights python -m fovea download
(or set DEFAULT_HF_REPO in fovea/weights.py so it's used automatically)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi  # noqa: E402

from fovea.config import default_weights_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="e.g. yourname/fovea-weights")
    p.add_argument("--weights-dir", type=Path, default=default_weights_dir())
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(args.weights_dir),
        repo_type="model",
        ignore_patterns=["README.md", "*.part", ".DS_Store"],
        commit_message="Upload Fovea weights",
    )
    print(f"Uploaded {args.weights_dir} -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
