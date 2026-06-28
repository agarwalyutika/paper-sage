"""
One-shot deploy to Hugging Face Spaces.

What it does:
  1. Uploads your fine-tuned reranker to a HF Hub *model* repo (keeps full quality).
  2. Creates a Streamlit *Space*.
  3. Uploads the app + the slim runtime data (NOT the raw PDFs, NOT your chats.db,
     NOT your .env). Big files become LFS automatically.
  4. Points the Space at the fine-tuned reranker and sets your Groq key as a secret.

Run it AFTER logging in:   huggingface-cli login
Then:                      python deploy/deploy_to_hf.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

USER = "yutika1708"
MODEL_REPO = f"{USER}/bge-reranker-base-ft"     # the fine-tuned reranker
SPACE_REPO = f"{USER}/papersage"                # the live demo

# Things we must NEVER upload to the public Space, plus stuff it doesn't need.
IGNORE = [
    ".env", ".git/*", ".gitignore",             # secrets / git internals
    "data/chats.db",                            # your private accounts + chats
    "data/papers/*",                            # raw PDFs (ingestion-only, ~560MB)
    "models/*",                                 # 2GB of models (reranker comes from the Hub)
    ".venv/*", "*.pyc", "*__pycache__*",        # local python junk
    "notebooks/*", "deploy/*", "PROJECT_GUIDE.md", "README.md",
]


def main() -> None:
    api = HfApi()
    try:
        me = api.whoami()
    except Exception:
        sys.exit("Not logged in. Run:  huggingface-cli login   (paste a WRITE token)")
    print("Logged in as:", me["name"])

    # 1) Fine-tuned reranker -> Hub model repo
    ft = ROOT / "models" / "bge-reranker-base-ft"
    if ft.exists():
        create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
        print(f"Uploading fine-tuned reranker -> {MODEL_REPO}  (~1GB, be patient)…")
        api.upload_folder(folder_path=str(ft), repo_id=MODEL_REPO, repo_type="model")
    else:
        print("WARNING: models/bge-reranker-base-ft not found — demo will use the base reranker.")

    # 2) The Space (Docker SDK — runs the Streamlit app via the Dockerfile)
    create_repo(SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)

    # 3a) Space README (frontmatter) first…
    api.upload_file(path_or_fileobj=str(ROOT / "deploy" / "space_readme.md"),
                    path_in_repo="README.md", repo_id=SPACE_REPO, repo_type="space")
    # 3b) …then the app + slim data (README.md is in IGNORE so it isn't overwritten).
    print("Uploading app + runtime data to the Space (large files auto-LFS)…")
    api.upload_folder(folder_path=str(ROOT), repo_id=SPACE_REPO, repo_type="space",
                      ignore_patterns=IGNORE)

    # 4) Configuration: point at the fine-tuned reranker + set the Groq key
    try:
        if ft.exists():
            api.add_space_variable(SPACE_REPO, "RERANKER_MODEL", MODEL_REPO)
        # Free DuckDuckGo returns junk from datacenter IPs -> turn web search off on the
        # demo so out-of-corpus questions answer from the model's general knowledge.
        api.add_space_variable(SPACE_REPO, "WEB_SEARCH_ENABLED", "false")
        groq = os.getenv("GROQ_API_KEY", "")
        if groq:
            api.add_space_secret(SPACE_REPO, "GROQ_API_KEY", groq)
            print("Set GROQ_API_KEY secret + RERANKER_MODEL variable on the Space.")
        else:
            print("WARNING: GROQ_API_KEY not in .env — set it manually in the Space settings.")
    except Exception as e:
        print(f"\nCouldn't set secrets via API ({e}).")
        print("Set them manually:  Space → Settings → Variables and secrets:")
        print(f"   • Variable  RERANKER_MODEL = {MODEL_REPO}")
        print("   • Secret    GROQ_API_KEY   = (your gsk_... key)")

    print("\n✅ Done! Your live demo (first build takes a few minutes):")
    print(f"   https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
