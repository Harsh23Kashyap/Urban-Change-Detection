# Instructions for the agent — push this folder to GitHub

This folder (`urban-change-detection/`) is a clean, ready-to-publish repository.
Datasets are intentionally excluded (they auto-download on first run and are
listed in `.gitignore`). Only commit the files already present here.

## What to do

1. Make sure you are in the repo folder:
   ```bash
   cd urban-change-detection
   ```

2. Initialize git (skip if already a repo):
   ```bash
   git init
   git branch -M main
   ```

3. Stage and commit everything in the folder (the `.gitignore` already excludes
   `data/`, caches, and model files — do NOT force-add those):
   ```bash
   git add .
   git status            # confirm: NO data/ files, only src/, results/, README, etc.
   git commit -m "Feasibility probes: built-up classification + change detection"
   ```

4. Create the GitHub repository. Two options:

   **Option A — GitHub CLI (easiest, if `gh` is installed and authenticated):**
   ```bash
   gh repo create urban-change-detection --public --source . --remote origin --push
   ```
   That one command creates the repo and pushes. Done.

   **Option B — manual (if no `gh`):**
   - Create an empty repo on github.com named `urban-change-detection`
     (do NOT initialize it with a README/license — this folder already has them).
   - Then connect and push:
     ```bash
     git remote add origin https://github.com/<YOUR_USERNAME>/urban-change-detection.git
     git push -u origin main
     ```
   Replace `<YOUR_USERNAME>` with the actual GitHub username.

5. Verify on github.com that these files are present and `data/` is NOT:
   ```
   src/probe_classification.py
   src/probe_changedetection.py
   results/classification_feasibility.json
   results/changedetection_feasibility.json
   README.md  requirements.txt  LICENSE  .gitignore
   ```

## Notes
- Public vs private: use `--public` or `--private` with `gh`, or pick the
  visibility when creating the repo manually. (Code only, no private data.)
- If `git push` asks for credentials, use a GitHub Personal Access Token as the
  password, or `gh auth login` first.
- Do NOT commit the `data/` folder or any `.parquet` / `.zip` / model files —
  `.gitignore` handles this; just don't `git add -f` them.
