# Recreate the Windows demo

**For another Windows x64 machine.** Mac/Linux need their own setup; do not run
this Windows launcher there. Existing users do not need to reinstall their setup.

## 1. Download the private repository

Sign into the GitHub account with access to
[Enterprise-Knowledge-Copilot](https://github.com/AlishaSonwane/Enterprise-Knowledge-Copilot).
Select **Code -> Download ZIP**, then extract it to a normal local folder, for
example `C:\Projects\Enterprise-Knowledge-Copilot`. The folder containing
`app.py`, `requirements.txt` and `run_windows.py` is the project root. Do not run
from inside the ZIP. GitHub Desktop and Git are optional for this ZIP route.

Open **Command Prompt (CMD)** and enter:

```bat
cd /d "C:\Projects\Enterprise-Knowledge-Copilot"
```

Replace this example folder if yours differs. All commands below run from that
folder. No PowerShell execution-policy change or activation script is required.

This repository is a reproducible source package, not a portable Python installation.
The owner's prepared Windows machine is verified; a clean-machine installation has
not been tested. Keep API keys and personal documents local even in a private repo.

### Prerequisites

Use Windows x64, trusted Python 3.11 x64 with pip/venv, internet for initial setup
and Groq requests, and at least 1.5 GiB available RAM before launch. The tested
runtime was Python 3.11.17. Do not reuse a copied Windows virtual environment.

## 2. Confirm Python, then create the environment

```bat
py -3.11 --version
py -3.11 -c "import struct; print(struct.calcsize('P') * 8)"
```

Expected: Python 3.11.x and `64`. If Python 3.11 is missing, provision a trusted
security-maintained 3.11 x64 runtime with pip and venv before continuing. Consult
[official Python Windows releases](https://www.python.org/downloads/windows/).
Do not select the newest Python major/minor automatically: the original dependency
pins were validated on 3.11, not 3.14/3.15. Some security-only 3.11 releases may
not have a traditional Windows installer; confirm a suitable trusted distribution
rather than recommending an old installer as security-current. An embeddable ZIP
alone is not the supported pip/venv setup used here.

If your trusted project-local Python is not registered with `py`, replace
`py -3.11` below with its quoted absolute `python.exe` path. Do not copy the old
`venv311` from another machine.

From a Windows CMD terminal in the downloaded repository:

```bat
py -3.11 -m venv venv311
venv311\Scripts\python.exe -m pip install --only-binary=:all: --index-url https://pypi.org/simple -r requirements.txt
venv311\Scripts\python.exe -m pip check
```

These are setup instructions, not operations performed by the launcher. The original
pins include Torch through sentence-transformers, so dependency installation can
consume substantial storage even when the ONNX backend avoids loading Torch.

## 3. Save and verify the two model files

Download these official assets into a local folder outside Git. Both use revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf` of sentence-transformers/all-MiniLM-L6-v2:

- [model.onnx](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model.onnx), about 90.4 MB;
  SHA-256 `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452`.
- [tokenizer.json](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json);
  SHA-256 `be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037`.

For example save both in `C:\Projects\Copilot-models`. Download actual file
bytes, not the Hugging Face webpage. Check the downloaded files in CMD:

```bat
certutil -hashfile "C:\Projects\Copilot-models\model.onnx" SHA256
certutil -hashfile "C:\Projects\Copilot-models\tokenizer.json" SHA256
```

Compare with the exact hashes above. Stop on any mismatch; do not disable the
runtime's integrity check.

The runtime rejects different bytes. Do not substitute another model. Windows native
imports also require a compatible Microsoft C++ runtime. This machine needed verified
process-local DLL registration; that machine-specific cache is not distributed here.
If an import fails on a new machine, diagnose it rather than downloading arbitrary DLLs.
The optional `dll_directory` profile field supports only the exact DLL hashes encoded
in run_windows.py, not an arbitrary runtime version.

## 4. Configure local paths and your API key

```bat
copy .launcher.example.json .launcher.json
copy .env.example .env
notepad .launcher.json
```

For the example folders, replace the profile with:

```json
{
  "python": "C:\\Projects\\Enterprise-Knowledge-Copilot\\venv311\\Scripts\\python.exe",
  "model": "C:\\Projects\\Copilot-models\\model.onnx",
  "tokenizer": "C:\\Projects\\Copilot-models\\tokenizer.json"
}
```

JSON Windows paths need doubled backslashes. Save it, then run `notepad .env`.
Make sure Notepad does not append `.txt` to either filename. Add the four active
settings below: remove the leading `#` from those settings in the example file.
Use the actual key locally in place of `YOUR_LOCAL_KEY`; do not paste it into a
terminal command, screenshot, chat or GitHub. A new machine needs an authorized
Groq API key; the repository does not contain one.

Copy `.launcher.example.json` to `.launcher.json` and replace all three paths with
absolute existing paths. Omit `dll_directory` unless the exact verified workaround
is required. Copy `.env.example` to `.env` and enable these settings locally:

```dotenv
GENERATION_PROVIDER=groq
GROQ_FREE_PLAN_CONFIRMED=1
GROQ_API_KEY=YOUR_LOCAL_KEY
GROQ_MAX_COMPLETION_TOKENS=1024
```

Use your own Groq Free account; the acknowledgment does not verify billing status.
The fixed model is openai/gpt-oss-20b. Never commit a populated `.env`. The launcher
selects offline ONNX embeddings from the profile; it does not require Ollama.

## 5. Launch and check the sample documents

```bat
venv311\Scripts\python.exe -I -B -X utf8 run_windows.py
```

Open the printed `Ready:` URL. Binding is loopback-only and the port can change.
The first fresh startup creates a local database. Use the UI's **Re-index All
Documents** action to index the three included public Markdown samples. On the
verified machine these produced seven chunks; verify your ingestion result.
Use unique filenames for uploads to avoid overwriting existing documents.

Ask: `What is Retrieval-Augmented Generation and how does it reduce hallucinations?`
Compare each claim/citation with `documents/rag_system_guide.md`. A private company
subscription-price question should abstain without cloud generation. The pricing
policy is narrow, not general factual validation. Cloud requests share questions and
retrieved passages with Groq; use only approved/public documents for this demo.

Keep the terminal open. Ctrl+C stops the server; default session bound is one hour.
RAM is checked before startup and monitored during use, with shutdown below 0.5 GiB.
Logs remain local under `data/launcher/`. `--smoke-test` requires an already-built
index and checks safe HTTP routes using an index copy without generation.

## 6. Optional verification and everyday restart

```bat
venv311\Scripts\python.exe -I -B tests\run_offline.py
node tests\frontend_checks.cjs
```

The JavaScript checks require Node.js; it is not needed to serve the static UI.
No real generation or native model loading is performed by these test harnesses.

Next time, open CMD, change to the project folder and run the same launch command
from step 5. Do not reinstall dependencies or recreate the index every session.
If desired, create a local `Start-Copilot.cmd` in the project root with:

```bat
@echo off
setlocal
cd /d "%~dp0" || exit /b 1
"%~dp0venv311\Scripts\python.exe" -I -B -X utf8 "%~dp0run_windows.py"
if errorlevel 1 pause
```

This shortcut assumes the step-2 environment is named `venv311` inside the project.
Double-click it and keep the terminal open. The shortcut stays Git-ignored.

## Troubleshooting: stop at the actual error

- **`py` / Python 3.11 not found:** use your trusted runtime's absolute path;
  do not switch Python versions just to get past this message.
- **No compatible binary wheel:** keep the original pins; verify Python 3.11
  x64 and the full pip error. Do not build native packages as a workaround.
- **Missing launcher path:** check every absolute path in `.launcher.json`.
- **JSON parse failure:** check doubled backslashes, double quotes and commas.
- **ONNX DLL load failure:** record the traceback; check architecture and the
  [official Microsoft C++ runtime requirements](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
  Installing a runtime may require administrator approval; the launcher does not
  install it. Do not copy random DLLs or the old machine's entire runtime cache.
- **RAM STOP:** close unused apps normally; do not reduce the safety threshold.
- **Groq configuration incomplete:** check the four active `.env` lines, including
  provider and Free acknowledgment, without displaying the key.
- **Cloud authentication/rate-limit error:** check the key or wait for Free limits;
  do not enable paid billing or automatic retries to continue the demo.
- **No relevant documents:** confirm the sample indexing action completed.
- **Browser cannot connect:** use the latest printed URL and keep the terminal
  open; the previous session's port may differ.

Storage still includes Python packages, both model assets and the local index.
Cloud generation removes the need for a local LLM, not all local disk/RAM needs.
This guide was checked against the prepared code; a second clean machine has
not been used to verify these instructions end to end.
