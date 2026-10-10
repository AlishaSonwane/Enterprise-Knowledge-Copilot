# Recreate the Windows demo

This repository is a reproducible source package, not a portable Python installation.
The owner's prepared Windows machine is verified; a clean-machine installation has
not been tested. Keep API keys and personal documents local even in a private repo.

## Prerequisites and dependencies

Use Windows x64, trusted Python 3.11 x64 with pip/venv, internet for initial setup
and Groq requests, and at least 1.5 GiB available RAM before launch. The tested
runtime was Python 3.11.17. Do not reuse a copied Windows virtual environment.

From a Windows CMD terminal in the downloaded repository:

```bat
py -3.11 -m venv venv311
venv311\Scripts\python.exe -m pip install --only-binary=:all: --index-url https://pypi.org/simple -r requirements.txt
venv311\Scripts\python.exe -m pip check
```

These are setup instructions, not operations performed by the launcher. The original
pins include Torch through sentence-transformers, so dependency installation can
consume substantial storage even when the ONNX backend avoids loading Torch.

## Pinned embedding assets

Download these official assets into a local folder outside Git. Both use revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf` of sentence-transformers/all-MiniLM-L6-v2:

- [model.onnx](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model.onnx), about 90.4 MB;
  SHA-256 `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452`.
- [tokenizer.json](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json);
  SHA-256 `be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037`.

The runtime rejects different bytes. Do not substitute another model. Windows native
imports also require a compatible Microsoft C++ runtime. This machine needed verified
process-local DLL registration; that machine-specific cache is not distributed here.
If an import fails on a new machine, diagnose it rather than downloading arbitrary DLLs.
The optional `dll_directory` profile field supports only the exact DLL hashes encoded
in run_windows.py, not an arbitrary runtime version.

## Private configuration

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

## Start and rebuild the demo index

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

## Offline checks

```bat
venv311\Scripts\python.exe -I -B tests\run_offline.py
node tests\frontend_checks.cjs
```

The JavaScript checks require Node.js; it is not needed to serve the static UI.
No real generation or native model loading is performed by these test harnesses.
