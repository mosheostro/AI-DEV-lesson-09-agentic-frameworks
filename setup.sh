# Lab setup script — copy this file as `setup.sh` into each lab's `lab/` folder.
# It is intentionally self-contained: no shared helper files, no cross-lab
# dependencies. A student who starts at Lesson 5 can run only that lab's
# setup.sh and have everything they need.
#
# USAGE
#   From the folder this file lives in (the lab's `code/` folder, or the SDD
#   build repo's root), source it — don't execute it:
#     source setup.sh
#
#   IMPORTANT: use `source` (or `.`), NOT `./setup.sh`. A venv activation only
#   persists in the shell that ran the activate command — if you execute the
#   script normally, the venv dies when the script exits and your terminal is
#   not actually activated.
#
# WINDOWS
#   Run this from Git Bash (the bash that ships with Git for Windows) — NOT
#   PowerShell or cmd, which cannot source a .sh file. Easiest path: open the
#   project in VS Code and set the integrated terminal to Git Bash. There is a
#   setup.ps1 next to this file that finds Git Bash and points you there if you
#   start from PowerShell. WSL also works — it behaves like native Linux.
#
# WHAT THIS DOES (in order; the Bedrock invoke is always the final step)
#   1. .env       — ensure a .env exists at the resolved "env root" (copy
#                   from .env.example if missing). The env root is the first
#                   directory walking up from this script that contains
#                   either file; if none is found, the lab folder itself
#                   is used (it ships its own .env.example).
#   2. venv       — create+activate the shared .venv at the env root.
#   3. deps       — install this lab's requirements.txt into the venv.
#   4. Bedrock    — invoke the configured Bedrock model with a tiny prompt to
#                   confirm end-to-end connectivity, then invoke the embedding
#                   model (this lab embeds documents with Titan). This is the
#                   last check by design — when it passes, the lab is truly ready.
#
# Every failure path below prints an actionable next step ("do X") on top of
# the error description. The student should never have to guess what to do.
#
# This script is idempotent — safe to source again after fixing an issue.

(return 0 2>/dev/null) || {
  echo "✗ This script must be sourced, not executed (venv activation needs"
  echo "  to persist in YOUR shell, not a child shell)."
  echo "  Run:  source ${BASH_SOURCE[0]:-$0}"
  exit 1
}

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The git root of wherever setup.sh lives: the clone root for a published
# standalone lab, or empty in the instructor's (non-git) course tree. We use it
# to keep the venv/.env INSIDE the repo a student cloned, and to label the lab.
_GIT_ROOT=""
_d="${_SCRIPT_DIR}"
for _ in 1 2; do   # setup.sh sits AT the repo root (build repo) or one level in (code/, SDD/)
  if [ -e "${_d}/.git" ]; then _GIT_ROOT="${_d}"; break; fi
  _next="$(dirname "${_d}")"
  [ "${_next}" = "${_d}" ] && break
  _d="${_next}"
done

# Lab label: when setup.sh sits AT the repo root (the published -sdd build repo)
# the lab name is that folder; when it's in a code/ or SDD/ subfolder, it's the
# folder above.
if [ -n "${_GIT_ROOT}" ] && [ "${_SCRIPT_DIR}" = "${_GIT_ROOT}" ]; then
  _LAB_NAME="$(basename "${_SCRIPT_DIR}")"
else
  _LAB_NAME="$(basename "$(dirname "${_SCRIPT_DIR}")")"
fi

# Detect the platform. This script is bash-only: macOS/Linux run it directly,
# WSL behaves like Linux, and on native Windows the only shell that can run it
# is Git Bash (which reports MINGW/MSYS). PowerShell and cmd cannot source a
# .sh file at all — a student who lands there should follow the Windows note in
# the lesson README (run from Git Bash, e.g. via VS Code's integrated terminal).
case "$(uname -s 2>/dev/null)" in
  Linux*)               _OS="linux" ;;
  Darwin*)              _OS="macos" ;;
  MINGW*|MSYS*|CYGWIN*) _OS="windows" ;;
  *)                    _OS="unknown" ;;
esac

# Resolve the "env root" — where .env and the venv live.
#
#  - Published standalone lab (a git clone): anchor at the clone root, so the
#    venv/.env stay INSIDE the repo. This is the key fix — never walk up and out
#    of the clone (an unrelated .env in a parent dir must not hijack the setup).
#  - Instructor's course tree (not a git repo): keep the old behaviour — walk up
#    from $PWD to find the single shared .env, so one venv serves every lab.
_PWD="$(pwd)"
if [ -n "${_GIT_ROOT}" ]; then
  _REPO_ROOT="${_GIT_ROOT}"
else
  _REPO_ROOT=""
  _d="${_PWD}"
  for _ in 1 2 3 4; do
    if [ -f "${_d}/.env" ]; then _REPO_ROOT="${_d}"; break; fi
    _next="$(dirname "${_d}")"
    [ "${_next}" = "${_d}" ] && break   # hit filesystem root
    _d="${_next}"
  done
  [ -z "${_REPO_ROOT}" ] && _REPO_ROOT="${_PWD}"
fi

echo "→ Lab:      ${_LAB_NAME}"
echo "→ Env root: ${_REPO_ROOT}"
[ "${_OS}" = "windows" ] && echo "→ Shell:    Git Bash on Windows (correct — not PowerShell/cmd)"

# --- 1. .env ----------------------------------------------------------------
if [ ! -f "${_REPO_ROOT}/.env" ]; then
  # Prefer an .env.example sitting next to where .env will live (the canonical
  # repo case). Otherwise fall back to the lab's own .env.example, which is
  # always present and works as a complete reference.
  _SEED=""
  if [ -f "${_REPO_ROOT}/.env.example" ]; then
    _SEED="${_REPO_ROOT}/.env.example"
  elif [ -f "${_SCRIPT_DIR}/.env.example" ]; then
    _SEED="${_SCRIPT_DIR}/.env.example"
  fi

  if [ -n "${_SEED}" ]; then
    cp "${_SEED}" "${_REPO_ROOT}/.env"
    echo "⚠ Created ${_REPO_ROOT}/.env from ${_SEED}."
    echo "  ACTION REQUIRED:"
    echo "    1. Open ${_REPO_ROOT}/.env in your editor."
    echo "    2. Fill in AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,"
    echo "       BEDROCK_MODEL_ID (and any lab-specific vars listed there)."
    echo "    3. Re-run:  source ${BASH_SOURCE[0]}"
    return 1
  else
    echo "✗ No .env at ${_REPO_ROOT}, and no .env.example to seed from."
    echo "  ACTION REQUIRED: create ${_REPO_ROOT}/.env with at least:"
    echo "    AWS_ACCESS_KEY_ID=..."
    echo "    AWS_SECRET_ACCESS_KEY=..."
    echo "    AWS_REGION=us-east-1"
    echo "    BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0"
    echo "  Then re-run:  source ${BASH_SOURCE[0]}"
    return 1
  fi
fi

# --- 2. venv ----------------------------------------------------------------
# Resolve a Python launcher. macOS/Linux/WSL expose `python3`; native Windows
# (Git Bash) typically only has `python` or the `py` launcher, not `python3`.
_PYTHON=""
for _cand in python3 python py; do
  if command -v "${_cand}" >/dev/null 2>&1; then
    _PYTHON="${_cand}"
    break
  fi
done
if [ -z "${_PYTHON}" ]; then
  echo "✗ No Python interpreter found on your PATH (looked for python3, python, py)."
  echo "  ACTION REQUIRED: install Python 3.11+ before continuing."
  echo "    Ubuntu/Debian/WSL:  sudo apt update && sudo apt install python3 python3-venv"
  echo "    macOS (Homebrew):   brew install python@3.11"
  echo "    Windows:            https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH')"
  return 1
fi

# A venv is only usable if its creation FINISHED. `python -m venv` makes the folder
# first and writes the activate script last, ~2 s later (much longer on a Windows or
# /mnt/c filesystem). Ctrl-C, a closed terminal, or a failed ensurepip in that window
# leaves a .venv with no activate script — and a plain "does .venv exist?" check then
# skips creation on every later run. So check the finished markers, and rebuild an
# unfinished venv automatically. It holds nothing but installed packages.
_venv_complete() {
  { [ -f "$1/bin/activate" ] || [ -f "$1/Scripts/activate" ]; } || return 1
  for _py in "$1/bin/python" "$1/Scripts/python.exe" "$1/Scripts/python"; do
    [ -x "${_py}" ] && "${_py}" -c "import sys" >/dev/null 2>&1 && return 0
  done
  return 1
}
if [ -d "${_REPO_ROOT}/.venv" ] && ! _venv_complete "${_REPO_ROOT}/.venv"; then
  echo "⚠ ${_REPO_ROOT}/.venv is incomplete or broken (an earlier setup was interrupted,"
  echo "  or the Python it was built from is gone). Rebuilding it."
  [ -n "${VIRTUAL_ENV:-}" ] && command -v deactivate >/dev/null 2>&1 && deactivate
  rm -rf "${_REPO_ROOT}/.venv"
fi

if [ ! -d "${_REPO_ROOT}/.venv" ]; then
  echo "→ Creating shared venv at ${_REPO_ROOT}/.venv"
  _VENV_ERR="$(mktemp 2>/dev/null || echo "${_REPO_ROOT}/.venv-create.err")"
  if ! "${_PYTHON}" -m venv "${_REPO_ROOT}/.venv" 2>"${_VENV_ERR}"; then
    echo "✗ Failed to create the virtualenv. Error:"
    sed 's/^/    /' "${_VENV_ERR}"
    rm -f "${_VENV_ERR}"
    # Never leave a half-made venv behind: the next run would trip over it.
    rm -rf "${_REPO_ROOT}/.venv"
    echo "  ACTION REQUIRED: the most common cause on Linux/WSL is a missing"
    echo "  venv module. Run:"
    echo "    sudo apt update && sudo apt install python3-venv"
    echo "  (On Windows this is rare — reinstall Python from python.org if it persists.)"
    echo "  Then re-source this script."
    return 1
  fi
  rm -f "${_VENV_ERR}"
fi

# The venv's activate script lives under bin/ on macOS/Linux but Scripts/ on
# Windows. Git Bash sources the same bash `activate` script — it just sits in
# Scripts/. Pick whichever exists.
if [ -f "${_REPO_ROOT}/.venv/bin/activate" ]; then
  _ACTIVATE="${_REPO_ROOT}/.venv/bin/activate"
elif [ -f "${_REPO_ROOT}/.venv/Scripts/activate" ]; then
  _ACTIVATE="${_REPO_ROOT}/.venv/Scripts/activate"
else
  echo "✗ Could not find the venv activate script under .venv/bin or .venv/Scripts."
  echo "  ACTION REQUIRED: recreate the venv:"
  echo "    rm -rf ${_REPO_ROOT}/.venv && source ${BASH_SOURCE[0]}"
  return 1
fi
# shellcheck disable=SC1090
source "${_ACTIVATE}"
echo "✓ Virtualenv active: ${VIRTUAL_ENV}"

# --- 3. requirements --------------------------------------------------------
if [ -f "${_SCRIPT_DIR}/requirements.txt" ]; then
  echo "→ Installing ${_LAB_NAME} dependencies..."
  if ! pip install --disable-pip-version-check -r "${_SCRIPT_DIR}/requirements.txt"; then
    echo "✗ pip install failed (see the pip output above for the exact reason)."
    echo "  COMMON CAUSES & ACTIONS:"
    echo "    - No internet / behind a corporate proxy → set HTTPS_PROXY and re-source."
    echo "    - Version conflict with a previous lab → recreate the venv:"
    echo "        deactivate && rm -rf ${_REPO_ROOT}/.venv && source ${BASH_SOURCE[0]}"
    echo "    - System package missing (e.g. gcc, headers) → install build tools."
    return 1
  fi
  echo "✓ Dependencies installed."
else
  # No requirements.txt — this is the minimal BUILD repo (the answer-free clean
  # slate). Install the universal Bedrock base so the smoke test can run, PLUS the
  # deps the provided server/ needs to start (it's given to you, so it has to run);
  # the student's coding agent adds whatever else the exercise turns out to need.
  #
  # The `mcp<2` pin is deliberate: the MCP 2.0 SDK removed a symbol that LangChain's
  # MCP bridge still imports, so anything that talks to this server through a
  # framework needs the 1.x SDK. If a later `pip install` pulls mcp 2.x back in,
  # re-run `pip install "mcp<2"`.
  echo "→ No requirements.txt (clean build repo) — installing the base + the provided server's deps..."
  if ! pip install --disable-pip-version-check boto3 python-dotenv "mcp<2" faiss-cpu numpy; then
    echo "✗ pip install failed (see the pip output above for the exact reason)."
    echo "  No internet / behind a proxy? Set HTTPS_PROXY and re-source."
    return 1
  fi
  echo "✓ Base dependencies installed (boto3, python-dotenv, mcp<2, faiss-cpu, numpy)."
fi

# --- 4. Bedrock smoke test (FINAL STEP — do not add checks after this) ------
# Inline Python so the lab's setup.sh has no external Python helper to ship.
# Each failure mode below maps to a concrete next action for the student.
echo "→ Calling Bedrock to confirm end-to-end connectivity..."
# The venv's python is a native Windows exe under Git Bash, so it can't read an
# MSYS-style path like /c/Users/... Convert to a Windows path (C:\Users\...)
# before handing it to Python, so its canonical .env check resolves cleanly.
_ENV_ROOT_FOR_PY="${_REPO_ROOT}"
if [ "${_OS}" = "windows" ] && command -v cygpath >/dev/null 2>&1; then
  _ENV_ROOT_FOR_PY="$(cygpath -w "${_REPO_ROOT}")"
fi
export LAB_SETUP_ENV_ROOT="${_ENV_ROOT_FOR_PY}"
python - <<'PYCHECK'
import os
import sys

# Imports done one-at-a-time so a missing dep produces a targeted message
# instead of a generic ImportError.
try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    print("✗ python-dotenv is not installed.", file=sys.stderr)
    print("  ACTION REQUIRED: add `python-dotenv` to this lab's requirements.txt", file=sys.stderr)
    print("  and re-source setup.sh.", file=sys.stderr)
    sys.exit(1)

try:
    import boto3
    from botocore.exceptions import (
        BotoCoreError, ClientError, NoCredentialsError, PartialCredentialsError,
        EndpointConnectionError,
    )
except ImportError:
    print("✗ boto3 is not installed.", file=sys.stderr)
    print("  ACTION REQUIRED: add `boto3` to this lab's requirements.txt and re-source.", file=sys.stderr)
    sys.exit(1)

# Resolve .env in priority order, in case the student has moved it:
#   1. The path setup.sh just resolved (LAB_SETUP_ENV_ROOT/.env) — canonical.
#   2. A walk-up from $PWD via find_dotenv(usecwd=True) — catches the student
#      who moved .env somewhere else. usecwd=True avoids the default
#      frame-walking variant, which asserts when run from heredoc/stdin.
#   3. Nothing found — print actionable instructions and exit.
canonical_env = os.path.join(os.environ.get("LAB_SETUP_ENV_ROOT", ""), ".env")
if os.path.isfile(canonical_env):
    load_dotenv(canonical_env)
else:
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found)
        print(f"  (loaded .env from {found} — not the expected {canonical_env})")
    else:
        print("✗ Could not find a .env file to load.", file=sys.stderr)
        print(f"  Checked: {canonical_env}", file=sys.stderr)
        print(f"  Searched up from: {os.getcwd()}", file=sys.stderr)
        print("  ACTION REQUIRED: either", file=sys.stderr)
        print("    (a) restore your .env at the path above and re-source, or", file=sys.stderr)
        print("    (b) cd to a directory that contains (or sits below) your .env", file=sys.stderr)
        print("        and re-source the script.", file=sys.stderr)
        sys.exit(1)

region = os.getenv("AWS_REGION", "us-east-1")
model_id = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
print(f"  region={region}  model={model_id}")

try:
    client = boto3.client("bedrock-runtime", region_name=region)
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": "Reply with exactly one word: ok"}]}],
        inferenceConfig={"maxTokens": 5, "temperature": 0.0},
    )
    reply = resp["output"]["message"]["content"][0]["text"].strip()
    print(f"✓ Bedrock OK — model replied: {reply!r}")

    # This lab's RAG tool (03-multi-tool-server) embeds the IT KB with the Titan model,
    # which needs its own model-access grant. Check it now so the MCP server can't
    # fail later when it builds its index.
    import json
    embed_model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    try:
        resp = client.invoke_model(
            modelId=embed_model_id,
            body=json.dumps({"inputText": "ok", "dimensions": 256, "normalize": True}),
        )
        dim = len(json.loads(resp["body"].read())["embedding"])
        print(f"✓ Embeddings OK — {embed_model_id} returned a {dim}-dim vector")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "Unknown")
        msg = e.response.get("Error", {}).get("Message", str(e))
        print(f"✗ Embedding model check failed [{code}]: {msg}", file=sys.stderr)
        if code == "AccessDeniedException":
            print("  ACTION REQUIRED: enable access to 'Titan Text Embeddings V2' in the", file=sys.stderr)
            print("  Bedrock console (Discover → Model catalog → Model access) — see", file=sys.stderr)
            print("  00-aws-setup.md. If your IAM policy allowlists specific models,", file=sys.stderr)
            print(f"  it must also cover {embed_model_id!r}.", file=sys.stderr)
        else:
            print("  ACTION REQUIRED: verify BEDROCK_EMBEDDING_MODEL_ID in .env is a model", file=sys.stderr)
            print(f"  available in {region!r} (default: amazon.titan-embed-text-v2:0).", file=sys.stderr)
        sys.exit(1)
except (NoCredentialsError, PartialCredentialsError):
    print("✗ AWS credentials missing or incomplete.", file=sys.stderr)
    print("  ACTION REQUIRED: open your repo-root .env and set:", file=sys.stderr)
    print("    AWS_ACCESS_KEY_ID=...", file=sys.stderr)
    print("    AWS_SECRET_ACCESS_KEY=...", file=sys.stderr)
    print("    AWS_REGION=us-east-1   (or your enabled Bedrock region)", file=sys.stderr)
    sys.exit(1)
except EndpointConnectionError:
    print(f"✗ Could not reach the Bedrock endpoint in region {region!r}.", file=sys.stderr)
    print("  ACTION REQUIRED: verify AWS_REGION in .env is a Bedrock-supported", file=sys.stderr)
    print("  region (us-east-1, us-west-2, etc.) and that you have network access.", file=sys.stderr)
    sys.exit(1)
except ClientError as e:
    code = e.response.get("Error", {}).get("Code", "Unknown")
    msg = e.response.get("Error", {}).get("Message", str(e))
    print(f"✗ Bedrock returned [{code}]: {msg}", file=sys.stderr)
    if code == "AccessDeniedException":
        print("  ACTION REQUIRED: two common causes —", file=sys.stderr)
        print("    (a) the IAM user lacks bedrock:InvokeModel — check its attached policy in IAM;", file=sys.stderr)
        print(f"    (b) model access for this model isn't enabled for your account.", file=sys.stderr)
        print(f"        Open {model_id!r} in the Bedrock console (Discover → Model", file=sys.stderr)
        print("        catalog → Model access) and enable it, then re-source.", file=sys.stderr)
    elif code == "ResourceNotFoundException":
        print("  ACTION REQUIRED: the model ID is not valid in this region.", file=sys.stderr)
        print(f"  Verify BEDROCK_MODEL_ID in .env matches a model available in {region!r}.", file=sys.stderr)
    elif code == "ValidationException":
        print("  ACTION REQUIRED: the model ID may be malformed or not enabled.", file=sys.stderr)
        print("  Confirm BEDROCK_MODEL_ID exactly matches the value shown in the Bedrock console.", file=sys.stderr)
    elif code in ("UnrecognizedClientException", "InvalidSignatureException", "InvalidClientTokenId"):
        print("  ACTION REQUIRED: AWS credentials are rejected. Verify", file=sys.stderr)
        print("  AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env, and that the", file=sys.stderr)
        print("  IAM user has bedrock:InvokeModel permission.", file=sys.stderr)
    elif code == "ThrottlingException":
        print("  ACTION REQUIRED: you've hit a Bedrock quota. Wait ~1 minute and re-source.", file=sys.stderr)
    else:
        print("  ACTION REQUIRED: search the error code above in the AWS Bedrock docs.", file=sys.stderr)
    sys.exit(1)
except BotoCoreError as e:
    print(f"✗ Bedrock BotoCoreError: {e}", file=sys.stderr)
    print("  ACTION REQUIRED: re-check AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /", file=sys.stderr)
    print("  AWS_REGION in .env.", file=sys.stderr)
    sys.exit(1)
PYCHECK

if [ $? -ne 0 ]; then
  echo ""
  echo "✗ Setup did not complete. Fix the issue above and re-source: source ${BASH_SOURCE[0]}"
  return 1
fi

echo ""
echo "✓ ${_LAB_NAME} is ready. Run an exercise with:  python <exercise>.py"
