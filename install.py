#!/usr/bin/env python3
"""
install.py – One-click installer for the Qwen provider.

What it does:
  1. Installs required Python packages (g4f, fastapi, uvicorn, aiohttp).
  2. Locates the g4f package on disk.
  3. Copies the custom Qwen provider files into the correct g4f subdirectory:
       g4f/Provider/Qwen.py          ← web provider (cookie-based, no login)
       g4f/Provider/qwen/            ← sub-package (OAuth2, cookies, utils)

Usage:
    python install.py
"""
import subprocess
import sys
import shutil
import importlib
from pathlib import Path

HERE = Path(__file__).parent
PROVIDER_SRC = HERE / "qwen_provider"  # the unzipped files


def run(*args):
    print(f"  $ {' '.join(str(a) for a in args)}")
    subprocess.check_call([str(a) for a in args])


def main():
    print("=" * 60)
    print("  Qwen Provider Installer")
    print("=" * 60)

    # ── 1. Install Python dependencies ────────────────────────────────────────
    print("\n[1/3] Installing dependencies …")
    run(
        sys.executable, "-m", "pip", "install", "--quiet",
        "g4f[all]",
        "fastapi>=0.111",
        "uvicorn[standard]",
        "aiohttp",
        "curl-cffi",
    )
    print("      ✓ Dependencies installed.")

    # ── 2. Locate g4f ────────────────────────────────────────────────────────
    print("\n[2/3] Locating g4f …")
    import importlib, importlib.util
    spec = importlib.util.find_spec("g4f")
    if spec is None:
        print("  ✗ g4f not found after install — something went wrong.")
        sys.exit(1)

    g4f_root = Path(spec.origin).parent          # …/site-packages/g4f/
    provider_dir = g4f_root / "Provider"
    qwen_sub = provider_dir / "qwen"
    qwen_sub.mkdir(parents=True, exist_ok=True)
    print(f"      g4f root  : {g4f_root}")
    print(f"      Target dir: {qwen_sub}")

    # ── 3. Copy files ────────────────────────────────────────────────────────
    print("\n[3/3] Installing provider files …")

    # Qwen.py → g4f/Provider/Qwen.py  (top-level provider entry point)
    src_qwen = PROVIDER_SRC / "Qwen.py"
    dst_qwen = provider_dir / "Qwen.py"
    shutil.copy2(src_qwen, dst_qwen)
    print(f"      Copied  Qwen.py  → {dst_qwen}")

    # Everything else → g4f/Provider/qwen/
    sub_files = [
        "QwenCode.py",
        "__init__.py",
        "cookie_generator.py",
        "fingerprint.py",
        "generate_ua.py",
        "oauthFlow.py",
        "qwenContentGenerator.py",
        "qwenOAuth2.py",
        "sharedTokenManager.py",
        "stubs.py",
    ]
    for fname in sub_files:
        src = PROVIDER_SRC / fname
        if src.exists():
            dst = qwen_sub / fname
            shutil.copy2(src, dst)
            print(f"      Copied  {fname:35s} → qwen/{fname}")
        else:
            print(f"      SKIP    {fname} (not found in source)")

    print("\n" + "=" * 60)
    print("  ✓ Installation complete!")
    print("=" * 60)
    print()
    print("  Start the server:")
    print("    uvicorn server:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("  Test it:")
    print("    curl http://localhost:8000/v1/models")
    print()
    print("  Chat (streaming):")
    print("    curl http://localhost:8000/v1/chat/completions \\")
    print("      -H 'Content-Type: application/json' \\")
    print("      -d '{\"model\":\"qwen3.7-plus\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"stream\":true}'")
    print()
    print("  For QwenCode OAuth login (run once):")
    print("    python -m g4f auth qwencode login")
    print()


if __name__ == "__main__":
    main()
