#!/usr/bin/env python3
"""Check for vLLM precompiled wheel availability."""

import json
import os
import re
import subprocess
import sys
import urllib.request


def get_commit_from_github(repo="vllm-project/vllm", branch="main"):
    """Get the latest commit SHA from GitHub API."""
    api_url = f"https://api.github.com/repos/{repo}/commits/{branch}"
    try:
        req = urllib.request.Request(api_url)
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data["sha"], data["sha"][:9]
    except Exception as e:
        print(f"ERROR: Failed to fetch commit from GitHub API: {e}")
        return None, None


def get_commit_from_local(vllm_dir):
    """Get commit SHA from local git repository."""
    try:
        commit = subprocess.check_output(["git", "-C", vllm_dir, "rev-parse", "HEAD"], text=True).strip()

        commit_short = subprocess.check_output(
            ["git", "-C", vllm_dir, "rev-parse", "--short", "HEAD"], text=True
        ).strip()

        return commit, commit_short
    except Exception as e:
        print(f"ERROR: Failed to get commit from local repository: {e}")
        return None, None


def main():
    vllm_dir = os.environ.get("VLLM_DIR")
    output_file = os.environ.get("WHEEL_ENV_FILE")

    if not output_file:
        print("ERROR: WHEEL_ENV_FILE environment variable must be set")
        sys.exit(1)

    # Get the latest commit - either from local repo or GitHub API
    if vllm_dir:
        print(f"Using local vLLM repository: {vllm_dir}")
        commit, commit_short = get_commit_from_local(vllm_dir)
    else:
        print("Fetching latest vLLM commit from GitHub...")
        commit, commit_short = get_commit_from_github()

    if not commit:
        print("ERROR: Could not determine vLLM commit")
        sys.exit(1)

    print(f"Latest vLLM commit: {commit_short}")

    # Check if wheel exists for this commit
    commit_url = f"https://wheels.vllm.ai/nightly/cu129/vllm/{commit}/"
    has_wheel = False

    try:
        urllib.request.urlopen(commit_url, timeout=5)
        html = urllib.request.urlopen(commit_url).read().decode()
        wheel_match = re.search(r'href="([^"]*x86_64[^"]*\.whl)"', html)
        has_wheel = wheel_match is not None
    except Exception:
        pass

    if has_wheel:
        print(f"✓ Wheel found for commit {commit_short}")
        return 0

    # No wheel for this commit, find the latest available wheel
    print(f"✗ No wheel found for commit {commit_short}")
    print("Searching for latest available wheel...")

    try:
        html = urllib.request.urlopen("https://wheels.vllm.ai/nightly/cu129/vllm/").read().decode()

        # Look for all wheel links and find the most recent one
        wheel_matches = re.findall(r'href="([^"]*x86_64[^"]*\.whl)"', html)

        if wheel_matches:
            # Take the last match (usually the most recent)
            wheel_filename = wheel_matches[-1]
            wheel_url = wheel_filename.replace("../../../", "https://wheels.vllm.ai/")
            print(f"Using fallback wheel: {wheel_url}")

            with open(output_file, "w") as f:
                f.write(f"export VLLM_PRECOMPILED_WHEEL_LOCATION={wheel_url}\n")

            return 0
        else:
            print("ERROR: No fallback wheel found in directory listing")
            print(f"Searched URL: https://wheels.vllm.ai/nightly/cu129/vllm/")
            return 1
    except Exception as e:
        print(f"ERROR: Failed to find fallback wheel: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
