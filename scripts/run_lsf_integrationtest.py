#!/usr/bin/env python3
"""
LSF Integration Test Runner for Terratorch

This script manages the submission of integration tests to LSF with proper dependencies.
It handles test_models_fit as a prerequisite for dependent tests and manages cleanup.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Constants for test categorization
DEPENDENT_TESTS = [
    "test_latest_terratorch_version_buildings_predict",
    "test_latest_terratorch_version_floods_predict",
    "test_latest_terratorch_version_burnscars_predict",
]

VLLM_TESTS = [
    "integration-tests-vllm-release",
    "vllm-tests-tt-main",
]

PREREQUISITE_TEST = "test_models_fit"
CLEANUP_TEST = "test_cleanup"


def extract_test_names(test_file_path):
    """Extract test function names from the test file."""
    test_file = Path(test_file_path)

    if not test_file.exists():
        print(f"Error: Test file not found at {test_file}")
        print(f"Current directory: {Path.cwd()}")
        sys.exit(1)

    test_names = []
    with test_file.open("r") as f:
        for line in f:
            match = re.match(r"^def (test_\w+)", line.strip())
            if match:
                test_names.append(match.group(1))

    return test_names


def submit_lsf_job(
    job_name, log_file, err_file, command, dependency=None, gpu_config="num=1", terminate_on_dependency_failure=False
):
    """Submit a job to LSF and return the job ID."""
    bsub_cmd = [
        "bsub",
        "-gpu",
        gpu_config,
        "-R",
        "rusage[cpu=8, mem=32GB]",
        "-J",
        job_name,
        "-o",
        str(log_file),
        "-e",
        str(err_file),
    ]

    if dependency:
        bsub_cmd.extend(["-w", dependency])
        # Add -ti flag to terminate job if dependency fails
        if terminate_on_dependency_failure:
            bsub_cmd.append("-ti")

    bsub_cmd.append(command)

    result = subprocess.run(bsub_cmd, capture_output=True, text=True)

    # Extract job ID from output like "Job <12345> is submitted..."
    match = re.search(r"Job <(\d+)>", result.stdout)
    if match:
        return match.group(1)

    print(f"Warning: Could not extract job ID from: {result.stdout}")
    return None


def check_job_status(output_dir):
    """Check the status of jobs from a previous run."""
    output_path = Path(output_dir).resolve()
    jobs_file = output_path / "job_ids.json"

    if not jobs_file.exists():
        print(f"Error: Job IDs file not found at {jobs_file}")
        print(f"Make sure you're pointing to the correct output directory.")
        sys.exit(1)

    # Load job information
    with jobs_file.open("r") as f:
        job_data = json.load(f)

    print("=" * 100)
    print(f"Job Status Report - {output_path.name}")
    print("=" * 100)
    print()

    # Get status for all jobs
    job_ids = [job["job_id"] for job in job_data["jobs"] if job["job_id"] != "FAILED"]

    if not job_ids:
        print("No valid job IDs found.")
        return

    # Query bjobs for all job IDs at once
    try:
        result = subprocess.run(
            ["bjobs", "-a", "-o", "jobid stat exit_code delimiter='|'"] + job_ids,
            capture_output=True,
            text=True,
            check=False,
        )

        # Parse bjobs output
        job_status_map = {}
        lines = result.stdout.strip().split("\n")

        # Debug: print raw output
        if os.environ.get("DEBUG_BJOBS"):
            print("\nDEBUG: bjobs output:")
            print(result.stdout)
            print()

        for line in lines[1:]:  # Skip header
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    job_id = parts[0].strip()
                    status = parts[1].strip()
                    # Exit code might be empty, "-", or a number
                    exit_code = parts[2].strip() if len(parts) >= 3 else ""
                    # Normalize empty or "-" to "0" for DONE jobs
                    if status == "DONE" and (not exit_code or exit_code == "-"):
                        exit_code = "0"
                    job_status_map[job_id] = (status, exit_code)
    except Exception as e:
        print(f"Error querying job status: {e}")
        job_status_map = {}

    # Print status table
    print(f"{'Type':<15} {'Test Name':<50} {'Job ID':<15} {'Status':<12} {'Result':<15}")
    print("-" * 107)

    for job in job_data["jobs"]:
        job_type = job["type"]
        test_name = job["test_name"]
        job_id = job["job_id"]

        if job_id == "FAILED":
            status = "N/A"
            result_str = "SUBMIT FAILED"
        elif job_id in job_status_map:
            status, exit_code = job_status_map[job_id]

            if status == "DONE":
                if exit_code == "0":
                    result_str = "✓ SUCCESS"
                else:
                    result_str = f"✗ FAILED (exit {exit_code})"
            elif status == "EXIT":
                result_str = f"✗ FAILED (exit {exit_code})"
            elif status in ["PEND", "RUN"]:
                result_str = "RUNNING"
            else:
                result_str = status
        else:
            status = "UNKNOWN"
            result_str = "NOT FOUND"

        print(f"{job_type:<15} {test_name:<50} {job_id:<15} {status:<12} {result_str:<15}")

    print("-" * 107)
    print()

    # Summary statistics
    total_jobs = len(job_data["jobs"])
    failed_submits = sum(1 for job in job_data["jobs"] if job["job_id"] == "FAILED")

    completed = sum(
        1 for job in job_data["jobs"] if job["job_id"] in job_status_map and job_status_map[job["job_id"]][0] == "DONE"
    )
    successful = sum(
        1
        for job in job_data["jobs"]
        if job["job_id"] in job_status_map
        and job_status_map[job["job_id"]][0] == "DONE"
        and job_status_map[job["job_id"]][1] == "0"
    )
    failed = sum(
        1
        for job in job_data["jobs"]
        if job["job_id"] in job_status_map
        and (job_status_map[job["job_id"]][0] in ["DONE", "EXIT"])
        and job_status_map[job["job_id"]][1] != "0"
    )
    running = sum(
        1
        for job in job_data["jobs"]
        if job["job_id"] in job_status_map and job_status_map[job["job_id"]][0] in ["PEND", "RUN"]
    )

    print("Summary:")
    print(f"  Total jobs: {total_jobs}")
    print(f"  Completed: {completed}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Running/Pending: {running}")
    print(f"  Submit failures: {failed_submits}")
    print()
    print(f"Logs directory: {output_path}")
    print("=" * 100)


def main():
    # Determine script location and repository root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent  # scripts/run_lsf_integrationtest.py -> repo root
    default_test_file = repo_root / "integrationtests" / "test_base_set.py"

    parser = argparse.ArgumentParser(
        description="Submit Terratorch integration tests to LSF or check status of previous runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--branch-name", default="main", help="Git branch name to test (default: main)")
    parser.add_argument(
        "--python-version",
        default="py312",
        choices=["py310", "py311", "py312", "py313"],
        help="Python version for tox environments (default: py312)",
    )
    parser.add_argument("--output-dir", help="Output directory for storing test logs (required for submit mode)")
    parser.add_argument("--execution-tag", help="Tag for this execution (creates subfolder in output_dir)")
    parser.add_argument("--venv-base-dir", help="Path to virtual environment containing tox (required for submit mode)")
    parser.add_argument(
        "--test-file",
        default=str(default_test_file.relative_to(repo_root)),
        help=f"Path to the test file relative to repository root (default: {default_test_file.relative_to(repo_root)})",
    )
    parser.add_argument("--no-cleanup", action="store_true", help="Skip running the cleanup test")
    parser.add_argument(
        "--terratorch-tmp-root",
        metavar="PATH",
        help="Path to temporary root directory (sets TERRATORCH_TMP_ROOT environment variable for non-vLLM tests)",
    )
    parser.add_argument(
        "--check-status",
        metavar="OUTPUT_DIR",
        help="Check status of jobs from a previous run (provide the output directory path)",
    )

    args = parser.parse_args()

    # If checking status, do that and exit
    if args.check_status:
        check_job_status(args.check_status)
        return

    # Validate mandatory arguments with graceful error messages
    missing_args = []
    if not args.output_dir:
        missing_args.append("--output-dir")
    if not args.venv_base_dir:
        missing_args.append("--venv-base-dir")

    if missing_args:
        print("=" * 80)
        print("ERROR: Missing required arguments")
        print("=" * 80)
        print(f"The following required arguments are missing: {', '.join(missing_args)}")
        print()
        print("Usage example:")
        print(f"  python3 {sys.argv[0]} --branch-name main --output-dir /path/to/logs --venv-base-dir /path/to/venv")
        print()
        print("For full help, run:")
        print(f"  python3 {sys.argv[0]} --help")
        print("=" * 80)
        sys.exit(1)

    branch_name = args.branch_name
    python_version = args.python_version
    skip_cleanup = args.no_cleanup

    # Handle TERRATORCH_TMP_ROOT
    terratorch_tmp_root = args.terratorch_tmp_root

    # Generate execution tag if not provided
    if args.execution_tag:
        execution_tag = args.execution_tag
    else:
        execution_tag = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 80)
    print("LSF Integration Test Runner - Configuration Summary")
    print("=" * 80)
    print(f"Branch name: {branch_name}")
    print(f"Python version: {python_version}")
    print(f"Output directory: {args.output_dir}")
    print(f"Execution tag: {execution_tag}")
    print(f"Test file: {args.test_file}")
    print(f"Skip cleanup: {skip_cleanup}")
    if terratorch_tmp_root:
        print(f"TERRATORCH_TMP_ROOT: {terratorch_tmp_root}")
    print()

    # Validate venv_base_dir (mandatory)
    print("Step 1: Validating virtual environment...")
    venv_path = Path(args.venv_base_dir)

    if not venv_path.exists():
        print(f"Error: Virtual environment does not exist: {venv_path}")
        sys.exit(1)

    # Check if tox is installed in the venv
    tox_path = venv_path / "bin" / "tox"
    if not tox_path.exists():
        print(f"Error: tox is not installed in the virtual environment: {venv_path}")
        print(f"Expected tox at: {tox_path}")
        sys.exit(1)

    print(f"✓ Virtual environment validated: {venv_path}")
    print(f"✓ tox found at: {tox_path}")
    print()

    # Use current directory as full_path since tox will checkout the code
    full_path = Path.cwd()
    print(f"Step 2: Working directory: {full_path}")
    print()

    # Test file path - validate it exists
    print("Step 3: Validating test file...")
    test_file_path = Path(args.test_file)
    full_test_path = full_path / test_file_path

    if not full_test_path.exists():
        print(f"Error: Test file not found at {full_test_path}")
        print(f"Current directory: {Path.cwd()}")
        print(f"Full path: {full_path}")
        sys.exit(1)

    print(f"✓ Test file found: {full_test_path}")
    print()

    # Extract test names
    print("Step 4: Extracting test cases...")
    test_list = extract_test_names(full_test_path)
    print(f"✓ Found {len(test_list)} test cases:")
    for i, test_name in enumerate(test_list, 1):
        print(f"  {i}. {test_name}")
    print()

    # Create log directory from the provided output_dir
    print("Step 5: Setting up log directory...")
    base_log_dir = Path(args.output_dir).resolve()
    log_dir = base_log_dir / execution_tag

    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Logs will be stored in: {log_dir}")
    print()

    # Categorize tests
    print("Step 6: Categorizing test cases...")
    models_fit_test = None
    dependent_test_list = []
    independent_test_list = []
    cleanup_test = None

    for test_name in test_list:
        if test_name == PREREQUISITE_TEST:
            models_fit_test = test_name
        elif test_name == CLEANUP_TEST:
            cleanup_test = test_name
        elif test_name in DEPENDENT_TESTS:
            dependent_test_list.append(test_name)
        else:
            independent_test_list.append(test_name)

    total_tests = len(test_list) + len(VLLM_TESTS)
    print(f"✓ Categorized {total_tests} tests:")
    print(f"  - Prerequisite: {1 if models_fit_test else 0}")
    print(f"  - Dependent: {len(dependent_test_list)}")
    print(f"  - Independent: {len(independent_test_list)} (from test file) + {len(VLLM_TESTS)} (vLLM environments)")
    print(f"  - Cleanup: {1 if cleanup_test else 0}")
    print()

    # Track dependent job IDs for cleanup dependency
    dependent_job_ids = []

    # Prepare activation command and environment variables
    activate_cmd = f"source {venv_path / 'bin' / 'activate'}"
    user = os.environ.get("USER", "user")

    # Build environment variable export string for TERRATORCH_TMP_ROOT
    env_exports = ""
    if terratorch_tmp_root:
        env_exports = f" && export TERRATORCH_TMP_ROOT={terratorch_tmp_root}"

    # Track all submitted jobs for final summary table
    submitted_jobs = []

    print("=" * 80)
    print("Step 7: Submitting jobs to LSF")
    print("=" * 80)

    # Submit test_models_fit first
    if models_fit_test:
        print(f"\n[1/4] Submitting prerequisite test: {models_fit_test}")
        job_name = f"tt_{user}_{models_fit_test}"
        tox_work_dir = f".tox/{branch_name}_{models_fit_test}"
        command = f"/bin/bash -c 'set -e; {activate_cmd} && export TEST_BRANCH={branch_name} && export TEST_FUNCTION={models_fit_test} && export TOX_WORK_DIR={tox_work_dir}{env_exports} && tox -r -e integration-tests-base-set-{python_version}; exit $?'"

        models_fit_job_id = submit_lsf_job(
            job_name=job_name,
            log_file=log_dir / f"{models_fit_test}.log",
            err_file=log_dir / f"{models_fit_test}.err",
            command=command,
            gpu_config="num=1:mode=exclusive_process",
        )

        if models_fit_job_id:
            dependent_job_ids.append(models_fit_job_id)
            submitted_jobs.append(("Prerequisite", models_fit_test, models_fit_job_id, "None"))
            print(f"  ✓ Job submitted with ID: {models_fit_job_id}")
        else:
            submitted_jobs.append(("Prerequisite", models_fit_test, "FAILED", "None"))
            print(f"  ✗ Failed to submit job")

        # Submit dependent tests
        if dependent_test_list and models_fit_job_id:
            print(f"\n[2/4] Submitting {len(dependent_test_list)} dependent test(s) (will wait for {models_fit_test}):")
            for idx, test_name in enumerate(dependent_test_list, 1):
                print(f"  [{idx}/{len(dependent_test_list)}] Submitting: {test_name}")
                job_name = f"tt_{user}_{test_name}"
                tox_work_dir = f".tox/{branch_name}_{test_name}"
                command = f"/bin/bash -c 'set -e; {activate_cmd} && export TEST_BRANCH={branch_name} && export TEST_FUNCTION={test_name} && export TOX_WORK_DIR={tox_work_dir}{env_exports} && tox -r -e integration-tests-base-set-{python_version}; exit $?'"

                job_id = submit_lsf_job(
                    job_name=job_name,
                    log_file=log_dir / f"{test_name}.log",
                    err_file=log_dir / f"{test_name}.err",
                    command=command,
                    dependency=f"done({models_fit_job_id})",
                    terminate_on_dependency_failure=True,
                )

                if job_id:
                    dependent_job_ids.append(job_id)
                    submitted_jobs.append(("Dependent", test_name, job_id, models_fit_job_id))
                    print(f"      ✓ Job submitted with ID: {job_id}")
                else:
                    submitted_jobs.append(("Dependent", test_name, "FAILED", models_fit_job_id))
                    print(f"      ✗ Failed to submit job")

        # Submit cleanup test
        if cleanup_test and dependent_job_ids and not skip_cleanup:
            print(f"\n[3/4] Submitting cleanup test: {cleanup_test}")
            print(f"  Will wait for {len(dependent_job_ids)} job(s) to complete")
            job_name = f"tt_{user}_{cleanup_test}"

            # Build dependency condition
            cleanup_dependency = " && ".join([f"ended({jid})" for jid in dependent_job_ids])

            tox_work_dir = f".tox/{branch_name}_{cleanup_test}"
            command = f"/bin/bash -c 'set -e; {activate_cmd} && export TEST_BRANCH={branch_name} && export TEST_FUNCTION={cleanup_test} && export TOX_WORK_DIR={tox_work_dir}{env_exports} && tox -r -e integration-tests-base-set-{python_version}; exit $?'"

            cleanup_job_id = submit_lsf_job(
                job_name=job_name,
                log_file=log_dir / f"{cleanup_test}.log",
                err_file=log_dir / f"{cleanup_test}.err",
                command=command,
                dependency=cleanup_dependency,
            )

            if cleanup_job_id:
                submitted_jobs.append(("Cleanup", cleanup_test, cleanup_job_id, "All dependent"))
                print(f"  ✓ Cleanup test submitted with ID: {cleanup_job_id}")
            else:
                submitted_jobs.append(("Cleanup", cleanup_test, "FAILED", "All dependent"))
                print(f"  ✗ Failed to submit cleanup test")
        elif skip_cleanup:
            print(f"\n[3/4] Skipping cleanup test (--no-cleanup flag set)")

        # Submit independent tests from test file
        total_independent = len(independent_test_list) + len(VLLM_TESTS)
        if independent_test_list or VLLM_TESTS:
            print(f"\n[4/4] Submitting {total_independent} independent test(s) (run immediately):")

            # Submit tests from test file
            for idx, test_name in enumerate(independent_test_list, 1):
                print(f"  [{idx}/{total_independent}] Submitting: {test_name}")
                job_name = f"tt_{user}_{test_name}"
                tox_work_dir = f".tox/{branch_name}_{test_name}"
                command = f"/bin/bash -c 'set -e; {activate_cmd} && export TEST_BRANCH={branch_name} && export TEST_FUNCTION={test_name} && export TOX_WORK_DIR={tox_work_dir}{env_exports} && tox -r -e integration-tests-base-set-{python_version}; exit $?'"

                job_id = submit_lsf_job(
                    job_name=job_name,
                    log_file=log_dir / f"{test_name}.log",
                    err_file=log_dir / f"{test_name}.err",
                    command=command,
                )
                if job_id:
                    submitted_jobs.append(("Independent", test_name, job_id, "None"))
                    print(f"      ✓ Job submitted with ID: {job_id}")
                else:
                    submitted_jobs.append(("Independent", test_name, "FAILED", "None"))
                    print(f"      ✗ Failed to submit job")

            # Submit vLLM test environments
            for idx, tox_env in enumerate(VLLM_TESTS, len(independent_test_list) + 1):
                print(f"  [{idx}/{total_independent}] Submitting vLLM test: {tox_env}")
                job_name = f"tt_{user}_{tox_env}"
                tox_work_dir = f".tox/{branch_name}_{tox_env}"
                command = f"/bin/bash -c 'set -e; {activate_cmd} && export TOX_WORK_DIR={tox_work_dir} && tox -r -e {tox_env}-{python_version}; exit $?'"

                job_id = submit_lsf_job(
                    job_name=job_name,
                    log_file=log_dir / f"{tox_env}.log",
                    err_file=log_dir / f"{tox_env}.err",
                    command=command,
                    gpu_config="num=1:mode=exclusive_process",
                )
                if job_id:
                    submitted_jobs.append(("Independent", tox_env, job_id, "None"))
                    print(f"      ✓ Job submitted with ID: {job_id}")
                else:
                    submitted_jobs.append(("Independent", tox_env, "FAILED", "None"))
                    print(f"      ✗ Failed to submit job")
    else:
        print("\n" + "=" * 80)
        print("ERROR: test_models_fit not found in test suite")
        print("=" * 80)
        print(
            "Error: test_models_fit is a required prerequisite that creates checkpoints for dependent tests.",
            file=sys.stderr,
        )
        print()

        if dependent_test_list:
            print(f"⚠ Warning: Skipping {len(dependent_test_list)} dependent test(s) (require test_models_fit):")
            for test_name in dependent_test_list:
                print(f"  - {test_name}")
            print()

        if independent_test_list:
            print(f"ℹ Info: Submitting {len(independent_test_list)} independent test(s):")
            for idx, test_name in enumerate(independent_test_list, 1):
                print(f"  [{idx}/{len(independent_test_list)}] Submitting: {test_name}")
                job_name = f"tt_{user}_{test_name}"
                command = f"/bin/bash -c 'set -e; {activate_cmd} && export TEST_BRANCH={branch_name} && export TEST_FUNCTION={test_name}{env_exports} && tox -r -e integration-tests-base-set-{python_version}; exit $?'"

                submit_lsf_job(
                    job_name=job_name,
                    log_file=log_dir / f"{test_name}.log",
                    err_file=log_dir / f"{test_name}.err",
                    command=command,
                )
                print(f"      ✓ Job submitted")
        else:
            print("\nError: No independent tests found. Cannot proceed without test_models_fit.")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("Job Submission Complete")
    print("=" * 80)

    # Save job IDs to file
    if submitted_jobs:
        jobs_file = log_dir / "job_ids.json"
        job_data = {
            "submission_time": datetime.now().isoformat(),
            "branch": branch_name,
            "execution_tag": execution_tag,
            "jobs": [
                {"type": job_type, "test_name": test_name, "job_id": job_id, "dependency": dependency}
                for job_type, test_name, job_id, dependency in submitted_jobs
            ],
        }

        with jobs_file.open("w") as f:
            json.dump(job_data, f, indent=2)

        print(f"\n✓ Job IDs saved to: {jobs_file}")

    # Print summary table of all submitted jobs
    if submitted_jobs:
        print("\nSubmitted Jobs Summary:")
        print("-" * 100)
        print(f"{'Type':<15} {'Test Name':<50} {'Job ID':<15} {'Depends On':<20}")
        print("-" * 100)
        for job_type, test_name, job_id, dependency in submitted_jobs:
            print(f"{job_type:<15} {test_name:<50} {job_id:<15} {dependency:<20}")
        print("-" * 100)
        print(f"Total jobs submitted: {len(submitted_jobs)}")
        print()

    print(f"✓ Logs directory: {log_dir}")
    print(f"✓ Monitor jobs with: bjobs -J 'tt_{user}_*'")
    print(f"✓ Check status later with: python3 {sys.argv[0]} --check-status {log_dir}")
    print()
    print("Notes:")
    print("  • Dependent tests will only run if test_models_fit passes (exit code 0)")
    if skip_cleanup:
        print("  • Cleanup test skipped (--no-cleanup flag set)")
    elif cleanup_test:
        print("  • test_cleanup will run last, after all dependent tests complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
