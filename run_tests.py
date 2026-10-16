#!/usr/bin/env python3
"""SpecRAG Multi-Runner Test Execution Script.

Supports native execution on Windows Python 3.12 via both standard library
`unittest.TestLoader` and `pytest` (when installed).

Usage:
    python run_tests.py
    python run_tests.py --tier 1 -v
    python run_tests.py --tier all --json-output test_results.json
    python run_tests.py --runner unittest --failfast
"""

import argparse
import json
import os
import sys
import time
import unittest
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


TIER_FILE_MAP = {
    "1": "tests/test_tier1_features.py",
    "2": "tests/test_tier2_boundaries.py",
    "3": "tests/test_tier3_combinations.py",
    "4": "tests/test_tier4_workloads.py",
}


def print_banner():
    print("=" * 78)
    print("  SpecRAG Multi-Runner Test Suite (Tiers 1-4)")
    print("  Target: Native Windows Python 3.12 | Integrity Mode: Strict")
    print("=" * 78)


def run_with_pytest(
    tier: str,
    verbose: bool,
    failfast: bool,
    pattern: Optional[str],
) -> int:
    """Execute tests via pytest."""
    try:
        import pytest
    except ImportError:
        print("[WARN] pytest is not installed. Falling back to built-in unittest runner.\n")
        return -1

    args = []
    if verbose:
        args.append("-v")
    if failfast:
        args.append("-x")

    if tier == "all":
        args.extend(list(TIER_FILE_MAP.values()))
    elif tier == "full":
        args.append("tests")
    elif tier in TIER_FILE_MAP:
        args.append(TIER_FILE_MAP[tier])
    else:
        print(f"[ERROR] Unknown tier: {tier}")
        return 2

    if pattern:
        args.extend(["-k", pattern])

    print(f"[RUNNER] Executing via pytest with args: {' '.join(args)}")
    return pytest.main(args)


def run_with_unittest(
    tier: str,
    verbose: bool,
    failfast: bool,
    pattern: Optional[str],
) -> Dict[str, Any]:
    """Execute tests via standard library unittest."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    tests_dir = os.path.join(PROJECT_ROOT, "tests")

    if tier == "all":
        for t, fpath in TIER_FILE_MAP.items():
            rel_no_ext = os.path.splitext(fpath)[0].replace("/", ".").replace("\\", ".")
            discovered = loader.loadTestsFromName(rel_no_ext)
            suite.addTests(discovered)
    elif tier == "full":
        file_pattern = pattern if pattern else "test_*.py"
        discovered = loader.discover(start_dir=tests_dir, pattern=file_pattern)
        suite.addTests(discovered)
    elif tier in TIER_FILE_MAP:
        module_path = TIER_FILE_MAP[tier]
        rel_no_ext = os.path.splitext(module_path)[0].replace("/", ".").replace("\\", ".")
        discovered = loader.loadTestsFromName(rel_no_ext)
        suite.addTests(discovered)
    else:
        raise ValueError(f"Unknown tier: {tier}")

    total_tests = suite.countTestCases()
    verbosity_level = 2 if verbose else 1

    runner = unittest.TextTestRunner(
        verbosity=verbosity_level,
        failfast=failfast,
    )

    start_time = time.perf_counter()
    result = runner.run(suite)
    duration = time.perf_counter() - start_time

    passed = total_tests - len(result.failures) - len(result.errors) - len(result.skipped)
    pass_rate = (passed / total_tests * 100.0) if total_tests > 0 else 0.0

    return {
        "tier": tier,
        "runner": "unittest",
        "total": total_tests,
        "passed": passed,
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "duration_seconds": round(duration, 3),
        "pass_rate_percent": round(pass_rate, 2),
        "success": result.wasSuccessful(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="SpecRAG Multi-Runner Test Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "4", "all", "full"],
        default="all",
        help="Select which test tier to execute.",
    )
    parser.add_argument(
        "--runner",
        choices=["auto", "pytest", "unittest"],
        default="auto",
        help="Test runner backend engine.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed verbose output.",
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop on first test failure or error.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Test name or filename glob filter pattern.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="File path to save JSON execution summary.",
    )

    args = parser.parse_args()

    print_banner()
    print(f"Selection: Tier={args.tier} | Runner={args.runner} | Verbose={args.verbose}\n")

    # Determine runner
    use_pytest = False
    if args.runner == "pytest":
        use_pytest = True
    elif args.runner == "auto":
        try:
            import pytest  # noqa: F401
            use_pytest = True
        except ImportError:
            use_pytest = False

    exit_code = 0
    metrics: Dict[str, Any] = {}

    if use_pytest:
        pytest_ret = run_with_pytest(
            tier=args.tier,
            verbose=args.verbose,
            failfast=args.failfast,
            pattern=args.pattern,
        )
        if pytest_ret == -1:
            # Fallback to unittest
            use_pytest = False
        else:
            exit_code = 0 if pytest_ret == 0 else 1
            metrics = {
                "tier": args.tier,
                "runner": "pytest",
                "exit_code": exit_code,
                "success": exit_code == 0,
            }

    if not use_pytest:
        try:
            metrics = run_with_unittest(
                tier=args.tier,
                verbose=args.verbose,
                failfast=args.failfast,
                pattern=args.pattern,
            )
            exit_code = 0 if metrics["success"] else 1
        except Exception as e:
            print(f"[FATAL] Test execution failed with unhandled exception: {e}")
            sys.exit(1)

    # Print summary block
    print("\n" + "=" * 78)
    print("  EXECUTION SUMMARY")
    print("=" * 78)
    if "total" in metrics:
        print(f"  Tier Target      : Tier {metrics['tier']}")
        print(f"  Runner Engine    : {metrics['runner']}")
        print(f"  Total Tests Run  : {metrics['total']}")
        print(f"  Passed           : {metrics['passed']}")
        print(f"  Failed           : {metrics['failed']}")
        print(f"  Errors           : {metrics['errors']}")
        print(f"  Skipped          : {metrics['skipped']}")
        print(f"  Duration         : {metrics['duration_seconds']}s")
        print(f"  Pass Rate        : {metrics['pass_rate_percent']}%")
        print(f"  Final Status     : {'PASSED (OK)' if metrics['success'] else 'FAILED'}")
    else:
        print(f"  Runner Engine    : pytest")
        print(f"  Exit Code        : {exit_code}")
        print(f"  Final Status     : {'PASSED (OK)' if exit_code == 0 else 'FAILED'}")
    print("=" * 78 + "\n")

    # Export JSON if requested
    if args.json_output:
        try:
            json_dir = os.path.dirname(os.path.abspath(args.json_output))
            os.makedirs(json_dir, exist_ok=True)
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"[REPORT] Saved JSON metrics to: {args.json_output}\n")
        except Exception as e:
            print(f"[WARN] Failed to write JSON output to {args.json_output}: {e}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
