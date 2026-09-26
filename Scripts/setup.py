#!/usr/bin/env python3
"""Solvers CMake Build Configuration Script.

Ported to follow KhwarizmiAnalytix/XSigma's Scripts/setup.py architecture
exactly (ErrorLogger + BuildDirectoryDetector + SummaryReporter, dotted-token
CLI, suggestion-rich error logging, robust parse_args, per-phase timers), but
scoped to the CMake options this standalone, single-module repo actually
defines. Solvers is a single flat CMake target, so there is no multi-module
fan-out, no --project.* scoping, and none of XSigma's GPU/MKL/TBB/mimalloc
backends: setup.py tokens map 1:1 to the SOLVERS_ENABLE_*/BUILD_SHARED_LIBS
options declared in CMakeLists.txt, plus the cppcheck/valgrind analysis tokens
that drive helper scripts rather than CMake cache variables.

Usage:
    python Scripts/setup.py config.build.test
    python Scripts/setup.py config.build.test.coverage
    python Scripts/setup.py config.build.test.nlopt.ceres
    python Scripts/setup.py config.build.test.gcc.release
"""

import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import colorama
    from colorama import Fore, Style

    colorama.init()
except ImportError:  # Windows CLI smoke and some CI jobs skip pip install

    class Fore:  # pylint: disable=too-few-public-methods
        CYAN = GREEN = YELLOW = RED = WHITE = ""

    class Style:  # pylint: disable=too-few-public-methods
        RESET_ALL = ""


# Import helper modules
from helpers import (
    build as build_helper,
    config as config_helper,
    cppcheck as cppcheck_helper,
    test as test_helper,
)

DEBUG_FLAG = False


class ErrorLogger:
    """Centralized error logging system for comprehensive error tracking."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"solvers_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.errors = []

    def log_error(
        self,
        command: str,
        error_output: str,
        context: str = "",
        suggestions: Optional[list[str]] = None,
    ):
        """Log a comprehensive error with context and suggestions."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        error_entry = {
            "timestamp": timestamp,
            "command": command,
            "error_output": error_output,
            "context": context,
            "suggestions": suggestions or [],
        }

        self.errors.append(error_entry)

        # Write to log file immediately
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"ERROR LOG ENTRY - {timestamp}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Context: {context}\n")
            f.write(f"Command: {command}\n")
            f.write(f"Error Output:\n{error_output}\n")
            if suggestions:
                f.write("Troubleshooting Suggestions:\n")
                for i, suggestion in enumerate(suggestions, 1):
                    f.write(f"  {i}. {suggestion}\n")
            f.write(f"{'=' * 80}\n\n")

    def get_log_file_path(self) -> str:
        """Get the path to the current log file."""
        return str(self.log_file)

    def has_errors(self) -> bool:
        """Check if any errors have been logged."""
        return len(self.errors) > 0


class BuildDirectoryDetector:
    """Utility to detect build directories dynamically, independent of naming conventions."""

    @staticmethod
    def find_build_directories(source_path: str) -> list[Path]:
        """Find all potential build directories in the project root."""
        source_root = Path(source_path)
        build_dirs = set()  # Use set to avoid duplicates

        # Common build directory patterns
        patterns = [
            "build*",
            "*build*",
        ]

        for pattern in patterns:
            matches = list(source_root.glob(pattern))
            for match in matches:
                if match.is_dir() and BuildDirectoryDetector._is_build_directory(match):
                    build_dirs.add(match)

        # Convert back to list and sort by modification time (most recent first)
        build_dirs_list = list(build_dirs)
        build_dirs_list.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return build_dirs_list

    @staticmethod
    def _is_build_directory(path: Path) -> bool:
        """Check if a directory looks like a CMake build directory."""
        cmake_indicators = [
            "CMakeCache.txt",
            "cmake_install.cmake",
            "CMakeFiles",
            "Makefile",
            "build.ninja",
        ]

        return any((path / indicator).exists() for indicator in cmake_indicators)

    @staticmethod
    def find_best_build_directory(source_path: str, preferred_name: Optional[str] = None) -> Optional[Path]:
        """Find the best build directory, optionally preferring a specific name."""
        build_dirs = BuildDirectoryDetector.find_build_directories(source_path)

        if not build_dirs:
            return None

        # If a preferred name is specified, look for it first
        if preferred_name:
            for build_dir in build_dirs:
                if preferred_name in build_dir.name:
                    return build_dir

        # Return the most recently modified build directory
        return build_dirs[0]


class SummaryReporter:
    """Generate and display summary reports for various analysis tools."""

    def __init__(self):
        self.reports = {}

    def add_cppcheck_report(self, log_file: str, exit_code: int):
        """Add cppcheck analysis results to the summary."""
        if not os.path.exists(log_file):
            self.reports["cppcheck"] = {
                "status": "not_run",
                "message": "Cppcheck was not executed",
            }
            return

        try:
            with open(log_file, encoding="utf-8") as f:
                content = f.read()

            # Count issues by severity
            issues = {
                "error": len(re.findall(r",error,", content)),
                "warning": len(re.findall(r",warning,", content)),
                "style": len(re.findall(r",style,", content)),
                "performance": len(re.findall(r",performance,", content)),
                "portability": len(re.findall(r",portability,", content)),
                "information": len(re.findall(r",information,", content)),
            }

            total_issues = sum(issues.values())

            self.reports["cppcheck"] = {
                "status": "completed",
                "exit_code": exit_code,
                "total_issues": total_issues,
                "issues_by_type": issues,
                "log_file": log_file,
            }
        except Exception as e:
            self.reports["cppcheck"] = {
                "status": "error",
                "message": f"Failed to parse cppcheck results: {e}",
            }

    def add_valgrind_report(self, build_path: str, exit_code: int):
        """Add valgrind analysis results to the summary."""
        valgrind_logs = glob.glob(os.path.join(build_path, "Testing", "Temporary", "MemoryChecker.*.log"))

        if not valgrind_logs:
            self.reports["valgrind"] = {
                "status": "not_run",
                "message": "Valgrind was not executed or no logs found",
            }
            return

        try:
            memory_leaks = 0
            memory_errors = 0

            for log_file in valgrind_logs:
                with open(log_file, encoding="utf-8") as f:
                    content = f.read()

                # Count memory issues
                leak_matches = re.findall(r"definitely lost: (\d+)", content)
                memory_leaks += sum(int(match) for match in leak_matches if int(match) > 0)

                error_matches = re.findall(r"ERROR SUMMARY: (\d+) errors", content)
                memory_errors += sum(int(match) for match in error_matches if int(match) > 0)

            self.reports["valgrind"] = {
                "status": "completed",
                "exit_code": exit_code,
                "memory_leaks": memory_leaks,
                "memory_errors": memory_errors,
                "log_files": valgrind_logs,
            }
        except Exception as e:
            self.reports["valgrind"] = {
                "status": "error",
                "message": f"Failed to parse valgrind results: {e}",
            }

    def add_coverage_report(self, build_path: str, exit_code: int):
        """Add coverage analysis results to the summary.

        Parses JSON coverage reports and extracts metrics including line coverage,
        function coverage, and region coverage.
        """
        import json

        # Look for coverage JSON files (primary source)
        coverage_json_paths = [
            os.path.join(build_path, "coverage_report", "coverage_summary.json"),
            os.path.join(build_path, "coverage_report", "coverage.json"),
        ]

        coverage_json = None
        for path in coverage_json_paths:
            if os.path.exists(path):
                coverage_json = path
                break

        if not coverage_json:
            self.reports["coverage"] = {
                "status": "not_run",
                "message": "Coverage report not found",
            }
            return

        try:
            with open(coverage_json, encoding="utf-8") as f:
                coverage_data = json.load(f)

            # Extract metrics from global_metrics (preferred format)
            if "global_metrics" in coverage_data:
                metrics = coverage_data["global_metrics"]
                self.reports["coverage"] = {
                    "status": "completed",
                    "exit_code": exit_code,
                    "total_lines": metrics.get("total_lines", 0),
                    "covered_lines": metrics.get("covered_lines", 0),
                    "line_coverage_percent": metrics.get("line_coverage_percent", 0.0),
                    "total_functions": metrics.get("total_functions", 0),
                    "covered_functions": metrics.get("covered_functions", 0),
                    "function_coverage_percent": metrics.get("function_coverage_percent", 0.0),
                    "total_regions": metrics.get("total_regions", 0),
                    "covered_regions": metrics.get("covered_regions", 0),
                    "region_coverage_percent": metrics.get("region_coverage_percent", 0.0),
                    "report_file": coverage_json,
                }
            # Extract metrics from summary (alternative format)
            elif "summary" in coverage_data:
                summary = coverage_data["summary"]
                line_cov = summary.get("line_coverage", {})
                func_cov = summary.get("function_coverage", {})

                self.reports["coverage"] = {
                    "status": "completed",
                    "exit_code": exit_code,
                    "total_lines": line_cov.get("total", 0),
                    "covered_lines": line_cov.get("covered", 0),
                    "line_coverage_percent": line_cov.get("percent", 0.0),
                    "total_functions": func_cov.get("total", 0),
                    "covered_functions": func_cov.get("covered", 0),
                    "function_coverage_percent": func_cov.get("percent", 0.0),
                    "total_regions": 0,
                    "covered_regions": 0,
                    "region_coverage_percent": 0.0,
                    "report_file": coverage_json,
                }
            else:
                self.reports["coverage"] = {
                    "status": "error",
                    "message": "Coverage JSON format not recognized",
                }
        except Exception as e:
            self.reports["coverage"] = {
                "status": "error",
                "message": f"Failed to parse coverage results: {e}",
            }

    def display_summary(self):
        """Display a comprehensive summary of all analysis results."""
        if not self.reports:
            return

        print_status("\n" + "=" * 80, "INFO")
        print_status("BUILD AND ANALYSIS SUMMARY REPORT", "INFO")
        print_status("=" * 80, "INFO")

        for tool, report in self.reports.items():
            self._display_tool_summary(tool, report)

        print_status("=" * 80, "INFO")

    def _display_tool_summary(self, tool: str, report: dict):
        """Display summary for a specific tool."""
        tool_name = tool.upper()

        if report["status"] == "not_run":
            print_status(f"{tool_name}: Not executed", "INFO")
            return

        if report["status"] == "error":
            print_status(f"{tool_name}: Error - {report['message']}", "ERROR")
            return

        if tool == "cppcheck":
            total = report["total_issues"]
            if total == 0:
                print_status(f"{tool_name}: No issues found", "SUCCESS")
            else:
                print_status(f"{tool_name}: Found {total} issues", "WARNING")
                for issue_type, count in report["issues_by_type"].items():
                    if count > 0:
                        print_status(f"  - {issue_type}: {count}", "INFO")
                print_status(f"  Log file: {report['log_file']}", "INFO")

        elif tool == "valgrind":
            leaks = report["memory_leaks"]
            errors = report["memory_errors"]
            if leaks == 0 and errors == 0:
                print_status(f"{tool_name}: No memory issues found", "SUCCESS")
            else:
                if leaks > 0:
                    print_status(f"{tool_name}: Found {leaks} memory leaks", "ERROR")
                if errors > 0:
                    print_status(f"{tool_name}: Found {errors} memory errors", "ERROR")

        elif tool == "coverage":
            print_status("\n" + "=" * 80, "INFO")
            print_status("CODE COVERAGE SUMMARY", "INFO")
            print_status("=" * 80, "INFO")

            total_lines = report.get("total_lines", 0)
            covered_lines = report.get("covered_lines", 0)
            coverage_percent = report.get("line_coverage_percent", 0.0)

            print_status(f"Total Lines:    {total_lines}", "INFO")
            print_status(f"Covered Lines:  {covered_lines}", "INFO")
            print_status(
                f"Coverage:       {coverage_percent:.2f}%",
                "SUCCESS" if coverage_percent >= 95.0 else "WARNING" if coverage_percent >= 80.0 else "ERROR",
            )

            if report.get("total_functions", 0) > 0:
                func_coverage = report.get("function_coverage_percent", 0.0)
                print_status(f"Function Coverage: {func_coverage:.2f}%", "INFO")

            if report.get("total_regions", 0) > 0:
                region_coverage = report.get("region_coverage_percent", 0.0)
                print_status(f"Region Coverage:   {region_coverage:.2f}%", "INFO")

            report_file = report.get("report_file", "")
            if report_file:
                report_dir = os.path.dirname(report_file)
                html_report_paths = [
                    os.path.join(report_dir, "html", "index.html"),
                    os.path.join(report_dir, "index.html"),
                ]

                for html_path in html_report_paths:
                    if os.path.exists(html_path):
                        print_status(f"\nHTML Report: {html_path}", "INFO")
                        break

            print_status("=" * 80, "INFO")


def check_dependencies() -> list[str]:
    """Check if required dependencies are installed."""
    missing_deps = []

    try:
        import psutil  # noqa: F401
    except ImportError:
        missing_deps.append("psutil")

    # Check for CMake
    try:
        subprocess.run(["cmake", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing_deps.append("CMake")

    # Check for compiler
    if platform.system() == "Windows":
        try:
            subprocess.run(["clang", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(["cl"], capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_deps.append("C++ compiler (MSVC or Clang)")
    elif platform.system() == "Darwin":
        # Check for Xcode command line tools
        try:
            subprocess.run(["xcode-select", "--print-path"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_deps.append("Xcode Command Line Tools (run: xcode-select --install)")

        # Check for clang++ (should be available with Xcode tools)
        try:
            subprocess.run(["clang++", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_deps.append("C++ compiler (Clang)")
    else:
        try:
            subprocess.run(["clang++", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(["g++", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_deps.append("C++ compiler (GCC or Clang)")

    return missing_deps


def check_xcode_availability() -> bool:
    """Check if Xcode is available on macOS."""
    if platform.system() != "Darwin":
        return False

    try:
        result = subprocess.run(["xcodebuild", "-version"], capture_output=True, check=True, text=True)
        print_status(f"Found Xcode: {result.stdout.strip().split()[1]}", "INFO")
        return True
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr if isinstance(e.stderr, str) else e.stderr.decode() if e.stderr else ""
        if "command line tools instance" in stderr_output:
            print_status(
                "Xcode Command Line Tools found, but full Xcode is required for Xcode generator",
                "WARNING",
            )
            print_status(
                "Install Xcode from the App Store or use 'sudo xcode-select -s /Applications/Xcode.app'",
                "INFO",
            )
        else:
            print_status(f"Xcode check failed: {stderr_output.strip()}", "WARNING")
        return False
    except FileNotFoundError:
        print_status(
            "xcodebuild not found - install Xcode or Xcode Command Line Tools",
            "WARNING",
        )
        return False


def check_xcode_installation() -> bool:
    """Check if Xcode app is installed but not configured."""
    if platform.system() != "Darwin":
        return False

    xcode_path = "/Applications/Xcode.app"
    if os.path.exists(xcode_path):
        print_status(f"Found Xcode at {xcode_path}", "INFO")
        print_status(
            "Try running: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer",
            "INFO",
        )

        # Ask user if they want to configure Xcode automatically
        try:
            response = input("Would you like to configure Xcode automatically? (y/N): ").strip().lower()
            if response in ["y", "yes"]:
                try:
                    subprocess.run(
                        [
                            "sudo",
                            "xcode-select",
                            "-s",
                            "/Applications/Xcode.app/Contents/Developer",
                        ],
                        check=True,
                    )
                    print_status("Xcode configured successfully", "SUCCESS")
                    return True
                except subprocess.CalledProcessError as e:
                    print_status(f"Failed to configure Xcode: {e}", "ERROR")
        except (KeyboardInterrupt, EOFError):
            print_status("\nSkipping Xcode configuration", "INFO")

        return True
    return False


def print_status(message: str, status: str = "INFO", end: str = "\n") -> None:
    """Print a formatted status message."""
    status_colors = {
        "INFO": Fore.BLUE,
        "SUCCESS": Fore.GREEN,
        "ERROR": Fore.RED,
        "WARNING": Fore.YELLOW,
    }
    color = status_colors.get(status, Fore.WHITE)
    print(f"{color}[{status}]{Style.RESET_ALL} {message}", end=end)


def get_logical_processor_count():
    try:
        import psutil

        return psutil.cpu_count(logical=True)
    except ImportError:
        # Fallback methods if psutil is not available
        try:
            return os.cpu_count()
        except AttributeError:
            import multiprocessing

            return multiprocessing.cpu_count()


def debug_print(message):
    if DEBUG_FLAG:
        print(message)


class SolversFlags:
    """Maps setup.py dotted tokens to Solvers's CMake cache variables.

    Follows XSigma's XSigmaFlags model (token registry + inverse-default logic +
    validation), but scoped 1:1 to the seven options CMakeLists.txt actually
    defines. Solvers is a single flat CMake target, so there is no per-module
    fan-out: recognized tokens either map straight to a -D flag (see __name) or
    drive a helper script (cppcheck/valgrind) with no CMake variable behind them.
    """

    OFF = "OFF"
    ON = "ON"

    def __init__(self, arg_list):
        self.__initialize_flags()
        if arg_list:
            self.__build_cmake_flag()
            self.__fill_option_flags(arg_list)
            self.__validate_flags()

    def __initialize_flags(self):
        self.__key = [
            # CMake options declared in CMakeLists.txt
            "static",
            "test",
            "ceres",
            "nlopt",
            "sanitizer",
            "sanitizer_enum",
            "coverage",
            "clangtidy",
            # Helper-driven analysis tokens (no CMake option behind them)
            "valgrind",
            "cppcheck",
        ]
        self.__description = [
            "build shared (default) or static libraries",
            "build the Solvers GoogleTest suite (SOLVERS_ENABLE_TESTING; default ON)",
            "build the optional Ceres backend (SOLVERS_ENABLE_CERES)",
            "build the optional NLopt backend (SOLVERS_ENABLE_NLOPT)",
            "enable address+undefined sanitizers (SOLVERS_ENABLE_SANITIZER)",
            "sanitizer type for the test runner: address, undefined, thread, memory, leak",
            "enable code coverage instrumentation (SOLVERS_ENABLE_COVERAGE)",
            "enable clang-tidy static analysis (SOLVERS_ENABLE_CLANGTIDY)",
            "execute the test suite under Valgrind",
            "run cppcheck static analysis after the build",
        ]

    def __build_cmake_flag(self):
        debug_print("Build cmake flag")
        # Only tokens that correspond to a real option() in CMakeLists.txt appear
        # here. sanitizer_enum feeds the ctest runner (SOLVERS_ENABLE_SANITIZER is
        # hardcoded to address,undefined in CMake), and valgrind/cppcheck run via
        # Scripts/helpers/*, so none of those emit a -D flag.
        self.__name = {
            "static": "BUILD_SHARED_LIBS",
            "test": "SOLVERS_ENABLE_TESTING",
            "ceres": "SOLVERS_ENABLE_CERES",
            "nlopt": "SOLVERS_ENABLE_NLOPT",
            "sanitizer": "SOLVERS_ENABLE_SANITIZER",
            "coverage": "SOLVERS_ENABLE_COVERAGE",
            "clangtidy": "SOLVERS_ENABLE_CLANGTIDY",
        }

    def __fill_option_flags(self, arg_list):
        debug_print("Fill option flags")
        self.__value = {}
        self.__set_default_flags()
        self.__process_arg_list(arg_list)

    def __set_default_flags(self):
        # Mirrors CMakeLists.txt's option() defaults. All flags start OFF; the
        # ones CMake defaults ON (or that setup.py drives ON) are set below.
        self.__value = dict.fromkeys(self.__key, self.OFF)
        self.__value.update(
            {
                # "static" holds the BUILD_SHARED_LIBS value directly: ON = shared
                # (setup.py default), passing the `static` token flips it to OFF.
                "static": self.ON,
                "test": self.ON,  # SOLVERS_ENABLE_TESTING default ON (top-level)
                "sanitizer_enum": "address",
            }
        )

    def __process_arg_list(self, arg_list):
        sanitizer_list = ["address", "undefined", "thread", "memory", "leak"]

        self.builder_suffix = ""
        for arg in arg_list:
            if arg == "static":
                self.__value["static"] = self.OFF
                self.builder_suffix += "_static"
            elif arg in sanitizer_list:
                self.__value["sanitizer"] = self.ON
                self.__value["sanitizer_enum"] = arg
                self.builder_suffix += f"_{arg}"
            elif arg in self.__key:
                # Solvers has no "default ON → token disables" flags (unlike
                # XSigma's gtest/mimalloc/cache); every recognized token turns
                # its flag ON.
                self.__value[arg] = self.ON

                # Add to builder suffix (except for command-phase tokens).
                if arg not in ("test", "build"):
                    self.builder_suffix += f"_{arg}"

    def __validate_flags(self):
        """Validate flag combinations and warn about potential issues."""
        if self.__value.get("sanitizer") == self.ON and self.__value.get("valgrind") == self.ON:
            print_status(
                "Both sanitizer and valgrind enabled - consider using only one.",
                "WARNING",
            )

        if self.__value.get("coverage") == self.ON and self.__value.get("test") != self.ON:
            print_status(
                "Coverage enabled but testing is disabled - enabling tests automatically.",
                "WARNING",
            )
            self.__value["test"] = self.ON

        if self.__value.get("coverage") == self.ON:
            try:
                import coverage_tool  # noqa: F401
            except ImportError:
                print_status(
                    "coverage-tool is not installed. Install with: pip install coverage-tool",
                    "ERROR",
                )
                sys.exit(1)

    @staticmethod
    def find_case_insensitive(element, lst):
        element_lower = element.lower()
        return next((item for item in lst if element_lower == item.lower()), None)

    def create_cmake_flags(self, cmake_cmd_flags, build_enum, system):
        debug_print("Create cmake flags")
        del system  # unused; kept for parity with XSigma's signature

        # Sanitizer, valgrind, and coverage all need a debug build to be useful.
        if (
            self.__value.get("valgrind") == self.ON
            or self.__value.get("sanitizer") == self.ON
            or self.__value.get("coverage") == self.ON
        ):
            print_status(
                "Enabling debug build for sanitizer, valgrind, or coverage analysis",
                "INFO",
            )
            build_type = "DEBUG"
        else:
            build_type = str(build_enum).capitalize()

        # Single flat target: emit one -D per token that maps to a real option.
        for key, value in self.__value.items():
            if key in self.__name:
                flag_name = self.__name[key]
                flag_value = "ON" if isinstance(value, bool) and value else str(value)
                if flag_value != "":
                    cmake_cmd_flags.append(f"-D{flag_name}={flag_value}")

        # Add compilation database generation flag
        cmake_cmd_flags.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")

        return build_type

    def helper(self):
        for key, description in zip(self.__key, self.__description):
            if key == "sanitizer":
                key = "sanitizer (or --sanitizer.TYPE)"
            elif key == "sanitizer_enum":
                key = "address, undefined, thread, memory, leak"
            print(f"{key:<30}{description}")

    def is_coverage(self):
        return self.__value["coverage"] == self.ON

    def is_valgrind(self):
        return self.__value["valgrind"] == self.ON

    def is_cppcheck(self):
        return self.__value["cppcheck"] == self.ON

    def get_sanitizer_type(self):
        """Get the current sanitizer type if enabled, None otherwise."""
        if self.__value.get("sanitizer") == self.ON:
            return self.__value.get("sanitizer_enum")
        return None


class SolversConfiguration:
    def __init__(self, args_list):
        # Check dependencies first
        missing_deps = check_dependencies()
        if missing_deps:
            print_status("Missing required dependencies:", "ERROR")
            for dep in missing_deps:
                print_status(f"  - {dep}", "ERROR")
            print_status("Please install missing dependencies and try again.", "ERROR")
            sys.exit(1)

        # Initialize utilities
        self.error_logger = ErrorLogger()
        self.summary_reporter = SummaryReporter()

        self.__initialize_values()
        self.__solvers_flags = SolversFlags(args_list)
        self.__fill_compilation_flags(args_list)

    def __initialize_values(self):
        # Set default compiler to Clang on all platforms
        default_cxx_compiler = "clang++"
        default_c_compiler = "clang"

        self.__value = {
            "system": platform.system(),
            "build_folder": "build_ninja",
            "builder": "ninja",
            "config": "",
            "build": "",
            "test": "",
            "build_enum": "Release",
            "cmake_generator": "Ninja",
            "cmake_cxx_compiler": f"-DCMAKE_CXX_COMPILER={default_cxx_compiler}",
            "cmake_c_compiler": f"-DCMAKE_C_COMPILER={default_c_compiler}",
            "compiler_flags": "--debug-trycompile",
            "verbosity": "",
            "arg_cmake_verbose": "--loglevel=NOTICE",
        }
        self.__compiler_user_specified = False
        print(f"================= {self.__value['system']} platform =================")

    def __fill_compilation_flags(self, args_list):
        debug_print("Fill Compilation flags")
        for arg in args_list:
            self.__process_arg(arg)

    def __process_arg(self, arg):
        if arg == "ninja":
            self.__set_ninja_flags()
        elif arg == "xcode":
            self.__set_xcode_flags()
        elif self.__is_clang_compiler(arg):
            self.__set_clang_compiler(arg)
        elif arg == "clang-cl":
            self.__value["cmake_cxx_compiler"] = "-DCMAKE_GENERATOR_TOOLSET=ClangCL"
            self.__value["cmake_c_compiler"] = ""
            self.__compiler_user_specified = True
        elif self.__is_gcc_compiler(arg):
            self.__set_gcc_compiler(arg)
        elif self.__is_visual_studio(arg):
            self.__set_visual_studio(arg)
        elif arg in ["config", "build", "test"]:
            self.__value[arg] = arg
        elif arg in ["release", "debug", "relwithdebinfo"]:
            self.__value["build_enum"] = arg.capitalize()
        elif arg in ["vv", "v"]:
            self.__set_verbose_flags()

    def __set_ninja_flags(self):
        self.__value["cmake_generator"] = "Ninja"
        self.__value["builder"] = "ninja"
        self.__value["build_folder"] = f"build_ninja{self.__solvers_flags.builder_suffix}"

    def __set_xcode_flags(self):
        if self.__value["system"] == "Darwin":
            if check_xcode_availability():
                self.__value["cmake_generator"] = "Xcode"
                self.__value["builder"] = "xcodebuild"
                self.__value["build_folder"] = f"build_xcode{self.__solvers_flags.builder_suffix}"
                self.__value["compiler_flags"] = ""
                if not self.__compiler_user_specified:
                    self.__value["cmake_c_compiler"] = ""
                    self.__value["cmake_cxx_compiler"] = ""
                print_status("Using Xcode generator", "SUCCESS")
            else:
                print_status("Xcode not found, falling back to Ninja", "WARNING")
                if check_xcode_installation():
                    print_status(
                        "Xcode appears to be installed but not configured properly",
                        "INFO",
                    )
                self.__set_ninja_flags()
        else:
            print_status("Xcode generator is only available on macOS", "WARNING")
            self.__set_ninja_flags()

    def __is_clang_compiler(self, arg):
        return "clang" in arg and arg not in ["clang-cl", "clangtidy"]

    def __set_clang_compiler(self, arg):
        self.__value["cmake_c_compiler"] = f"-DCMAKE_C_COMPILER={arg}"
        self.__value["cmake_cxx_compiler"] = f"-DCMAKE_CXX_COMPILER={arg.replace('clang', 'clang++')}"
        self.__compiler_user_specified = True

    def __is_gcc_compiler(self, arg):
        """Check if argument is a GCC compiler specification (gcc, g++, gcc-11, g++-11, etc.)"""
        return ("gcc" in arg or "g++" in arg) and arg not in ["cppcheck"]

    def __set_gcc_compiler(self, arg):
        """Set GCC compiler for CMake configuration"""
        if "g++" in arg:
            self.__value["cmake_cxx_compiler"] = f"-DCMAKE_CXX_COMPILER={arg}"
            c_compiler = arg.replace("g++", "gcc")
            self.__value["cmake_c_compiler"] = f"-DCMAKE_C_COMPILER={c_compiler}"
        else:
            self.__value["cmake_c_compiler"] = f"-DCMAKE_C_COMPILER={arg}"
            cxx_compiler = arg.replace("gcc", "g++")
            self.__value["cmake_cxx_compiler"] = f"-DCMAKE_CXX_COMPILER={cxx_compiler}"
        self.__compiler_user_specified = True

    def __is_visual_studio(self, arg):
        return arg in ["vs17", "vs19", "vs22", "vs26"] and self.__value["system"] == "Windows"

    def __set_visual_studio(self, arg):
        vs_versions = {
            "vs17": ("Visual Studio 15 2017 Win64", "build_vs17"),
            "vs19": ("Visual Studio 16 2019", "build_vs19"),
            "vs22": ("Visual Studio 17 2022", "build_vs22"),
            "vs26": ("Visual Studio 18 2026", "build_vs26"),
        }
        self.__value["compiler_flags"] = "-A x64"
        self.__value["cmake_generator"], base_build_folder = vs_versions[arg]
        self.__value["builder"] = "cmake"
        self.__value["build_folder"] = f"{base_build_folder}{self.__solvers_flags.builder_suffix}"
        if not self.__compiler_user_specified:
            # Let Visual Studio decide the native MSVC toolchain unless overridden.
            self.__value["cmake_cxx_compiler"] = ""
            self.__value["cmake_c_compiler"] = ""

    def __set_verbose_flags(self):
        self.__value["arg_cmake_verbose"] = "--loglevel=VERBOSE"
        self.__value["verbosity"] = "-VV"

    def config(self, source_path, build_path):
        if self.__value["config"] != "config":
            return 0

        print_status("Configuring build...", "INFO")
        try:
            cmake_flags = []
            self.__value["build_enum"] = self.__solvers_flags.create_cmake_flags(
                cmake_flags, self.__value["build_enum"], self.__value["system"]
            )
            print(f"build enum: {self.__value['build_enum']}")
            cmake_flags.append(f"-DCMAKE_BUILD_TYPE={self.__value['build_enum']}")

            generator_toolset = self.__value.get("generator_toolset")
            exit_code = config_helper.configure_build(
                source_path,
                build_path,
                self.__value["cmake_generator"],
                self.__value["cmake_cxx_compiler"],
                self.__value["cmake_c_compiler"],
                cmake_flags,
                self.__value["arg_cmake_verbose"],
                self.__shell_flag(),
                generator_toolset=generator_toolset,
            )

            if exit_code == 0:
                print_status("Build configured successfully", "SUCCESS")
                if self.__value["cmake_generator"] == "Xcode":
                    config_helper.handle_xcode_project_opening()
            else:
                print_status("Configuration failed", "ERROR")
                sys.exit(1)

        except subprocess.CalledProcessError as e:
            suggestions = [
                "Check if CMake is properly installed",
                "Verify all required dependencies are available",
                "Check if the generator is supported on your system",
                "Try a different build generator (e.g., ninja instead of make)",
            ]
            self.error_logger.log_error("cmake", str(e), "Configuring the build system", suggestions)
            print_status(f"Configuration failed: {e}", "ERROR")
            print_status(
                f"Detailed error log saved to: {self.error_logger.get_log_file_path()}",
                "INFO",
            )
            sys.exit(1)

    def build(self):
        if self.__value["build"] != "build":
            return 0

        print_status("Building project...", "INFO")
        try:
            exit_code = build_helper.build_project(
                self.__value["builder"],
                self.__value["build_enum"],
                self.__value["system"],
                self.__shell_flag(),
            )
            if exit_code == 0:
                print_status("Build completed successfully", "SUCCESS")
            else:
                print_status("Build failed", "ERROR")
                sys.exit(1)
        except subprocess.CalledProcessError as e:
            suggestions = [
                "Check if all dependencies are installed",
                "Verify the build configuration is correct",
                "Try cleaning the build directory and reconfiguring",
                "Check for compiler errors in the output above",
            ]
            self.error_logger.log_error("build", str(e), "Building the project", suggestions)
            print_status(f"Build failed: {e}", "ERROR")
            print_status(
                f"Detailed error log saved to: {self.error_logger.get_log_file_path()}",
                "INFO",
            )
            sys.exit(1)

    def cppcheck(self, source_path, build_path):
        """Run cppcheck static analysis with user-friendly interface."""
        if self.__value["build"] != "build" or not self.__solvers_flags.is_cppcheck():
            return 0

        print_status("Starting static code analysis with cppcheck...", "INFO")

        # Check if cppcheck is installed
        try:
            version_result = subprocess.run(["cppcheck", "--version"], capture_output=True, check=True, text=True)
            print_status(f"Found cppcheck: {version_result.stdout.strip()}", "SUCCESS")
        except (subprocess.CalledProcessError, FileNotFoundError):
            suggestions = [
                "Ubuntu/Debian: sudo apt-get install cppcheck",
                "CentOS/RHEL/Fedora: sudo dnf install cppcheck",
                "macOS: brew install cppcheck",
                "Windows: choco install cppcheck or winget install cppcheck",
            ]
            self.error_logger.log_error(
                "cppcheck --version",
                "cppcheck command not found",
                "Checking for cppcheck installation",
                suggestions,
            )
            print_status("cppcheck not found. Please install cppcheck:", "ERROR")
            for suggestion in suggestions:
                print_status(f"  - {suggestion}", "INFO")
            return 1

        # Prepare output directory and file
        os.makedirs(build_path, exist_ok=True)
        output_file = os.path.join(build_path, "cppcheck_output.log")

        # Build cppcheck command with optimized settings
        cppcheck_cmd = self._build_cppcheck_command(source_path, output_file)

        try:
            original_dir = os.getcwd()
            os.chdir(source_path)

            print_status("Analyzing source code for potential issues...", "INFO")
            print_status("This may take a few minutes for large codebases", "INFO")

            result = subprocess.run(cppcheck_cmd, capture_output=True, text=True, check=False)

            os.chdir(original_dir)

            exit_code = self._process_cppcheck_results(result, output_file, source_path)
            self.summary_reporter.add_cppcheck_report(output_file, exit_code)

            return exit_code

        except Exception as e:
            error_msg = f"Unexpected error during cppcheck execution: {e}"
            self.error_logger.log_error(
                " ".join(cppcheck_cmd),
                str(e),
                "Running cppcheck static analysis",
                [
                    "Check if the source directory is accessible",
                    "Verify cppcheck installation",
                ],
            )
            print_status(error_msg, "ERROR")
            try:
                os.chdir(original_dir)
            except Exception:  # noqa: E722
                pass
            return 1

    def _build_cppcheck_command(self, source_path: str, output_file: str) -> list[str]:
        """Build the cppcheck command with appropriate settings."""
        return cppcheck_helper.build_cppcheck_command(source_path, output_file)

    def _process_cppcheck_results(self, result: subprocess.CompletedProcess, output_file: str, source_path: str) -> int:
        """Process cppcheck results and provide user-friendly feedback."""
        return cppcheck_helper.process_cppcheck_results(result, output_file)

    def test(self, source_path, build_path):
        if self.__value["test"] != "test":
            return 0

        if self.__solvers_flags.is_valgrind():
            exit_code = test_helper.run_valgrind_test(source_path, build_path, self.__shell_flag())
            self.summary_reporter.add_valgrind_report(build_path, exit_code)
            return exit_code

        return test_helper.run_ctest(
            self.__value["builder"],
            self.__value["build_enum"],
            self.__value["system"],
            self.__value["verbosity"],
            self.__shell_flag(),
            sanitizer_type=self.__solvers_flags.get_sanitizer_type(),
            source_path=source_path,
        )

    def coverage(self, source_path, build_path):
        """Run code coverage analysis.

        Args:
            source_path: Path to source directory (project root).
            build_path: Path to build directory.

        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        if self.__value["build"] != "build" or not self.__solvers_flags.is_coverage():
            return 0

        print_status("Starting code coverage collection and report generation...", "INFO")

        try:
            from coverage_tool import get_coverage
        except ImportError:
            print_status(
                "coverage-tool is not installed. Install with: pip install coverage-tool",
                "ERROR",
            )
            return 1

        coverage_result = get_coverage(
            compiler="auto",
            build_folder=build_path,
            source_folder=source_path,
            output_folder=os.path.join(build_path, "coverage_report"),
            summary=True,
            project_root=source_path,
        )
        if coverage_result == 0:
            print_status("Coverage collection completed successfully", "SUCCESS")
            self.summary_reporter.add_coverage_report(build_path, 0)
            return 0
        else:
            print_status("Coverage collection failed", "ERROR")
            return 1

    def __shell_flag(self):
        return self.__value["system"] == "Windows"

    def move_to_build_folder(self):
        os.chdir("..")
        build_folder = self.__value["build_folder"]

        if os.path.isdir(build_folder) and self.__value.get("config") == "config":
            shutil.rmtree(build_folder, ignore_errors=True)

        if not os.path.isdir(build_folder):
            os.mkdir(build_folder)

        os.chdir(build_folder)
        return os.getcwd()

    def find_build_directory_for_analysis(self, source_path: str) -> Optional[str]:
        """Find the most appropriate build directory for analysis tools."""
        current_build = self.__value.get("build_folder")
        if current_build and os.path.isdir(os.path.join(source_path, current_build)):
            return os.path.join(source_path, current_build)

        build_dir = BuildDirectoryDetector.find_best_build_directory(source_path, current_build)
        if build_dir:
            return str(build_dir)

        return None


def parse_args(args):
    """Parse command line arguments, handling special flags first."""
    processed_args = []

    for arg in args:
        # Handle sanitizer flags with dot notation (e.g., --sanitizer.undefined)
        if arg.startswith("--sanitizer."):
            sanitizer_type = arg.split(".", 1)[1].lower()
            valid_sanitizers = ["address", "undefined", "thread", "memory", "leak"]
            if sanitizer_type in valid_sanitizers:
                processed_args.extend(["sanitizer", sanitizer_type])
            else:
                print_status(
                    f"Invalid sanitizer type: {sanitizer_type}. Valid options: {', '.join(valid_sanitizers)}",
                    "ERROR",
                )
                sys.exit(1)
        elif re.search(r"[/\\]", arg) and re.search(r"[Cc]lang|[Gg][Cc][Cc]|[Gg]\+\+", arg):
            # Compiler path argument: contains a directory separator and a compiler
            # name. Pass through verbatim — do NOT split on '.'/'_' or lowercase,
            # as that would destroy paths like C:/msys64/mingw64/bin/clang.exe.
            processed_args.append(arg)
        else:
            # Split dotted/space-separated tokens, then split each on '_'.
            for part in re.split(r"\.|\ ", arg.lower()):
                processed_args.extend(re.split(r"_", part))

    return processed_args


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        print_status("Solvers Build Configuration Helper", "INFO")
        print("\n" + "=" * 80)
        print("DEFAULT CONFIGURATION:")
        print("  Build System: Ninja (fast, cross-platform)")
        print("  Compiler:     Clang (clang/clang++)")
        print("=" * 80)
        print("\nUsage examples:")
        print("  1. Default build (Ninja + Clang):")
        print("     setup.py config.build.test")
        print("  2. Release build with GCC:")
        print("     setup.py config.build.test.gcc.release")
        print("  3. macOS build with Xcode:")
        print("     setup.py config.build.test.xcode")
        print("  4. Build with coverage (analysis runs automatically):")
        print("     setup.py config.build.test.coverage")
        print("  5. Build the optional backends:")
        print("     setup.py config.build.test.nlopt.ceres")
        print("\nBuild system generators:")
        print("  ninja     - Ninja build system (DEFAULT, fast, cross-platform)")
        print("  xcode     - Xcode (macOS only, full IDE integration)")
        print("  vs17/19/22- Visual Studio (Windows only)")
        print("\nCompiler options:")
        print("  clang     - Clang compiler (DEFAULT)")
        print("  clang-XX  - Specific Clang version (e.g., clang-15)")
        print("  gcc       - GCC compiler (Unix/Linux)")
        print("  gcc-XX    - Specific GCC version (e.g., gcc-11)")
        print("  g++       - G++ compiler (Unix/Linux)")
        print("  g++-XX    - Specific G++ version (e.g., g++-11)")
        print("\nBuild commands:")
        print("  config    - Configure the build system")
        print("  build     - Build the project")
        print("  test      - Run tests")
        print("  coverage  - Enable coverage (automatically displays summary)")
        print("\nSanitizer flags:")
        print("  --sanitizer.address        Enable AddressSanitizer")
        print("  --sanitizer.undefined      Enable UndefinedBehaviorSanitizer")
        print("  --sanitizer.thread         Enable ThreadSanitizer")
        print("  --sanitizer.memory         Enable MemorySanitizer (Clang only)")
        print("  --sanitizer.leak           Enable LeakSanitizer")
        print("\nSanitizer examples:")
        print("  python setup.py config.build.test.ninja.clang --sanitizer.address")
        print("\nCoverage analysis examples:")
        print("  python setup.py config.build.test.ninja.clang.coverage")
        print("\nAvailable options:")
        SolversFlags([]).helper()
        return

    try:
        arg_list = parse_args(sys.argv[1:])
        if not arg_list:
            print_status(
                "No build configuration specified. Use --help for usage information.",
                "ERROR",
            )
            sys.exit(1)

        print_status(f"Starting build configuration for {platform.system()}", "INFO")
        compilation_calc = SolversConfiguration(arg_list)

        source_path = os.path.dirname(os.getcwd())
        build_path = compilation_calc.move_to_build_folder()

        print_status(f"Build directory: {build_path}", "INFO")

        # Execute build pipeline
        try:
            start = time.perf_counter()
            compilation_calc.config(source_path, build_path)
            config_end = time.perf_counter()

            build_start = time.perf_counter()
            compilation_calc.build()
            build_end = time.perf_counter()

            cppcheck_start = time.perf_counter()
            compilation_calc.cppcheck(source_path, build_path)
            cppcheck_end = time.perf_counter()

            test_start = time.perf_counter()
            compilation_calc.test(source_path, build_path)
            test_end = time.perf_counter()

            coverage_start = time.perf_counter()
            compilation_calc.coverage(source_path, build_path)
            end = time.perf_counter()

            print_status(f"Config time: {config_end - start:.4f} seconds", "INFO")
            print_status(f"Build time: {build_end - build_start:.4f} seconds", "INFO")
            print_status(f"Cppcheck time: {cppcheck_end - cppcheck_start:.4f} seconds", "INFO")
            print_status(f"Test time: {test_end - test_start:.4f} seconds", "INFO")
            print_status(f"Coverage time: {end - coverage_start:.4f} seconds", "INFO")

            print_status(f"Total time: {end - start:.4f} seconds", "INFO")

            print_status("Build process completed successfully!", "SUCCESS")

            # Display comprehensive summary report
            compilation_calc.summary_reporter.display_summary()

            if compilation_calc.error_logger.has_errors():
                print_status(
                    f"Error log available at: {compilation_calc.error_logger.get_log_file_path()}",
                    "INFO",
                )
                print_status(
                    "Review the error log for detailed troubleshooting information",
                    "INFO",
                )

        except SystemExit:
            # Re-raise SystemExit to preserve exit codes from subprocess failures
            compilation_calc.summary_reporter.display_summary()
            if compilation_calc.error_logger.has_errors():
                print_status(
                    f"Error log available at: {compilation_calc.error_logger.get_log_file_path()}",
                    "ERROR",
                )
            raise

    except KeyboardInterrupt:
        print_status("\nBuild process interrupted by user", "WARNING")
        try:
            if "compilation_calc" in locals():
                compilation_calc.summary_reporter.display_summary()
        except Exception:  # noqa: E722
            pass
        sys.exit(1)
    except Exception as e:
        print_status(f"An unexpected error occurred: {e}", "ERROR")
        try:
            if "compilation_calc" in locals():
                compilation_calc.summary_reporter.display_summary()
                if compilation_calc.error_logger.has_errors():
                    print_status(
                        f"Error log available at: {compilation_calc.error_logger.get_log_file_path()}",
                        "ERROR",
                    )
        except Exception:  # noqa: E722
            pass
        if DEBUG_FLAG:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
