#!/usr/bin/env python3
"""
RVC Rust Application Launcher
=============================

This script launches the RVC Rust application with comprehensive logging
and error monitoring. It provides real-time feedback about the application
status and helps debug any issues that may occur.

Features:
- Detailed startup logging
- Real-time process monitoring
- Error detection and reporting
- Graceful shutdown handling
- Configuration validation
"""

import os
import sys
import time
import signal
import subprocess
import threading
import json
from pathlib import Path
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'


class RVCLauncher:
    """Main launcher class for RVC Rust application"""

    def __init__(self):
        self.process = None
        self.running = False
        self.start_time = None
        self.log_file = None

    def log(self, message, level="INFO", color=Colors.WHITE):
        """Log message with timestamp and color"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_message = f"{color}[{timestamp}] [{level}] {message}{Colors.RESET}"
        print(formatted_message)

        # Also write to log file if available
        if self.log_file:
            self.log_file.write(f"[{timestamp}] [{level}] {message}\n")
            self.log_file.flush()

    def setup_logging(self):
        """Setup log file for debugging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_filename = f"rvc_rust_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = log_dir / log_filename

        try:
            self.log_file = open(log_path, 'w', encoding='utf-8')
            self.log(f"Log file created: {log_path}", "INFO", Colors.CYAN)
        except Exception as e:
            self.log(f"Failed to create log file: {e}", "ERROR", Colors.RED)

    def check_prerequisites(self):
        """Check if all required files and tools are available"""
        self.log("🔍 Checking prerequisites...", "INFO", Colors.CYAN)

        checks = [
            ("Model file", "assets/weights/kikiV1.pth"),
            ("Index file", "logs/kikiV1.index"),
            ("Rust project", "rvc-rs/Cargo.toml"),
            ("UI project", "rvc-rs/ui/package.json"),
        ]

        all_good = True
        for name, path in checks:
            if os.path.exists(path):
                if name.endswith("file"):
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    self.log(f"✅ {name}: {path} ({size_mb:.1f} MB)", "INFO", Colors.GREEN)
                else:
                    self.log(f"✅ {name}: {path}", "INFO", Colors.GREEN)
            else:
                self.log(f"❌ {name}: {path} (missing)", "ERROR", Colors.RED)
                all_good = False

        # Check tools
        tools = ["cargo", "npm", "node"]
        for tool in tools:
            try:
                result = subprocess.run([tool, "--version"],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    self.log(f"✅ {tool}: {version}", "INFO", Colors.GREEN)
                else:
                    self.log(f"❌ {tool}: Not working properly", "ERROR", Colors.RED)
                    all_good = False
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self.log(f"❌ {tool}: Not found", "ERROR", Colors.RED)
                all_good = False

        return all_good

    def build_frontend(self):
        """Build the frontend if needed"""
        self.log("🎨 Building frontend...", "INFO", Colors.CYAN)

        try:
            # Check if we need to install dependencies
            node_modules = Path("rvc-rs/ui/node_modules")
            if not node_modules.exists():
                self.log("📦 Installing npm dependencies...", "INFO", Colors.YELLOW)
                result = subprocess.run(
                    ["npm", "install"],
                    cwd="rvc-rs/ui",
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode != 0:
                    self.log(f"❌ npm install failed: {result.stderr}", "ERROR", Colors.RED)
                    return False
                self.log("✅ npm dependencies installed", "INFO", Colors.GREEN)

            # Build the frontend
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd="rvc-rs/ui",
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                self.log("✅ Frontend build successful", "INFO", Colors.GREEN)
                return True
            else:
                self.log(f"❌ Frontend build failed: {result.stderr}", "ERROR", Colors.RED)
                return False

        except subprocess.TimeoutExpired:
            self.log("❌ Frontend build timed out", "ERROR", Colors.RED)
            return False
        except Exception as e:
            self.log(f"❌ Frontend build error: {e}", "ERROR", Colors.RED)
            return False

    def compile_rust(self):
        """Compile the Rust backend"""
        self.log("🦀 Compiling Rust backend...", "INFO", Colors.CYAN)

        try:
            result = subprocess.run(
                ["cargo", "build", "--release", "--manifest-path", "rvc-rs/Cargo.toml"],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                self.log("✅ Rust compilation successful", "INFO", Colors.GREEN)
                return True
            else:
                self.log(f"❌ Rust compilation failed: {result.stderr}", "ERROR", Colors.RED)
                return False

        except subprocess.TimeoutExpired:
            self.log("❌ Rust compilation timed out", "ERROR", Colors.RED)
            return False
        except Exception as e:
            self.log(f"❌ Rust compilation error: {e}", "ERROR", Colors.RED)
            return False

    def start_application(self):
        """Start the RVC Rust application"""
        self.log("🚀 Starting RVC Rust application...", "INFO", Colors.CYAN)

        try:
            # Start the Tauri application
            self.process = subprocess.Popen(
                ["cargo", "tauri", "dev"],
                cwd="rvc-rs/ui",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            self.running = True
            self.start_time = time.time()

            self.log("✅ Application started successfully", "INFO", Colors.GREEN)
            self.log("🌐 Application should open in your default browser", "INFO", Colors.YELLOW)
            self.log("📝 Monitoring application logs...", "INFO", Colors.CYAN)

            # Start log monitoring thread
            log_thread = threading.Thread(target=self.monitor_logs, daemon=True)
            log_thread.start()

            return True

        except Exception as e:
            self.log(f"❌ Failed to start application: {e}", "ERROR", Colors.RED)
            return False

    def monitor_logs(self):
        """Monitor application logs in real-time"""
        while self.running and self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    # Categorize log messages
                    if "ERROR" in line or "❌" in line:
                        self.log(f"APP: {line}", "ERROR", Colors.RED)
                    elif "WARN" in line or "⚠️" in line:
                        self.log(f"APP: {line}", "WARN", Colors.YELLOW)
                    elif "✅" in line or "SUCCESS" in line:
                        self.log(f"APP: {line}", "INFO", Colors.GREEN)
                    elif "🚀" in line or "START" in line:
                        self.log(f"APP: {line}", "INFO", Colors.BLUE)
                    else:
                        self.log(f"APP: {line}", "DEBUG", Colors.WHITE)
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.log(f"Log monitoring error: {e}", "ERROR", Colors.RED)
                break

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.log(f"📡 Received signal {signum}, shutting down...", "INFO", Colors.YELLOW)
        self.shutdown()

    def shutdown(self):
        """Gracefully shutdown the application"""
        self.log("🛑 Shutting down application...", "INFO", Colors.YELLOW)

        self.running = False

        if self.process:
            try:
                # Try to terminate gracefully
                self.process.terminate()
                self.process.wait(timeout=10)
                self.log("✅ Application stopped gracefully", "INFO", Colors.GREEN)
            except subprocess.TimeoutExpired:
                # Force kill if necessary
                self.log("⚠️  Force killing application...", "WARN", Colors.YELLOW)
                self.process.kill()
                self.process.wait()
                self.log("✅ Application force stopped", "INFO", Colors.GREEN)
            except Exception as e:
                self.log(f"❌ Error stopping application: {e}", "ERROR", Colors.RED)

        if self.start_time:
            runtime = time.time() - self.start_time
            self.log(f"📊 Total runtime: {runtime:.1f} seconds", "INFO", Colors.CYAN)

        if self.log_file:
            self.log_file.close()

        self.log("👋 Goodbye!", "INFO", Colors.MAGENTA)

    def run(self):
        """Main run method"""
        print(f"{Colors.BOLD}{Colors.CYAN}")
        print("=" * 60)
        print("🎤 RVC Rust Application Launcher")
        print("=" * 60)
        print(f"{Colors.RESET}")

        # Setup logging
        self.setup_logging()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Run all checks and setup
            if not self.check_prerequisites():
                self.log("❌ Prerequisites check failed", "ERROR", Colors.RED)
                return 1

            if not self.build_frontend():
                self.log("❌ Frontend build failed", "ERROR", Colors.RED)
                return 1

            # Note: We skip Rust compilation for dev mode as Tauri handles it
            self.log("ℹ️  Skipping Rust compilation (dev mode)", "INFO", Colors.YELLOW)

            if not self.start_application():
                self.log("❌ Failed to start application", "ERROR", Colors.RED)
                return 1

            # Wait for the application to finish
            try:
                while self.running and self.process and self.process.poll() is None:
                    time.sleep(1)

                if self.process:
                    exit_code = self.process.returncode
                    if exit_code == 0:
                        self.log("✅ Application exited normally", "INFO", Colors.GREEN)
                    else:
                        self.log(f"⚠️  Application exited with code: {exit_code}", "WARN", Colors.YELLOW)

            except KeyboardInterrupt:
                self.log("🔄 Interrupted by user", "INFO", Colors.YELLOW)

            return 0

        except Exception as e:
            self.log(f"❌ Unexpected error: {e}", "ERROR", Colors.RED)
            return 1

        finally:
            self.shutdown()


def main():
    """Main entry point"""
    launcher = RVCLauncher()
    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
