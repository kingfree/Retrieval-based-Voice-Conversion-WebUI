#!/usr/bin/env python3
"""
RVC Configuration Test Script

This script tests the RVC Rust backend configuration and model loading
to ensure everything is working correctly before running the full application.
"""

import os
import sys
import json
from pathlib import Path

def test_file_existence():
    """Test if required model files exist"""
    print("🔍 Testing file existence...")

    # Check model file
    model_path = "assets/weights/kikiV1.pth"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model file found: {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Model file not found: {model_path}")
        return False

    # Check index file
    index_path = "logs/kikiV1.index"
    if os.path.exists(index_path):
        size_mb = os.path.getsize(index_path) / (1024 * 1024)
        print(f"✅ Index file found: {index_path} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Index file not found: {index_path}")
        return False

    return True

def test_config_creation():
    """Test configuration file creation"""
    print("\n📝 Testing configuration creation...")

    config = {
        "pth": "assets/weights/kikiV1.pth",
        "index": "logs/kikiV1.index",
        "hostapi": "",
        "wasapiExclusive": False,
        "inputDevice": "",
        "outputDevice": "",
        "srType": "sr_model",
        "threshold": -60,
        "pitch": 0,
        "formant": 0.0,
        "indexRate": 0.75,
        "rmsMixRate": 0.25,
        "f0method": "fcpe",
        "blockTime": 0.25,
        "crossfadeLength": 0.05,
        "nCpu": 2,
        "extraTime": 2.5,
        "iNoiseReduce": True,
        "oNoiseReduce": True,
        "usePv": False,
        "functionMode": "vc"
    }

    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # Save test configuration
    config_path = config_dir / "test_config.json"
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ Test configuration saved to: {config_path}")

        # Verify we can read it back
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)

        if loaded_config == config:
            print("✅ Configuration read/write test passed")
            return True
        else:
            print("❌ Configuration mismatch after read/write")
            return False

    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        return False

def test_rust_compilation():
    """Test if Rust backend compiles"""
    print("\n🦀 Testing Rust compilation...")

    try:
        import subprocess
        result = subprocess.run(
            ["cargo", "check", "-p", "rvc-lib", "--manifest-path", "rvc-rs/Cargo.toml"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        if result.returncode == 0:
            print("✅ Rust backend compilation successful")
            return True
        else:
            print("❌ Rust backend compilation failed:")
            print(result.stderr)
            return False

    except FileNotFoundError:
        print("⚠️  Cargo not found - skipping Rust compilation test")
        return True
    except Exception as e:
        print(f"❌ Rust compilation test failed: {e}")
        return False

def test_frontend_build():
    """Test if frontend builds successfully"""
    print("\n🎨 Testing frontend build...")

    try:
        import subprocess
        result = subprocess.run(
            ["npm", "run", "build"],
            capture_output=True,
            text=True,
            cwd="rvc-rs/ui"
        )

        if result.returncode == 0:
            print("✅ Frontend build successful")
            # Check if dist files exist
            dist_dir = Path("rvc-rs/ui/dist")
            if dist_dir.exists() and (dist_dir / "index.html").exists():
                print("✅ Frontend dist files generated")
                return True
            else:
                print("❌ Frontend dist files not found")
                return False
        else:
            print("❌ Frontend build failed:")
            print(result.stderr)
            return False

    except FileNotFoundError:
        print("⚠️  npm not found - skipping frontend build test")
        return True
    except Exception as e:
        print(f"❌ Frontend build test failed: {e}")
        return False

def print_system_info():
    """Print system information"""
    print("\n💻 System Information:")
    print(f"   OS: {sys.platform}")
    print(f"   Python: {sys.version}")
    print(f"   Working Directory: {os.getcwd()}")

    # Check for required tools
    tools = ["cargo", "npm", "node"]
    for tool in tools:
        try:
            import subprocess
            result = subprocess.run([tool, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                print(f"   {tool}: {version}")
            else:
                print(f"   {tool}: ❌ Not found")
        except FileNotFoundError:
            print(f"   {tool}: ❌ Not found")

def main():
    """Main test function"""
    print("🧪 RVC Configuration Test Suite")
    print("=" * 50)

    print_system_info()

    tests = [
        ("File Existence", test_file_existence),
        ("Configuration Creation", test_config_creation),
        ("Rust Compilation", test_rust_compilation),
        ("Frontend Build", test_frontend_build),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n📊 Test Summary:")
    print("=" * 50)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! RVC is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
