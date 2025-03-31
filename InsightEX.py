import os
import sys
import subprocess

def is_inside_base_env():
    """Check if running inside BaseEnv."""
    return os.path.basename(sys.prefix) == "BaseEnv"

def run_setup_script():
    """Run setup only if BaseEnv is missing."""
    if is_inside_base_env():
        return
    base_env_path = os.path.join(os.getcwd(), "BaseEnv")
    if not os.path.exists(base_env_path):
        print("BaseEnv not found. Running setup script...")
        script = "setup.bat" if os.name == "nt" else "setup.sh"
        subprocess.run(script, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("BaseEnv already exists.")

def activate_base_env():
    """Activate BaseEnv and restart script inside it."""
    if is_inside_base_env():
        return
    base_env_path = os.path.join(os.getcwd(), "BaseEnv")
    print(f"Activating BaseEnv from: {base_env_path}")
    if os.name == "nt":
        activate_script = os.path.join(base_env_path, "Scripts", "activate.bat")
        activate_cmd = f'cmd.exe /c "{activate_script} && python {" ".join(sys.argv)}"'
    else:
        activate_script = os.path.join(base_env_path, "bin", "activate")
        activate_cmd = f'bash -c "source {activate_script} && python {" ".join(sys.argv)}"'
    subprocess.run(activate_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sys.exit(0)

def run_insightex():
    """Run Main.py inside BaseEnv."""
    command = [sys.executable, "Main.py"] + sys.argv[1:]
    print("Running InsightEX.py...")
    subprocess.run(command, check=True)

def main():
    if not is_inside_base_env():
        run_setup_script()
        activate_base_env()
    run_insightex()

if __name__ == "__main__":
    main()
