import os
import sys
import subprocess

def is_inside_base_env():
    """Returns True if the current Python environment is BaseEnv."""
    return os.path.basename(sys.prefix) == "BaseEnv"

def run_setup_script():
    """If BaseEnv is missing, run the appropriate setup script to create it."""
    base_env_path = os.path.join(os.getcwd(), "BaseEnv")
    if not os.path.exists(base_env_path):
        print("BaseEnv not found.")
        if os.name == 'nt':  # Windows
            print("Running setup.bat to create BaseEnv...")
            subprocess.run("setup.bat", shell=True, check=True)
        else:  # Linux
            print("Running setup.sh to create BaseEnv...")
            subprocess.run("chmod +x setup.sh", shell=True, check=True)
            subprocess.run("./setup.sh", shell=True, check=True)
    else:
        print("BaseEnv already exists.")

def activate_base_env():
    """Activates BaseEnv and restarts the current script in that environment."""
    base_env_path = os.path.join(os.getcwd(), "BaseEnv")
    if is_inside_base_env():
        print("Already running inside BaseEnv.")
        return
    print(f"Activating BaseEnv from: {base_env_path}")
    if os.name == 'nt':  # Windows
        activate_script = os.path.join(base_env_path, "Scripts", "activate.bat")
        activate_cmd = f'cmd.exe /c "{activate_script} && python {" ".join(sys.argv)}"'
    else:  # Linux
        activate_script = os.path.join(base_env_path, "bin", "activate")
        activate_cmd = f'bash -c "source {activate_script} && python {" ".join(sys.argv)}"'

    subprocess.run(activate_cmd, shell=True, check=True)
    sys.exit(0)

def run_insightex():
    """Run the Main.py script with any passed command-line arguments."""
    command = [sys.executable, "Main.py"] + sys.argv[1:]
    print("Running InsightEX.py with command:", " ".join(command))
    subprocess.run(command, check=True)

def main():
    run_setup_script()
    if not is_inside_base_env():
        activate_base_env()
    run_insightex()

if __name__ == "__main__":
    main()
