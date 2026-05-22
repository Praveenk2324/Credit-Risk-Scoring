import sys
import subprocess

def run_command(command):
    print(f"\nRunning: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Pipeline failed at: {command}")
        print("Stopping deployment to prevent a broken model from reaching production.")
        sys.exit(1)

def main():
    # 1. Check which batch we are processing
    batch_name = sys.argv[1] if len(sys.argv) > 1 else "01_base_data.csv"
    
    print("==================================================")
    print(f"  INITIATING CONTINUOUS TRAINING PIPELINE")
    print(f"  Target Dataset: {batch_name}")
    print("==================================================")

    run_command(f"python src/ingest.py {batch_name}")

    run_command("dvc repro")

    run_command("python src/register.py")

    run_command("python src/export.py")

    print("\n==================================================")
    print(" LOCAL PIPELINE COMPLETE & PASSED!")
    print(" To trigger the Cloud CI/CD Deployment, run:")
    print(" git add .")
    print(" git commit -m \"Auto-retrained model on new data\"")
    print(" git push origin main")
    print("==================================================")

if __name__ == "__main__":
    main()