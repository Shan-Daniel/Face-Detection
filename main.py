import subprocess
import sys
import os

def run_python(script):
    if not os.path.exists(script):
        print(f"Script not found: {script}")
        return
    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")

def menu():
    while True:
        print("\n==== Face Mask Detection System ====")
        print("1. Train Model (train_model.py)")
        print("2. Run Detection GUI (mask_detection_gui.py)")
        print("3. View Analytics Dashboard (analytics_dashboard.py)")
        print("4. Run YOLO Detection (yolo_mask_detection.py)")
        print("5. Run Web App (app.py)")
        print("6. Exit")
        choice = input("Select an option [1-6]: ").strip()
        if choice == '1':
            run_python('train_model.py')
        elif choice == '2':
            run_python('mask_detection_gui.py')
        elif choice == '3':
            run_python('analytics_dashboard.py')
        elif choice == '4':
            run_python('yolo_mask_detection.py')
        elif choice == '5':
            run_python('app.py')
        elif choice == '6':
            print("Exiting.")
            break
        else:
            print("Invalid option. Please select 1-6.")

if __name__ == '__main__':
    menu()
