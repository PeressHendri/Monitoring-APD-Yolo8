#!/usr/bin/env python3
"""
Run All APD Monitoring Systems
- Desktop monitoring
- Web monitoring
- Choose which one to run
"""

import subprocess
import sys
import os
import time

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🛡️  APD MONITORING SYSTEM - ALL SYSTEMS")
    print("=" * 60)
    print("Choose which monitoring system to run:")
    print()

def show_menu():
    """Show menu options"""
    print("1. 🖥️  Simple Monitor (Desktop)")
    print("   - Tampilan sederhana")
    print("   - Fullscreen otomatis")
    print("   - Alarm untuk setiap pelanggaran")
    print()
    
    print("2. 📺  CCTV Monitor (Desktop)")
    print("   - Interface CCTV profesional")
    print("   - Auto-scaling sesuai kamera")
    print("   - Tracking per orang")
    print()
    
    print("3. 🌐  Web Monitor (Browser)")
    print("   - Real-time video stream")
    print("   - Live statistics")
    print("   - Alarm notifications")
    print("   - Remote monitoring")
    print()
    
    print("4. 🔧  Install Requirements")
    print("   - Install semua dependencies")
    print()
    
    print("5. ❌  Exit")
    print()

def run_simple_monitor():
    """Run simple monitor"""
    print("🚀 Starting Simple Monitor...")
    print("Press 'q' to quit")
    print()
    try:
        subprocess.run([sys.executable, "simple_monitor.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️ Simple Monitor stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Simple Monitor: {e}")

def run_cctv_monitor():
    """Run CCTV monitor"""
    print("🚀 Starting CCTV Monitor...")
    print("Press 'q' to quit")
    print()
    try:
        subprocess.run([sys.executable, "cctv_monitor.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️ CCTV Monitor stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running CCTV Monitor: {e}")

def run_web_monitor():
    """Run web monitor"""
    print("🚀 Starting Web Monitor...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print()
    try:
        subprocess.run([sys.executable, "web_monitor.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️ Web Monitor stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Web Monitor: {e}")

def install_requirements():
    """Install requirements"""
    print("🔧 Installing requirements...")
    try:
        subprocess.run([sys.executable, "install_requirements.py"], check=True)
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")

def main():
    """Main function"""
    while True:
        print_banner()
        show_menu()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            print()
            
            if choice == '1':
                run_simple_monitor()
            elif choice == '2':
                run_cctv_monitor()
            elif choice == '3':
                run_web_monitor()
            elif choice == '4':
                install_requirements()
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                time.sleep(1)
            
            print("\n" + "=" * 60)
            input("Press Enter to continue...")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()
