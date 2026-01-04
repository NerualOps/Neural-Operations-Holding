#!/usr/bin/env python3
"""
Epsilon AI - Python Services Startup Script
© 2025 Neural Operation's & Holding's LLC. All rights reserved.

This script starts all Python services for Epsilon AI
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

class PythonServicesManager:
    def __init__(self):
        self.services = []
        self.running = False
        
    def start_service(self, script_name, port):
        """Start a Python service"""
        # Safety check: validate inputs
        if not script_name or not isinstance(script_name, str):
            print(f"Error: Invalid script_name: must be a non-empty string")
            return False
        if '..' in script_name or '/' in script_name or '\\' in script_name:
            print(f"Error: Invalid script_name: path traversal detected")
            return False
        if not isinstance(port, int) or port < 1 or port > 65535:
            print(f"Error: Invalid port: must be between 1 and 65535")
            return False
        
        # Ensure script_name is in the current directory (prevent path traversal)
        script_path = Path(__file__).parent / script_name
        if not script_path.exists():
            print(f"Error: Script not found: {script_name}")
            return False
        if not script_path.is_file():
            print(f"Error: Not a file: {script_name}")
            return False
        
        try:
            print(f"Starting {script_name} on port {port}...")
            
            # Determine Python command
            python_cmd = 'python' if os.name == 'nt' else 'python3'
            
            # Start the service with file logging (avoid PIPE blocking)
            logs_dir = Path('/tmp')
            stdout_path = logs_dir / f"{Path(script_name).stem}.out.log"
            stderr_path = logs_dir / f"{Path(script_name).stem}.err.log"
            stdout_f = open(stdout_path, 'a', buffering=1)
            stderr_f = open(stderr_path, 'a', buffering=1)
            process = subprocess.Popen(
                [python_cmd, str(script_path)],
                cwd=Path(__file__).parent,
                stdout=stdout_f,
                stderr=stderr_f,
                text=True
            )
            
            self.services.append({
                'name': script_name,
                'port': port,
                'process': process
            })
            
            print(f"{script_name} started with PID {process.pid}")
            print(f"   ↳ logs: {stdout_path} | {stderr_path}")
            return True
            
        except Exception as e:
            print(f"Error: Failed to start {script_name}: {e}")
            return False
    
    def start_all_services(self):
        """Start all Python services"""
        print("Starting Epsilon AI Python Services...")
        
        services_config = [
            ('nlp_processor.py', 8001),
            ('learning_analytics.py', 8002),
            ('content_generator.py', 8003),
            ('document_learning.py', 8004)
        ]
        
        success_count = 0
        for script, port in services_config:
            if self.start_service(script, port):
                success_count += 1
            time.sleep(2)  # Give each service time to start
        
        if success_count == len(services_config):
            print(f"All {success_count} Python services started successfully!")
            self.running = True
            return True
        else:
            print(f"Warning: Only {success_count}/{len(services_config)} services started")
            return False
    
    def stop_all_services(self):
        """Stop all Python services"""
        print("Stopping Python services...")
        
        for service in self.services:
            try:
                print(f"Stopping {service['name']}...")
                service['process'].terminate()
                service['process'].wait(timeout=5)
                print(f"{service['name']} stopped")
            except subprocess.TimeoutExpired:
                print(f"Warning: Force killing {service['name']}...")
                service['process'].kill()
            except Exception as e:
                print(f"Error: Error stopping {service['name']}: {e}")
        
        self.services.clear()
        self.running = False
        print("All Python services stopped")
    
    def monitor_services(self):
        """Monitor running services"""
        while self.running:
            for service in self.services:
                if service['process'].poll() is not None:
                    print(f"Warning: Service {service['name']} has stopped unexpectedly")
                    # Restart the service
                    self.restart_service(service)
            time.sleep(10)
    
    def restart_service(self, service):
        """Restart a specific service"""
        # Safety check: validate service
        if not service or not isinstance(service, dict):
            print(f"Error: Invalid service object")
            return
        if 'name' not in service or not isinstance(service['name'], str):
            print(f"Error: Invalid service name")
            return
        
        script_name = service['name']
        # Prevent path traversal
        if '..' in script_name or '/' in script_name or '\\' in script_name:
            print(f"Error: Invalid script_name: path traversal detected")
            return
        
        # Ensure script_name is in the current directory
        script_path = Path(__file__).parent / script_name
        if not script_path.exists() or not script_path.is_file():
            print(f"Error: Script not found: {script_name}")
            return
        
        try:
            print(f"Restarting {service['name']}...")
            if service.get('process'):
                service['process'].terminate()
            time.sleep(2)
            
            python_cmd = 'python' if os.name == 'nt' else 'python3'
            logs_dir = Path('/tmp')
            stdout_path = logs_dir / f"{Path(script_name).stem}.out.log"
            stderr_path = logs_dir / f"{Path(script_name).stem}.err.log"
            stdout_f = open(stdout_path, 'a', buffering=1)
            stderr_f = open(stderr_path, 'a', buffering=1)
            new_process = subprocess.Popen(
                [python_cmd, str(script_path)],
                cwd=Path(__file__).parent,
                stdout=stdout_f,
                stderr=stderr_f,
                text=True
            )
            
            service['process'] = new_process
            print(f"{service['name']} restarted with PID {new_process.pid}")
            
        except Exception as e:
            print(f"Error: Failed to restart {service['name']}: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nReceived shutdown signal...")
        self.stop_all_services()
        sys.exit(0)

def main():
    """Main function"""
    manager = PythonServicesManager()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)
    
    try:
        # Start all services
        if manager.start_all_services():
            print("\nEpsilon AI Python Services are running!")
            print("Press Ctrl+C to stop all services")
            
            # Start monitoring in a separate thread
            monitor_thread = threading.Thread(target=manager.monitor_services, daemon=True)
            monitor_thread.start()
            
            # Keep the main thread alive
            while manager.running:
                time.sleep(1)
        else:
            print("Error: Failed to start all services")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: Unexpected error: {e}")
    finally:
        manager.stop_all_services()

if __name__ == "__main__":
    main()
