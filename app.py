from flask import Flask, request, jsonify
import subprocess
import os
import csv
import sys

# Force logs to flush immediately
sys.stdout.reconfigure(line_buffering=True)

app = Flask(__name__)

@app.route('/')
def home():
    return "Pysimverse UAV Model API is running!"

def collect_files_with_metadata(path):
    """Recursively collect all files with metadata"""
    files_list = []
    try:
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, path)
                try:
                    file_size = os.path.getsize(file_path)
                    file_ext = os.path.splitext(file)[1]
                    files_list.append({
                        "filename": file,
                        "path": rel_path,
                        "size_bytes": file_size,
                        "type": file_ext if file_ext else "no_extension"
                    })
                except OSError as e:
                    print(f"[WARNING] Could not access file {file_path}: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Error collecting files: {str(e)}")
    
    return files_list

def read_csv_file(csv_path):
    """Read CSV file and return as list of dictionaries"""
    data = []
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        print(f"[INFO] Successfully read CSV file: {csv_path}")
    except FileNotFoundError:
        print(f"[WARNING] CSV file not found: {csv_path}")
    except Exception as e:
        print(f"[ERROR] Error reading CSV file: {str(e)}")
    
    return data

@app.route('/run-comparison', methods=['POST'])
def run_comparison():
    try:
        print("[INFO] Starting UAV comparison test...")
        print("[INFO] Executing uav_comparison_test_new.py...")
        
        # Use Popen to stream logs in real-time
        process = subprocess.Popen(
            ['python', 'uav_comparison_test_new.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=os.getcwd()
        )
        
        # Read and print stdout in real-time
        print("[SCRIPT OUTPUT START]")
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[SCRIPT] {output.rstrip()}")
        
        # Get any remaining stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            print("[SCRIPT ERRORS]")
            print(stderr_output)
        
        print("[SCRIPT OUTPUT END]")
        
        return_code = process.returncode
        
        if return_code != 0:
            error_msg = f"Script execution failed with return code {return_code}"
            print(f"[ERROR] {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg,
                "stderr": stderr_output
            }), 500
        
        print(f"[SUCCESS] Comparison completed successfully!")
        
        # Find the Agents folder
        base_dir = os.getcwd()
        results_folder = "Agents"
        results_path = os.path.join(base_dir, results_folder)
        
        if not os.path.exists(results_path):
            error_msg = "No Agents folder found"
            print(f"[ERROR] {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 404
        
        print(f"[INFO] Results folder: {results_folder}")
        
        # Collect all files with metadata
        files_list = collect_files_with_metadata(results_path)
        print(f"[INFO] Found {len(files_list)} files in results folder")
        
        # Read summary CSV file
        summary_csv_path = os.path.join(results_path, "results_summary.csv")
        summary_data = read_csv_file(summary_csv_path)
        
        # Count files by type
        file_types_count = {}
        for file_info in files_list:
            file_type = file_info['type']
            file_types_count[file_type] = file_types_count.get(file_type, 0) + 1
        
        return jsonify({
            "status": "success",
            "message": "Comparison test completed successfully",
            "results_folder": results_folder,
            "summary": {
                "total_files": len(files_list),
                "file_types": file_types_count,
                "csv_records": len(summary_data)
            },
            "files": files_list,
            "csv_data": summary_data
        }), 200
        
    except FileNotFoundError as e:
        error_msg = f"Script file not found: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 404
    except Exception as e:
        error_msg = f"Unexpected error occurred: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
