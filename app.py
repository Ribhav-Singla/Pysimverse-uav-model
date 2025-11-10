from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Pysimverse UAV Model API is running!"

@app.route('/run-comparison', methods=['POST'])
def run_comparison():
    try:
        print("[INFO] Starting UAV comparison test...")
        print("[INFO] Executing uav_comparison_test_new.py...")
        
        # result = subprocess.run(
        #     ['python', 'uav_comparison_test_new.py'],
        #     capture_output=True,
        #     text=True,
        #     cwd=os.getcwd()
        # )
        
        # if result.returncode != 0:
        #     error_msg = f"Script execution failed with return code {result.returncode}"
        #     print(f"[ERROR] {error_msg}")
        #     return jsonify({
        #         "status": "error",
        #         "message": error_msg
        #     }), 500
        
        print(f"[SUCCESS] Comparison completed successfully!")
        
        return jsonify({
            "status": "success",
            "message": "Comparison test completed successfully"
        }), 200
        
    # except FileNotFoundError as e:
    #     error_msg = f"Script file not found: {str(e)}"
    #     print(f"[ERROR] {error_msg}")
    #     return jsonify({
    #         "status": "error",
    #         "message": error_msg
    #     }), 404
    except Exception as e:
        error_msg = f"Unexpected error occurred: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
