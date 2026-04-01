import os
import re

def run_validation():
    print("🔍 Running Pre-Submission Validation...\n")
    passed = True

    # 1. Check for Required Files
    required_files = ["Dockerfile", "inference.py", "openenv.yaml"]
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found {file}")
        else:
            print(f"❌ Missing {file} in the root directory!")
            passed = False

    # 2. Check inference.py for required Environment Variables
    if os.path.exists("inference.py"):
        with open("inference.py", "r", encoding="utf-8") as f:
            content = f.read()
            
            required_vars = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
            for var in required_vars:
                # Check if the exact string is in the code
                if var in content:
                    print(f"✅ inference.py uses required variable: {var}")
                else:
                    print(f"❌ inference.py is MISSING required variable: {var}")
                    passed = False
            
            # 3. Check for Hardcoded OpenAI Keys (Instant Disqualification)
            # Looks for standard OpenAI key formats to ensure they aren't hardcoded
            if re.search(r"sk-[a-zA-Z0-9]{20,}", content):
                print("❌ WARNING: Hardcoded API key detected in inference.py! Remove immediately.")
                passed = False
            else:
                print("✅ No hardcoded API keys detected.")

    print("\n--------------------------------------------------")
    if passed:
        print("🎉 VALIDATION PASSED! Your repo is ready for submission.")
    else:
        print("🛑 VALIDATION FAILED! Fix the errors above before submitting.")
        
if __name__ == "__main__":
    run_validation()