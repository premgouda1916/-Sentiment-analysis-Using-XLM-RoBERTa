import os
import sys
from huggingface_hub import HfApi, login

def main():
    print("=" * 60)
    # 1. Ask for Hugging Face Write Token
    token = input("Enter your Hugging Face Write Token (https://huggingface.co/settings/tokens): ").strip()
    if not token:
        print("[ERROR] Token cannot be empty!")
        sys.exit(1)
        
    try:
        # Authenticate
        login(token)
        print("[OK] Successfully logged in to Hugging Face.")
    except Exception as e:
        print(f"[ERROR] Login failed: {e}")
        sys.exit(1)

    # 2. Get username or extract from token details
    api = HfApi()
    try:
        user_info = api.whoami(token=token)
        username = user_info["name"]
        print(f"[OK] Authenticated as user: {username}")
    except Exception as e:
        print(f"[ERROR] Failed to get user info: {e}")
        sys.exit(1)

    # 3. Repository details
    repo_id = f"{username}/kannada-emotion-classifier"
    print(f"\nCreating Hugging Face Space (Docker) at: {repo_id}")
    
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True
        )
        print("[OK] Space repository created/verified successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to create Space: {e}")
        sys.exit(1)
        
    # 4. Upload specific runtime files to the Space
    files_to_upload = {
        "app.py": "app.py",
        "Dockerfile": "Dockerfile",
        "requirements.txt": "requirements.txt",
        "templates/index.html": "templates/index.html"
    }
    
    print("\nUploading project files to the Space...")
    for local_path, repo_path in files_to_upload.items():
        if not os.path.exists(local_path):
            print(f"[ERROR] Required file '{local_path}' not found in workspace!")
            sys.exit(1)
            
        print(f"Uploading {local_path} -> {repo_path}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="space"
            )
            print(f"  [OK] Uploaded {local_path}")
        except Exception as e:
            print(f"  [ERROR] Failed to upload {local_path}: {e}")
            sys.exit(1)
            
    print("\n[SUCCESS] DEPLOYMENT STARTED!")
    print(f"You can view your Space build and application at:")
    print(f"URL: https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    main()
