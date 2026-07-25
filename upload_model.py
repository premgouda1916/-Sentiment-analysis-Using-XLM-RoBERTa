import os
import sys
from huggingface_hub import HfApi, login

def main():
    print("=" * 60)
    print("HUGGING FACE MODEL UPLOADER")
    print("=" * 60)
    
    # 1. Ask for user token
    token = input("Enter your Hugging Face Write Token (get one from https://huggingface.co/settings/tokens): ").strip()
    if not token:
        print("❌ Token cannot be empty!")
        sys.exit(1)
        
    try:
        # Authenticate
        login(token)
        print("✅ Successfully logged in to Hugging Face.")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        sys.exit(1)

    # Repository details
    repo_id = "premgouda1916/kannada-sentiment-classifier-Xlm_RoBERTa"
    model_folder = "./kannada_emotion_model_xlm"
    
    if not os.path.exists(model_folder):
        print(f"❌ Model directory '{model_folder}' not found in the current folder!")
        sys.exit(1)
        
    print(f"\nCreating repository (if it doesn't exist): {repo_id}")
    api = HfApi()
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True
        )
        print("✅ Repository is ready.")
    except Exception as e:
        print(f"❌ Failed to create/verify repository: {e}")
        sys.exit(1)
        
    print(f"\nUploading contents of '{model_folder}' to Hugging Face...")
    print("This might take a few minutes as the model weights are around 1.1 GB. Please wait...")
    
    try:
        api.upload_folder(
            folder_path=model_folder,
            repo_id=repo_id,
            repo_type="model"
        )
        print("\n🎉 SUCCESS! Your model has been uploaded to Hugging Face.")
        print(f"URL: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
