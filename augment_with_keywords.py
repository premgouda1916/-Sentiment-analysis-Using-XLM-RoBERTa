import os
import random

# Define keywords for each emotion
KEYWORDS = {
    "joy": ["ಸಂತೋಷ", "ನಗು", "ಹರ್ಷ", "ಖುಷಿ", "ಆನಂದ", "ಉಲ್ಲಾಸ", "ಸಂಭ್ರಮ", "ಇಷ್ಟ", "ಪ್ರೀತಿ", "ಅದ್ಭುತ", "ಚೆನ್ನಾಗಿದೆ", "ಹೆಮ್ಮೆ"],
    "anger": ["ಕೋಪ", "ಸಿಟ್ಟು", "ಆಕ್ರೋಶ", "ಅಸಮಾಧಾನ", "ರೋಷ", "ದ್ವೇಷ", "ಕೆಟ್ಟ"],
    "sadness": ["ದುಃಖ", "ಬೇಸರ", "ಅಳು", "ನೋವು", "ಖಿನ್ನತೆ", "ವ್ಯಥೆ", "ಸಂಕಟ", "ಹಸಿವು", "ದಣಿವು"],
    "fear": ["ಭಯ", "ಆತಂಕ", "ಹೆದರಿಕೆ", "ನಡುಕ", "ಗಾಬರಿ"],
    "neutral": ["ಸಾಮಾನ್ಯ", "ಪರವಾಗಿಲ್ಲ", "ಸರಿ", "ಹೌದು", "ಇಲ್ಲ"]
}

# Templates for generating short sentences
TEMPLATES = [
    "{keyword}",
    "ನನಗೆ {keyword} ಆಗುತ್ತಿದೆ",  # I am feeling {keyword}
    "{keyword} ಇದೆ",            # There is {keyword}
    "ತುಂಬಾ {keyword}",          # Too much {keyword}
    "{keyword} ಅನಿಸುತ್ತಿದೆ",      # Feeling {keyword}
]

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "joy": "joy_sentences_2000_kannada.txt",
    "anger": "anger_sentences_2000_kannada.txt",
    "sadness": "sad_sentences_2000_kannada.txt",
    "fear": "fear_sentences_2000_kannada.txt",
    "neutral": "neutral_sentences_2000_kannada.txt"
}

def augment_data():
    print("Starting data augmentation...")
    
    for emotion, filename in FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File {filename} not found. Skipping.")
            continue
            
        print(f"Augmenting {emotion} data in {filename}...")
        
        new_sentences = []
        keywords = KEYWORDS.get(emotion, [])
        
        # Generate sentences
        for keyword in keywords:
            for template in TEMPLATES:
                sentence = template.format(keyword=keyword)
                
                # Default copies
                copies = 20
                
                # Boost specific problematic keywords
                if keyword in ["ಹಸಿವು", "ಇಷ್ಟ", "ಕೆಟ್ಟ", "ಹೆಮ್ಮೆ"]:
                    copies = 100  # Boost these significantly
                
                for _ in range(copies):
                    new_sentences.append(sentence)
        
        # Append to file
        with open(filepath, "a", encoding="utf-8") as f:
            for sentence in new_sentences:
                f.write(sentence + "\n")
                
        print(f"Added {len(new_sentences)} sentences to {filename}.")

    print("Data augmentation complete.")

if __name__ == "__main__":
    augment_data()
