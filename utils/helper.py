import json
from pathlib import Path

def load_and_display_settings():
    """Load and display all generation settings from config files"""
    settings_dir = Path("configs/settings")
    
    settings_files = {
        "Domains": "domains.jsonl",
        "Difficulties": "difficulties.jsonl",
        "Tones": "tones.jsonl",
        "Languages": "languages.jsonl",
        "Question Lengths": "length_categories.jsonl",
        "Topics": "topics.jsonl"
    }
    
    print("=" * 70)
    print(" SYNTHETIC QUESTION GENERATION SETTINGS")
    print("=" * 70)
    
    for category, filename in settings_files.items():
        filepath = settings_dir / filename
        if filepath.exists():
            print(f"\n📋 {category}:")
            print("-" * 70)
            
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        config = json.loads(line)
                        option = config.get('option', 'N/A')
                        weight = config.get('weight', 1.0)
                        
                        # Create a simple visual bar for weight
                        bar_length = int(weight * 20)
                        bar = "█" * bar_length
                        
                        print(f"  • {option:<30} Weight: {weight:.2f}  {bar}")
        else:
            print(f"\n❌ {category}: File not found")
    
    print("\n" + "=" * 70)
    print("💡 Higher weights = more likely to be selected during generation")
    print("=" * 70)

load_and_display_settings()