from __future__ import annotations
import os
from pathlib import Path

# Configuration: Files to include
INCLUDE_EXTS = {".py", ".md", ".txt", ".yml", ".json"}
EXCLUDE_DIRS = {
    ".git", "__pycache__", "outputs", "data", "models", 
    ".idea", ".vscode", ".venv", "envs", "features", "wandb",
    "evaluations" # Skip raw json evals, keep summary? Maybe skip large jsons.
}
EXCLUDE_FILES = {
    "package-lock.json", "yarn.lock", "Poetry.lock", 
    "wisenet_split.json", "Project_Code_Complete.md"
}

def is_text_file(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.read(1024)
        return True
    except UnicodeDecodeError:
        return False

def collect_files(root: Path) -> list[Path]:
    files = []
    for root_dir, dirs, filenames in os.walk(root):
        # Filter directories inplace
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for name in filenames:
            path = Path(root_dir) / name
            if path.suffix.lower() not in INCLUDE_EXTS:
                continue
            if name in EXCLUDE_FILES:
                continue
            # Skip large files (e.g. > 1MB)
            if path.stat().st_size > 1_000_000:
                print(f"Skipping large file: {path}")
                continue
            
            files.append(path)
    return sorted(files)

def main():
    root = Path(".")
    output_file = root / "Project_Code_Complete.md"
    
    files = collect_files(root)
    
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# CSY3058 Smart Security Camera - Complete Project Code\n\n")
        out.write(f"Generated at: {os.path.basename(os.getcwd())}\n")
        out.write("This document contains all source code and configuration files for the project.\n\n")
        
        out.write("## Table of Contents\n")
        for f in files:
            anchor = str(f).replace('\\', '-').replace('/', '-').replace('.', '-')
            out.write(f"- [{f}](#{anchor})\n")
        out.write("\n---\n\n")
        
        for f in files:
            try:
                content = f.read_text(encoding="utf-8")
                # Determine language for markdown syntax
                ext = f.suffix.lower()
                lang = "text"
                if ext == ".py": lang = "python"
                if ext == ".md": lang = "markdown"
                if ext == ".json": lang = "json"
                if ext == ".yml": lang = "yaml"
                # Safe anchor generation
                anchor = str(f).replace('\\', '-').replace('/', '-').replace('.', '-')
                
                out.write(f"## {f.name} <a id='{anchor}'></a>\n\n")
                out.write(f"```{lang}\n")
                out.write(content)
                out.write("\n```\n\n")
                out.write("---\n\n")
            except Exception as e:
                print(f"Error reading {f}: {e}")

    print(f"Successfully aggregated {len(files)} files into {output_file}")

if __name__ == "__main__":
    main()
