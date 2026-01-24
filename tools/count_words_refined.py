import re
from pathlib import Path

def count_words_refined(file_path):
    text = Path(file_path).read_text(encoding='utf-8')
    
    text = re.sub(r'%.*', '', text)
    match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', text, re.DOTALL)
    body = match.group(1) if match else text

    # Exclude Code Blocks
    body = re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', ' ', body, flags=re.DOTALL)

    # Exclude Titles/TOC
    body = re.sub(r'\\maketitle', '', body)
    body = re.sub(r'\\tableofcontents', '', body)
    body = re.sub(r'\\listoffigures', '', body)
    body = re.sub(r'\\listoftables', '', body)
    body = re.sub(r'\\newpage', '', body)
    body = re.sub(r'\\clearpage', '', body)
    body = re.sub(r'\\begin\{abstract\}', '', body)
    body = re.sub(r'\\end\{abstract\}', '', body)

    # Exclude Captions/Labels/Images
    def remove_command_content(text, command_name):
        result = []
        idx = 0
        cmd_len = len(command_name)
        while idx < len(text):
            if text[idx:idx+cmd_len] == command_name:
                peek_idx = idx + cmd_len
                while peek_idx < len(text) and text[peek_idx].isspace(): peek_idx += 1
                if peek_idx < len(text) and text[peek_idx] == '{':
                    brace_count = 1
                    curr = peek_idx + 1
                    while curr < len(text) and brace_count > 0:
                        if text[curr] == '{': brace_count += 1
                        elif text[curr] == '}': brace_count -= 1
                        curr += 1
                    idx = curr
                    continue
            result.append(text[idx])
            idx += 1
        return "".join(result)

    body = remove_command_content(body, "\\caption")
    body = remove_command_content(body, "\\label")
    body = remove_command_content(body, "\\includegraphics")
    
    body = re.sub(r'\\bibliography\{.*?\}', '', body)
    body = re.sub(r'\\bibliographystyle\{.*?\}', '', body)
    
    body = re.sub(r'\$.*?\$', ' ', body)
    body = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', ' ', body, flags=re.DOTALL)
    
    body = re.sub(r'\\[a-zA-Z]+\*?', ' ', body)
    body = re.sub(r'[\{\}\[\]]', ' ', body)
    
    words = [w for w in body.split() if any(c.isalnum() for c in w)]
    return len(words)

if __name__ == "__main__":
    p = r"e:\22837352\Media-Technology-Module\csy3058_as2_smart_security_camera\docs\report\CSY3058_A2_Report.tex"
    c = count_words_refined(p)
    Path("word_count_result.txt").write_text(str(c), encoding="utf-8")
