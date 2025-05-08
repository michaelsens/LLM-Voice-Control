import json
from pathlib import Path

INPUT_PATH = 'expanded.jsonl'
OUTPUT_PATH = 'methods.txt'

def extract_unique_methods(input_path: str, output_path: str):
    methods = set()
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            items = entry if isinstance(entry, list) else [entry]
            for obj in items:
                if isinstance(obj, dict):
                    rpc = obj.get('rpc')
                    if isinstance(rpc, dict):
                        method = rpc.get('method')
                        if method:
                            methods.add(method)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out:
        for method in sorted(methods):
            out.write(method + '\n')

if __name__ == '__main__':
    extract_unique_methods(INPUT_PATH, OUTPUT_PATH)
    count = len(open(OUTPUT_PATH, encoding='utf-8').read().splitlines())
    print(f"Wrote {OUTPUT_PATH} with {count} unique methods.")
