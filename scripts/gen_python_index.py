import ast
import os
from typing import List, Tuple


def parse_file(path: str) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]], List[Tuple[str, int, int, List[Tuple[str, int, int]]]]]:
    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except Exception:
        return [], [], []
    variables = []
    functions = []
    classes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables.append((target.id, node.lineno, getattr(node, 'end_lineno', node.lineno)))
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name):
                variables.append((target.id, node.lineno, getattr(node, 'end_lineno', node.lineno)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append((node.name, node.lineno, node.end_lineno))
        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append((item.name, item.lineno, item.end_lineno))
            classes.append((node.name, node.lineno, node.end_lineno, methods))
    return variables, functions, classes


def gather_python_files(base_dir: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(base_dir):
        for fname in filenames:
            if fname.endswith('.py'):
                files.append(os.path.join(root, fname))
    return sorted(files)


def generate_index(base_dir: str) -> str:
    lines = []
    lines.append('## Python Code Index')
    for path in gather_python_files(base_dir):
        rel_path = os.path.relpath(path, base_dir)
        variables, functions, classes = parse_file(path)
        lines.append(f'\n### {rel_path}')
        for name, start, end in variables:
            lines.append(f'- Variable `{name}`: lines {start}-{end}')
        for name, start, end, methods in classes:
            lines.append(f'- Class `{name}`: lines {start}-{end}')
            for m_name, m_start, m_end in methods:
                lines.append(f'  - Method `{m_name}`: lines {m_start}-{m_end}')
        for name, start, end in functions:
            lines.append(f'- Function `{name}`: lines {start}-{end}')
    lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(generate_index(base_dir))
