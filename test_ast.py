import ast

file_path = 'pages/4_Forecast.py'
with open(file_path, 'r') as f:
    code = f.read()

try:
    tree = ast.parse(code, file_path, "exec")
    print("AST parse successful!")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
    print(f"Line: {e.lineno}")
    print(f"Offset: {e.offset}")
    print(f"Text: {e.text}")
except Exception as e:
    print(f"Error: {e}")
