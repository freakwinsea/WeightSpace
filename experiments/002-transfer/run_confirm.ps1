# Held-out confirmation of C: base vs recipe transfer, 128 MATH-500 problems @16k (pre-registered in protocol.md).
Set-Location D:\WeightSpace\experiments\001-claude-editor
$env:PYTHONIOENCODING = 'utf-8'; $env:WS_HERE = 'D:\WeightSpace\experiments\002-transfer'; $env:WS_MODEL = 'Qwen/Qwen3.5-0.8B'
$py = 'D:\WeightSpace\.venv\Scripts\python.exe'; $o = 'D:\WeightSpace\experiments\002-transfer'
& $py heldout.py base 16384 128 32 *> "$o\confirm_base.out"
& $py heldout.py C 16384 128 32 ..\002-transfer\edits\c_recipe.py *> "$o\confirm_C.out"
