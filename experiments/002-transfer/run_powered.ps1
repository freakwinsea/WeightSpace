# Powered B vs BC (pre-registered in protocol.md). 8 budgeted calls in total.
$env:PYTHONIOENCODING = 'utf-8'; $env:WS_HERE = 'D:\WeightSpace\experiments\002-transfer'
$py = 'D:\WeightSpace\.venv\Scripts\python.exe'; $o = 'D:\WeightSpace\experiments\002-transfer'
& $py "$o\run_powered.py" B Qwen/Qwen3.5-0.8B-wsp-edit *> "$o\powered_B.out"
& $py "$o\run_powered.py" BC Qwen/Qwen3.5-0.8B-wsp-edit edits/c_recipe.py *> "$o\powered_BC.out"
