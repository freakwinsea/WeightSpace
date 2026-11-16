$env:WS_HERE = $PSScriptRoot; $env:WS_MODEL = "Qwen/Qwen3.5-4B"
$py = "D:\WeightSpace\.venv\Scripts\python.exe"
& $py "$PSScriptRoot\run_calib.py" base; if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $py "$PSScriptRoot\run_calib.py" edit edits/recipe.py
