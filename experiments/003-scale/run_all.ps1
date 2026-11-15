# Exp 003 chain (pre-registered in protocol.md): calibration gen -> forced doubt call -> eval base -> eval edit.
Set-Location D:\WeightSpace\experiments\001-claude-editor
$env:PYTHONIOENCODING = 'utf-8'; $env:WS_HERE = 'D:\WeightSpace\experiments\003-scale'; $env:WS_MODEL = 'Qwen/Qwen3.5-4B'
$py = 'D:\WeightSpace\.venv\Scripts\python.exe'; $o = 'D:\WeightSpace\experiments\003-scale'
if (-not (Test-Path "$o\calls\call001_gen.json")) {
  & $py run_gen.py Qwen/Qwen3.5-4B 1 32 4096 "CALL 1: base 4B calibration texts @4k" "indecision: finished answers correct, capped often contain gold" *> "$o\call1.out" }
if (-not (Test-Path "$o\calls\call002_forced_doubt.pt")) {
  & $py call_doubt.py calls/call001_gen.json 32 "CALL 2: 4B doubt direction" "top MLP writers at relative depth 0.62-0.75 (L20-24)" *> "$o\call2.out" }
& $py "$o\run_eval.py" base *> "$o\eval_base.out"
& $py "$o\run_eval.py" edit edits/recipe.py *> "$o\eval_edit.out"
