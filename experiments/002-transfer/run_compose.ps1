# Composition test (pre-registered in protocol.md): B, C, BC on 64 MATH-train problems (seed 4) @16k. Budgeted.
Set-Location D:\WeightSpace\experiments\001-claude-editor
$env:PYTHONIOENCODING = 'utf-8'; $env:WS_HERE = 'D:\WeightSpace\experiments\002-transfer'
$py = 'D:\WeightSpace\.venv\Scripts\python.exe'; $o = 'D:\WeightSpace\experiments\002-transfer'
& $py run_gen.py Qwen/Qwen3.5-0.8B-wsp-edit 4 64 16384 "COMPOSE B: MIT edit" "loops gone, still uncommitted" *> "$o\compose_B.out"
& $py run_gen.py Qwen/Qwen3.5-0.8B 4 64 16384 "COMPOSE C: doubt edit" "commits more, loops remain" edits/c_recipe.py *> "$o\compose_C.out"
& $py run_gen.py Qwen/Qwen3.5-0.8B-wsp-edit 4 64 16384 "COMPOSE BC: MIT edit + doubt edit" "BC > B (indecision is the root)" edits/c_recipe.py *> "$o\compose_BC.out"
