# Arm 2 generations: 0.8B, 32 fresh MATH-train problems (seed 3), 16k cap, paired. Budgeted in this folder's log.
Set-Location D:\WeightSpace\experiments\001-claude-editor
$env:PYTHONIOENCODING = 'utf-8'; $env:WS_HERE = 'D:\WeightSpace\experiments\002-transfer'
$py = 'D:\WeightSpace\.venv\Scripts\python.exe'; $o = 'D:\WeightSpace\experiments\002-transfer'
& $py run_gen.py Qwen/Qwen3.5-0.8B 3 32 16384 "A: base 0.8B @16k" "mostly loops/caps" *> "$o\A.out"
& $py run_gen.py Qwen/Qwen3.5-0.8B-wsp-edit 3 32 16384 "B: MIT/Astra published edit @16k" "large gain (MIT: +26.6 pts @16k)" *> "$o\B.out"
& $py run_gen.py Qwen/Qwen3.5-0.8B 3 32 16384 "C: recipe transfer, own doubt u" "helps, less than B" edits/c_recipe.py *> "$o\C.out"
& $py run_gen.py Qwen/Qwen3.5-0.8B 3 32 16384 "D: vector transfer via vocab bridge" "some effect; compare with F" edits/d_bridge.py *> "$o\D.out"
& $py run_gen.py Qwen/Qwen3.5-0.8B 3 32 16384 "E: random control (C's columns and norm)" "no effect" edits/e_control.py *> "$o\E.out"
& $py run_gen.py Qwen/Qwen3.5-0.8B 3 32 16384 "F: lexical doubt-commit direction" "lexical-only baseline for D" edits/f_lexical.py *> "$o\F.out"
