# Interleaved so a crash leaves balanced arms: per chunk, per seed, base then edit.
$env:WS_HERE = $PSScriptRoot; $env:WS_MODEL = "Qwen/Qwen3.5-4B"
$py = "D:\WeightSpace\.venv\Scripts\python.exe"; $r = "$PSScriptRoot\run_final.py"
& $py $r plan; if ($LASTEXITCODE) { exit $LASTEXITCODE }
$plan = Get-Content "$PSScriptRoot\calls\final_plan.json" | ConvertFrom-Json
if (-not $plan.n) { "No test: $($plan.verdict)"; exit 0 }
for ($c = 0; $c -lt $plan.chunks; $c++) { foreach ($s in 0, 1000) {
  & $py $r base $c $s; if ($LASTEXITCODE) { exit $LASTEXITCODE }
  & $py $r edit $c $s edits/recipe.py; if ($LASTEXITCODE) { exit $LASTEXITCODE }
} }
