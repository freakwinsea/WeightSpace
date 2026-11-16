# Every 10 min: aggregate decode speed on llama-server and per-process GPU memory -> gpuwatch.log. Diagnostic only.
$log = 'D:\WeightSpace\experiments\003-scale\gpuwatch.log'
function Decoded { try { $s = Invoke-RestMethod http://127.0.0.1:8090/slots -TimeoutSec 5; ($s | ForEach-Object { $n = $_.next_token; if ($n -is [array]) { $n = $n[0] }; [int]$n.n_decoded } | Measure-Object -Sum).Sum } catch { $null } }
while ($true) {
  $a = Decoded; Start-Sleep 60; $b = Decoded
  $rate = if ($a -ne $null -and $b -ne $null -and $b -ge $a) { [int](($b - $a) / 60) } else { 'n/a' }
  $ded = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples | Where-Object CookedValue -gt 200MB
  $sh  = (Get-Counter '\GPU Process Memory(*)\Shared Usage').CounterSamples | Where-Object CookedValue -gt 200MB
  $fmt = { param($x) ($x | ForEach-Object { $p = [regex]::Match($_.InstanceName, 'pid_(\d+)').Groups[1].Value; '{0}:{1}={2:N1}G' -f (Get-Process -Id $p -EA SilentlyContinue).ProcessName, $p, ($_.CookedValue/1GB) }) -join ' ' }
  "{0:MM-dd HH:mm} tok/s={1} | dedicated {2} | shared {3}" -f (Get-Date), $rate, (& $fmt $ded), (& $fmt $sh) | Add-Content $log
  Start-Sleep 540
}
