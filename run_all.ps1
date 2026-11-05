# Run the whole pipeline end-to-end.
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

Write-Host "=== 1/9 generate data ===" -ForegroundColor Cyan
& $py generate_data.py

Write-Host "=== 2/9 EDA ===" -ForegroundColor Cyan
& $py eda.py

Write-Host "=== 3/9 baseline ===" -ForegroundColor Cyan
& $py baseline.py

Write-Host "=== 4/9 linear + ridge + lasso ===" -ForegroundColor Cyan
& $py model_linear.py

Write-Host "=== 5/9 lightgbm ===" -ForegroundColor Cyan
& $py model_lgbm.py

Write-Host "=== 6/9 prophet ===" -ForegroundColor Cyan
& $py model_prophet.py

Write-Host "=== 7/9 lstm ===" -ForegroundColor Cyan
& $py model_lstm.py

Write-Host "=== 8/9 bayesian ===" -ForegroundColor Cyan
& $py model_bayesian.py

Write-Host "=== 9/9 ensemble + summary ===" -ForegroundColor Cyan
& $py ensemble.py
& $py summarize.py

Write-Host "`nAll done. See reports/ folder." -ForegroundColor Green
