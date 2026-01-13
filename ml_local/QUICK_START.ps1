# Quick Start Training Script
# Run this to start training from scratch

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Epsilon AI Training - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "ml_local\.venv\Scripts\Activate.ps1")) {
    Write-Host "[ERROR] Virtual environment not found!" -ForegroundColor Red
    Write-Host "Run: py -m venv ml_local\.venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "[1/10] Activating virtual environment..." -ForegroundColor Green
& "ml_local\.venv\Scripts\Activate.ps1"

# Set training flag
Write-Host "[2/10] Setting LOCAL_TRAINING=1..." -ForegroundColor Green
$env:LOCAL_TRAINING="1"

# Pull corpus
Write-Host "[3/10] Pulling corpus from Supabase..." -ForegroundColor Green
Write-Host "  This may take a few minutes..." -ForegroundColor Yellow
python ml_local\scripts\pull_corpus_from_supabase.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to pull corpus!" -ForegroundColor Red
    exit 1
}

# Check if we got data
if (-not (Test-Path "ml_local\data\processed\train.txt")) {
    Write-Host "[ERROR] No training data found! Check your Supabase documents." -ForegroundColor Red
    exit 1
}

# Train tokenizer
Write-Host "[4/10] Training tokenizer..." -ForegroundColor Green
python ml_local\scripts\train_tokenizer.py --corpus ml_local\data\processed\train.txt --vocab-size 50000
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to train tokenizer!" -ForegroundColor Red
    exit 1
}

# Build token bins
Write-Host "[5/10] Building training token bins..." -ForegroundColor Green
python ml_local\scripts\build_token_bins.py --text ml_local\data\processed\train.txt --tokenizer ml_local\tokenizer\tokenizer.json --output ml_local\data\processed\train.bin
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to build train.bin!" -ForegroundColor Red
    exit 1
}

Write-Host "[6/10] Building validation token bins..." -ForegroundColor Green
python ml_local\scripts\build_token_bins.py --text ml_local\data\processed\val.txt --tokenizer ml_local\tokenizer\tokenizer.json --output ml_local\data\processed\val.bin
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to build val.bin!" -ForegroundColor Red
    exit 1
}

# Create config if needed
if (-not (Test-Path "ml_local\config.json")) {
    Write-Host "[7/10] Creating default config..." -ForegroundColor Green
    python -c "import json; config = {'vocab_size': 50000, 'n_layers': 6, 'n_heads': 6, 'd_model': 510, 'd_ff': 2048, 'max_seq_len': 512, 'dropout': 0.1, 'use_rope': True, 'use_alibi': False, 'attention_type': 'standard', 'activation': 'gelu'}; open('ml_local/config.json', 'w').write(json.dumps(config, indent=2))"
}

# Train model
Write-Host "[8/10] Starting model training..." -ForegroundColor Green
Write-Host "  THIS WILL TAKE A LONG TIME (hours or days)!" -ForegroundColor Yellow
Write-Host "  You can monitor progress in the output below." -ForegroundColor Yellow
Write-Host ""
python ml_local\train\pretrain.py --config ml_local\config.json --data ml_local\data\processed\train.bin --val-data ml_local\data\processed\val.bin --tokenizer ml_local\tokenizer\tokenizer.json --output-dir ml_local\runs --epochs 3 --batch-size 4
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Training failed!" -ForegroundColor Red
    exit 1
}

# Find the latest checkpoint
Write-Host "[9/10] Finding latest checkpoint..." -ForegroundColor Green
$checkpoints = Get-ChildItem "ml_local\runs" -Filter "checkpoint_*.pt" | Sort-Object LastWriteTime -Descending
if ($checkpoints.Count -eq 0) {
    Write-Host "[ERROR] No checkpoints found!" -ForegroundColor Red
    exit 1
}
$latestCheckpoint = $checkpoints[0].FullName
Write-Host "  Using: $latestCheckpoint" -ForegroundColor Cyan

# Export model
Write-Host "[10/10] Exporting model..." -ForegroundColor Green
python ml_local\train\export.py --checkpoint $latestCheckpoint --tokenizer ml_local\tokenizer\tokenizer.json
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Export failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next step: Deploy the model" -ForegroundColor Yellow
Write-Host "  node scripts\deploy-epsilon-model.js" -ForegroundColor Cyan
Write-Host ""

