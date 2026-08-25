# ==========================================
# LiteLLM Proxy Launcher for AWS Bedrock
# File: run_litellm.ps1
# ==========================================

# 1. Prompt for your short-term API key / Token
$Token = Read-Host "Enter your short-term Bedrock API Key / Token"

if ([string]::IsNullOrWhiteSpace($Token)) {
    Write-Host "Error: Token cannot be empty!" -ForegroundColor Red
    exit
}

# 2. Set Environment Variables for AWS Bedrock (Ohio / us-east-2)
$env:AWS_DEFAULT_REGION = "us-east-2"
$env:AWS_REGION_NAME    = "us-east-2"
$env:BEDROCK_API_KEY    = $Token
$env:AWS_BEARER_TOKEN   = $Token

Write-Host "Environment variables set for AWS Region: us-east-2" -ForegroundColor Green

# 3. Activate Virtual Environment if present
$VenvPath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    . $VenvPath
} else {
    Write-Host "No .venv found in current directory, proceeding with system Python..." -ForegroundColor Yellow
}

# 4. Launch LiteLLM Proxy
Write-Host "Starting LiteLLM Proxy on http://127.0.0.1:4000..." -ForegroundColor Cyan
litellm --config litellm_config.yaml --port 4000