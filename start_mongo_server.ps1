# Loads .env into this session's environment, then starts the MongoDB MCP server.
# Run this from the folder containing your .env file.

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
    Write-Host "ERROR: no .env file in $(Get-Location). Run this from the folder containing it." -ForegroundColor Red
    exit 1
}

# 1. Load .env into this process (and therefore into child processes).
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        $name  = $matches[1].Trim()
        $value = $matches[2].Trim().Trim('"').Trim("'")
        [System.Environment]::SetEnvironmentVariable($name, $value)
    }
}

$ConnectionString = $env:MDB_MCP_CONNECTION_STRING
$MongoHost = $env:MONGO_MCP_HOST
$MongoPort = $env:MONGO_MCP_PORT

if ([string]::IsNullOrWhiteSpace($ConnectionString)) {
    Write-Host "ERROR: MDB_MCP_CONNECTION_STRING is empty or missing from .env." -ForegroundColor Red
    Write-Host "Without it the server starts and lists its tools, but every query fails." -ForegroundColor Red
    exit 1
}
if ([string]::IsNullOrWhiteSpace($MongoHost)) { $MongoHost = "127.0.0.1" }
if ([string]::IsNullOrWhiteSpace($MongoPort)) { $MongoPort = "8002" }

# Mask credentials when echoing the connection string back.
$Masked = $ConnectionString -replace '://[^@/]+@', '://***:***@'
Write-Host "Connection string: $Masked" -ForegroundColor Green

# 2. Preflight: is mongod actually accepting connections?
if ($ConnectionString -match '^mongodb(\+srv)?://(?:[^@/]+@)?([^:/,?]+)(?::(\d+))?') {
    $DbHost = $matches[2]
    if ($matches[3]) { $DbPort = [int]$matches[3] } else { $DbPort = 27017 }

    if ($matches[1]) {
        Write-Host "Note: mongodb+srv URI -- skipping the direct TCP check." -ForegroundColor Yellow
    } else {
        Write-Host "Checking mongod at ${DbHost}:${DbPort} ..." -ForegroundColor Cyan
        $tcp = New-Object System.Net.Sockets.TcpClient
        $reachable = $false
        try {
            $async = $tcp.BeginConnect($DbHost, $DbPort, $null, $null)
            if ($async.AsyncWaitHandle.WaitOne(3000, $false) -and $tcp.Connected) {
                $tcp.EndConnect($async)
                $reachable = $true
            }
        } catch {
            $reachable = $false
        } finally {
            $tcp.Close()
        }

        if ($reachable) {
            Write-Host "mongod is reachable." -ForegroundColor Green
        } else {
            Write-Host "ERROR: nothing is listening on ${DbHost}:${DbPort}." -ForegroundColor Red
            Write-Host "Start MongoDB first, then re-run this script." -ForegroundColor Red
            Write-Host "The MCP server would otherwise start, list its tools, and fail every query." -ForegroundColor Red
            exit 1
        }
    }
} else {
    Write-Host "Note: could not parse a host/port from the connection string; skipping the TCP check." -ForegroundColor Yellow
}

# 3. Start the MongoDB MCP server.
# 3. Start the MongoDB MCP server.
Write-Host "Starting MongoDB MCP server on ${MongoHost}:${MongoPort} ..." -ForegroundColor Cyan

$mongoArgs = @(
    "-y"
    "mongodb-mcp-server@latest"
    $ConnectionString
    "--transport"
    "http"
    "--httpHost"
    $MongoHost
    "--httpPort"
    $MongoPort
)

if ($env:MDB_MCP_READ_ONLY -eq "true") {
    $mongoArgs += "--readOnly"
    Write-Host "Read-only mode enabled." -ForegroundColor Yellow
}

& npx @mongoArgs
