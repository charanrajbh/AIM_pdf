# Loads .env into this session's environment, then starts the MongoDB MCP server.

# Run this from the folder containing your .env file.
 
Get-Content .env | ForEach-Object {

    if ($_ -match '^\s*([^#=]+)=(.*)$') {

        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())

    }

}
 
Write-Host "Starting MongoDB MCP server on $($env:MONGO_MCP_HOST):$($env:MONGO_MCP_PORT) ..."
 
$mongoArgs = @(

    "-y"

    "mongodb-mcp-server@latest"

    "--transport"

    "http"

    "--httpHost"

    $env:MONGO_MCP_HOST

    "--httpPort"

    $env:MONGO_MCP_PORT

)
 
& npx @mongoArgs
 