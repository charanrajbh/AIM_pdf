# Loads .env into this session's environment, then starts the MySQL MCP server.

# Run this from the folder containing your .env file.
 
Get-Content .env | ForEach-Object {

    if ($_ -match '^\s*([^#=]+)=(.*)$') {

        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())

    }

}
 
# mysql_mcp_server reads the port from a plain "PORT" variable, so bridge it here.

$env:PORT = $env:MYSQL_MCP_PORT
 
Write-Host "Starting MySQL MCP server on $($env:MCP_SSE_HOST):$($env:PORT) ..."
 
python -m mysql_mcp_server

 