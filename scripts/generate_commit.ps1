# Get repository root directory
$RepoRoot = git rev-parse --show-toplevel
$PythonScript = Join-Path $RepoRoot "scripts\generate_commit_message.py"

# Load environment variables from .env
$EnvFile = Join-Path $RepoRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            $key = $matches[1]
            $value = $matches[2]
            Set-Item "env:$key" $value
        }
    }
}

# Set default values if not in .env
$OllamaHost = if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { "http://localhost:11434" }
$OllamaModel = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "llama3.2:3b-instruct-q8_0" }

# Create temporary diff file
$TempDiff = New-TemporaryFile

# Get staged changes
git diff --staged > $TempDiff

if (-not (Get-Content $TempDiff)) {
    Write-Host "No staged changes found."
    Remove-Item $TempDiff
    exit 1
}

try {
    # Generate message
    python $PythonScript `
        --host $OllamaHost `
        --model $OllamaModel `
        --prompt-file $TempDiff `
        > "$RepoRoot\.gitmessage"

    Write-Host "Commit message generated and saved to .gitmessage"
    Write-Host "Use 'git commit -F .gitmessage' to commit with this message"
}
catch {
    Write-Error "Failed to generate commit message: $_"
}
finally {
    Remove-Item $TempDiff -ErrorAction SilentlyContinue
}