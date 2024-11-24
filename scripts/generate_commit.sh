#!/bin/bash

REPO_ROOT=$(git rev-parse --show-toplevel)
PYTHON_SCRIPT="${REPO_ROOT}/scripts/generate_commit_message.py"

# Load environment variables
if [ -f "${REPO_ROOT}/.env" ]; then
    export $(cat "${REPO_ROOT}/.env" | xargs)
fi

# Set default values if not provided in .env
OLLAMA_HOST=${OLLAMA_HOST:-"http://localhost:11434"}
OLLAMA_MODEL=${OLLAMA_MODEL:-"llama3.2:3b-instruct-q8_0"}

# Create temporary diff file
TEMP_DIFF=$(mktemp)
git diff --staged > "$TEMP_DIFF"

if [ ! -s "$TEMP_DIFF" ]; then
    echo "No staged changes found."
    rm -f "$TEMP_DIFF"
    exit 1
fi

# Generate message
python "$PYTHON_SCRIPT" \
    --host "$OLLAMA_HOST" \
    --model "$OLLAMA_MODEL" \
    --prompt-file "$TEMP_DIFF" \
    > "${REPO_ROOT}/.gitmessage"

rm -f "$TEMP_DIFF"
echo "Commit message generated and saved to .gitmessage"
echo "Use 'git commit -F .gitmessage' to commit with this message"