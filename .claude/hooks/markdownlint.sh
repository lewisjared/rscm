#!/bin/bash
# Run markdownlint on markdown files after Write tool

# Read hook input from stdin
input=$(cat)

# Extract file path from hook input
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

# Exit if no file path
if [[ -z "$file_path" ]]; then
    exit 0
fi

# Only process markdown files
if [[ "$file_path" =~ \.(md|mdx)$ ]]; then
    # Run markdownlint-cli2 with auto-fix
    npx --yes markdownlint-cli2 --fix "$file_path" 2>&1 || true
fi

exit 0
