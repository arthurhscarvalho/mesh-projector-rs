#!/bin/bash

command1="ruff format" # Format the code
command2="ruff check --fix" # Look for minor linting problems and fix them
command3="ruff check --fix --select I" # Sort imports

# Find and format all python files in the repository
find . -type f -name "*.py" | while read file; do
    echo "Applying command to: $file"
    $command1 "$file"
    $command2 "$file"
    $command3 "$file"
done

echo "All Python files processed."
