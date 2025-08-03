#!/bin/bash

# GitHub Actions Workflows Installation Script
# This script helps repository maintainers install the workflows manually

set -e

echo "🚀 GitHub Actions Workflows Installation Script"
echo "=============================================="
echo

# Check if we're on the correct branch
current_branch=$(git branch --show-current)
if [ "$current_branch" != "terragon/implement-sdlc-checkpoints-p3k9yq" ]; then
    echo "❌ Please run this script from the terragon/implement-sdlc-checkpoints-p3k9yq branch"
    echo "   Current branch: $current_branch"
    echo "   Run: git checkout terragon/implement-sdlc-checkpoints-p3k9yq"
    exit 1
fi

echo "✅ On correct branch: $current_branch"
echo

# Check if workflow files exist
if [ ! -d ".github/workflows" ]; then
    echo "❌ Workflow files not found in .github/workflows/"
    exit 1
fi

echo "📋 Available workflow files:"
ls -la .github/workflows/
echo

# Prompt for confirmation
read -p "📤 Do you want to copy these workflows to the main branch? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "🚫 Installation cancelled"
    exit 0
fi

echo
echo "🔄 Switching to main branch..."
git checkout main

echo "📁 Creating workflows directory..."
mkdir -p .github/workflows

echo "📝 Copying workflow files..."
git checkout terragon/implement-sdlc-checkpoints-p3k9yq -- .github/workflows/
git checkout terragon/implement-sdlc-checkpoints-p3k9yq -- .github/dependabot.yml

echo
echo "📋 Files copied:"
ls -la .github/workflows/
echo

echo "✅ Workflow files successfully copied to main branch!"
echo
echo "🔧 Next steps:"
echo "1. Review the copied files"
echo "2. Configure repository secrets (see GITHUB_WORKFLOWS_IMPLEMENTATION_SUMMARY.md)"
echo "3. Set up GitHub environments"
echo "4. Commit and push changes:"
echo "   git add .github/"
echo "   git commit -m \"feat: add comprehensive GitHub Actions workflows\""
echo "   git push origin main"
echo
echo "📖 For detailed setup instructions, see:"
echo "   GITHUB_WORKFLOWS_IMPLEMENTATION_SUMMARY.md"
echo
echo "🎉 Installation complete!"