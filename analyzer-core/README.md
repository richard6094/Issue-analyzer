# Intelligent Function Dispatcher System

This directory contains the core components and documentation for the GitHub Issue Agent's intelligent function dispatcher system, implementing advanced AI-driven trigger processing, intent analysis, intelligent function dispatching, and decision making.

## Architecture Overview

The intelligent dispatcher system consists of the following components:

## Overview

The Intelligent Function Dispatcher system is now fully integrated with GitHub Actions and operates automatically when issues are created or updated in your repository. The system no longer requires local deployment or webhook setup.

## GitHub Actions Integration

The system operates through the following workflow file:
- `.github/workflows/intelligent-dispatcher.yml` - Main workflow for issue analysis and intelligent dispatch

The dispatcher automatically:
- Analyzes new issues for intent and priority
- Applies appropriate labels
- Provides intelligent responses
- Routes complex issues to support personnel

## Configuration

The system is configured through GitHub Actions workflow files and repository secrets. Key configuration includes:

### Repository Secrets

Set the following secrets in your GitHub repository:

- **GITHUB_TOKEN** - GitHub personal access token with repo permissions
- **LLM_PROVIDER** - Either "azure_openai" or "openai"
- **AZURE_OPENAI_ENDPOINT** - (For Azure OpenAI) Your endpoint URL
- **AZURE_OPENAI_DEPLOYMENT_NAME** - (For Azure OpenAI) Model deployment name
- **AZURE_OPENAI_API_VERSION** - (For Azure OpenAI) API version (e.g., "2024-02-01")
- **OPENAI_API_KEY** - (For OpenAI) Your OpenAI API key
- **AZURE_COMPUTER_VISION_ENDPOINT** - (Optional) For image analysis
- **AZURE_COMPUTER_VISION_KEY** - (Optional) For image analysis

## Integration with Existing Modules

The intelligent dispatcher integrates with existing project modules:

- **LLM Module** (`../LLM/`) - Language model providers
- **RAG Module** (`../RAG/`) - Vector database queries
- **Image Recognition** (`../image_recognition/`) - Image analysis
- **Scripts** (`../scripts/`) - Issue analysis logic

## Testing

### Local Testing

For local development and testing, you can use the GitHub Actions dispatcher directly:

```powershell
cd scripts
python intelligent_dispatch_action.py
```

## Production Deployment

The Smart Function Dispatcher system now operates exclusively through GitHub Actions for automated issue analysis and processing. The local Docker deployment has been deprecated in favor of the GitHub Actions integration.

### Environment Variables for Production

Set all required environment variables and ensure proper logging:

```powershell
$env:LOG_LEVEL = "INFO"
LOG_FILE_PATH=logs/intelligent_dispatcher.log
```

## Monitoring and Logging

The system provides comprehensive logging:

- **Console logging** for development
- **File logging** for production
- **Structured logging** for log aggregation
- **Health check endpoint** for monitoring

## Troubleshooting

### Common Issues

1. **GitHub Token Issues**: Ensure token has proper permissions (issues, comments, labels)
2. **LLM Provider Issues**: Verify endpoint and API key configuration
3. **Vector DB Issues**: Ensure ChromaDB path exists and is accessible
4. **Webhook Issues**: Check webhook secret and payload format

### Debug Mode

Enable debug logging:
```powershell
python main.py --debug --mode server
```

## Contributing

1. Follow the existing code structure and patterns
2. Add appropriate error handling and logging
3. Update tests for new functionality
4. Update this README for new features

## License

This project is part of the GitHub Issue Analyzer system.
