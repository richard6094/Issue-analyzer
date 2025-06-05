# Smart Function Dispatcher - GitHub Actions Integration

## Overview

This implementation integrates the smart function dispatcher system with GitHub Actions, similar to the existing regression analysis functionality. Instead of running as a local webhook server, the system operates entirely within GitHub Actions workflows triggered by GitHub events.

## Architecture

### GitHub Actions Based Design

The intelligent dispatcher runs as a GitHub Actions workflow that is triggered by:
- **New Issues** (`issues: opened`)
- **Issue Updates** (`issues: edited`, `issues: labeled`, `issues: unlabeled`)
- **Comment Activity** (`issue_comment: created`, `issue_comment: edited`, `issue_comment: deleted`)
- **Manual Dispatch** (`workflow_dispatch`)

### Workflow Components

1. **Intelligent Dispatcher Workflow** (`.github/workflows/intelligent-dispatcher.yml`)
   - Triggered by issue events and manual dispatch
   - Executes the complete intelligent function dispatcher pipeline
   - Generates analysis results and takes appropriate actions automatically

## Setup Instructions

### 1. Required GitHub Secrets

Configure the following secrets in your repository settings:

#### Azure OpenAI Configuration (Primary)
```
AZURE_CLIENT_ID=<your-azure-client-id>
AZURE_TENANT_ID=<your-azure-tenant-id>
AZURE_SUBSCRIPTION_ID=<your-azure-subscription-id>
AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=<your-gpt-4o-deployment-name>
```

#### OpenAI Configuration (Fallback - Optional)
```
OPENAI_API_KEY=<your-openai-api-key>
```

### 2. Workflow Files

Ensure this workflow file is present in your repository:

- `.github/workflows/intelligent-dispatcher.yml` - Main intelligent dispatcher workflow

### 3. Required Scripts

The following script must be present in your `scripts/` directory:

- `scripts/intelligent_dispatch_action.py` - Main intelligent dispatcher logic

## Features

### Intelligent Issue Analysis

The system performs comprehensive analysis of GitHub issues including:

1. **Trigger Validation**
   - Validates issue content length and quality
   - Filters out bot-created issues (unless forced)
   - Checks for existing analysis labels

2. **Intent Analysis**
   - Determines issue type (bug report, feature request, question, etc.)
   - Assesses urgency level and complexity
   - Generates confidence scores

3. **Function Execution**
   - **RAG Analysis**: Searches for similar issues in vector database
   - **Image Analysis**: Analyzes screenshots and visual content
   - **Regression Analysis**: Determines if issue is a regression bug
   - **Template Suggestions**: Recommends appropriate issue templates

4. **Decision Making & Actions**
   - Automatically applies appropriate labels
   - Adds analysis summary comments
   - Handles uncertain cases with user confirmation

### Supported Issue Types

- **Bug Report** → `bug` label
- **Feature Request** → `enhancement` label  
- **Question** → `question` label
- **Documentation** → `documentation` label
- **Regression** → `regression` + `bug` labels
- **Support Request** → `question` + `help wanted` labels

### Priority and Complexity Labels

- **Priority**: `priority:critical`, `priority:high`
- **Complexity**: `complexity:high`

## Usage

### Automatic Processing

The system automatically processes issues when:

1. **New Issue Created**: Analyzes and labels new issues immediately
2. **Issue Updated**: Re-analyzes when issue content changes
3. **Comments Added**: Monitors for additional context

### Manual Processing

Trigger manual analysis using workflow dispatch:

1. Go to **Actions** tab in your repository
2. Select **Smart Function Dispatcher** workflow
3. Click **Run workflow**
4. Optionally specify:
   - `issue_number`: Specific issue to analyze
   - `force_analysis`: Re-analyze even if already processed

### User Feedback for Uncertain Cases

When the system cannot determine issue type with confidence:

1. **Uncertainty Comment**: System posts a comment with explanation
2. **User Confirmation**: User checks a box to confirm issue type
3. **Automatic Processing**: System detects the confirmation and applies labels
4. **Feedback Lock**: Selection is locked with visual confirmation

Example uncertainty comment:
```markdown
## Regression Analysis Needs Your Input

Our automated system analyzed this issue but couldn't determine with confidence if this is a regression bug.

**If you believe this is a regression issue, please confirm below:**

- [ ] Yes, confirm this is a regression issue

> Note: Once confirmed, the issue will be labeled as regression.
```

## Results and Artifacts

### Analysis Results

Each workflow run produces:

1. **JSON Results** (`smart_dispatch_results.json`)
   - Complete analysis results with timestamps
   - Intent analysis with confidence scores
   - Function execution results
   - Actions taken and recommendations

2. **Log Files** (`smart_dispatch_logs.txt`)
   - Detailed execution logs
   - Error messages and debugging information

### GitHub Actions Integration

- **Artifacts**: Results are uploaded as workflow artifacts
- **Status Checks**: Workflow status indicates analysis success/failure
- **Comments**: Analysis summary posted as issue comments
- **Labels**: Automatic label application based on analysis

## Configuration

### Environment Variables

The following environment variables are automatically set by GitHub Actions:

```bash
GITHUB_TOKEN=<automatic-github-token>
GITHUB_EVENT_NAME=<event-type>
GITHUB_EVENT_ACTION=<event-action>
ISSUE_NUMBER=<issue-number>
REPO_OWNER=<repository-owner>
REPO_NAME=<repository-name>
REPO_FULL_NAME=<owner/repo>
SENDER_LOGIN=<event-sender>
SENDER_TYPE=<User|Bot>
```

### Feature Flags

Control system behavior through code configuration:

```python
# Trigger thresholds
MIN_ISSUE_TITLE_LENGTH = 5
MIN_ISSUE_BODY_LENGTH = 10

# Analysis features
ENABLE_RAG_ANALYSIS = True
ENABLE_IMAGE_ANALYSIS = True
ENABLE_REGRESSION_ANALYSIS = True

# Processing limits
MAX_IMAGES_PER_ISSUE = 3
MAX_CONTEXT_LENGTH = 1000
```

## Integration with Existing Systems

### Regression Analysis

The intelligent dispatcher integrates with the existing regression analysis system:

- Reuses `analyze_issue_for_regression()` function
- Maintains compatibility with existing feedback processing
- Uses same uncertainty handling workflow

### RAG Database

Leverages existing RAG infrastructure:

- Uses `query_vectordb()` for similar issue search
- Integrates with ChromaDB for vector similarity
- Maintains existing vector database structure

### LLM Providers

Utilizes existing LLM provider infrastructure:

- Supports both Azure OpenAI and OpenAI
- Reuses authentication and configuration patterns
- Maintains consistent model selection logic

## Monitoring and Debugging

### Workflow Monitoring

1. **Actions Tab**: Monitor workflow execution status
2. **Artifacts**: Download analysis results and logs
3. **Issue Comments**: Review posted analysis summaries

### Common Issues

1. **Authentication Failures**
   - Verify Azure credentials and permissions
   - Check API key validity and quotas

2. **Analysis Errors**
   - Review workflow logs for error details
   - Check issue content for problematic characters

3. **Label Application Issues**
   - Verify GitHub token permissions
   - Ensure labels exist in repository

### Debugging Tools

1. **Manual Dispatch**: Test specific issues
2. **Force Analysis**: Override existing analysis
3. **Log Artifacts**: Detailed execution logs
4. **JSON Results**: Complete analysis output

## Comparison with Local Webhook Server

| Aspect | GitHub Actions | Local Webhook Server |
|--------|----------------|----------------------|
| **Deployment** | No infrastructure needed | Requires server hosting |
| **Scalability** | Automatic scaling | Manual scaling required |
| **Reliability** | GitHub SLA | Self-managed uptime |
| **Security** | GitHub-managed | Self-managed security |
| **Configuration** | Repository secrets | Environment variables |
| **Monitoring** | GitHub Actions UI | Custom monitoring |
| **Cost** | GitHub Actions quota | Server hosting costs |

## Future Enhancements

1. **Enhanced Analytics**
   - Issue trend analysis
   - Label accuracy metrics
   - Response time tracking

2. **Advanced Integration**
   - Pull request analysis
   - Project board automation
   - Notification routing

3. **Machine Learning**
   - Custom model training
   - Feedback loop integration
   - Accuracy improvement

## Troubleshooting

### Common Error Messages

1. **"Missing required environment variables"**
   - Check GitHub secrets configuration
   - Verify workflow environment variable mapping

2. **"Failed to fetch issue data"**
   - Check GitHub token permissions
   - Verify issue number format

3. **"LLM analysis failed"**
   - Check Azure OpenAI service availability
   - Verify API quotas and rate limits

### Support

For issues and questions:

1. Check workflow execution logs
2. Review analysis result artifacts
3. Examine issue comments for error details
4. Verify configuration and permissions
