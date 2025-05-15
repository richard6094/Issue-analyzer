# GitHub Issue Migration Tool - User Guide

## Overview

This tool allows you to retrieve GitHub issues from a repository, save them in various formats, and optionally migrate them to another repository, preserving all content, comments, and metadata.

## Requirements

- Python 3.6+
- Required Python packages:
  - requests
  - tqdm
  - argparse
- GitHub Personal Access Token with appropriate permissions

## Installation

```bash
# Clone the repository or download the scripts
git clone <repository-url>

# Install required packages
pip install requests tqdm
```

## Basic Usage

### Retrieve and Save Issues

```bash
python scripts/migrate_issues.py --source "owner/repo" --token "YOUR_TOKEN" --output "issues.json" --save-only
```

### Migrate Issues to Another Repository

```bash
python scripts/migrate_issues.py --source "owner/repo" --target "target/repo" --token "YOUR_TOKEN"
```

## Common Scenarios

### 1. Save Recent Issues

Save the 20 most recent issues to a file without migration:

```bash
python scripts/migrate_issues.py --source "OfficeDev/office-js" --token "YOUR_TOKEN" --recent 20 --output "recent_issues.json" --save-only
```

### 2. Save Issues in a Specific Range

Save issues #5000 to #5500:

```bash
python scripts/migrate_issues.py --source "OfficeDev/office-js" --token "YOUR_TOKEN" --start 5000 --end 5500 --output "specific_issues.json" --save-only
```

### 3. Save Issues from a Specific Number to Latest

Save all issues from #5000 to the most recent:

```bash
python scripts/migrate_issues.py --source "OfficeDev/office-js" --token "YOUR_TOKEN" --start 5000 --output "recent_issues.json" --save-only
```

### 4. Save Only Open Issues

```bash
python scripts/migrate_issues.py --source "OfficeDev/office-js" --token "YOUR_TOKEN" --state "open" --output "open_issues.json" --save-only
```

### 5. Save Summarized Data

Save just the essential issue information to a more compact file:

```bash
python scripts/migrate_issues.py --source "OfficeDev/office-js" --token "YOUR_TOKEN" --summary-output "issue_summary.json" --save-only
```

### 6. Save Both Full and Summary Data

```bash
python scripts/migrate_issues.py --source "OfficeDev/office-js" --token "YOUR_TOKEN" --output "full_issues.json" --summary-output "issue_summary.json" --save-only
```

### 7. Migrate Issues with Full Data

Migrate issues from one repository to another:

```bash
python scripts/migrate_issues.py --source "OfficeDev/office-js" --target "YourOrg/your-repo" --token "YOUR_TOKEN" --recent 50
```

### 8. Migrate from Saved File

Use previously saved issues data to migrate to another repo:

```bash
python scripts/migrate_issues.py --source "OfficeDev/office-js" --target "YourOrg/your-repo" --token "YOUR_TOKEN" --input "issues.json"
```

### 9. Migrate Without Migration Information

Don't add migration source info to the migrated issues:

```bash
python scripts/migrate_issues.py --source "OfficeDev/office-js" --target "YourOrg/your-repo" --token "YOUR_TOKEN" --no-migration-info
```

## Parameters Reference

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--source` | Yes | Source repository in format "owner/repo" |
| `--target` | No* | Target repository in format "owner/repo" (*Required unless --save-only is used) |
| `--token` | Yes | GitHub Personal Access Token |
| `--start` | No | Starting issue number |
| `--end` | No | Ending issue number |
| `--recent` | No | Number of most recent issues to retrieve |
| `--state` | No | Issue state: "open", "closed", or "all" (default: "all") |
| `--output` | No | Path to save full issue data as JSON |
| `--summary-output` | No | Path to save condensed issue data as JSON |
| `--input` | No | Path to load issues from a previously saved JSON file |
| `--no-migration-info` | No | Flag to disable adding migration information to migrated issues |
| `--save-only` | No | Flag to only save issues to file without migration |

## Notes and Best Practices

1. **Token Permissions**:
   - For retrieving issues: `repo:public_repo` (public repositories) or `repo` (private repositories)
   - For migrating issues: `repo` or `public_repo` for both source and target repositories

2. **Rate Limiting**:
   - GitHub API has rate limits. For large migrations, use authenticated requests (provided token)
   - If hitting rate limits, the script will display the error message

3. **Large Repositories**:
   - For repositories with many issues, use pagination parameters (`--start`, `--end`, `--recent`)
   - Process issues in batches for very large repositories

4. **File Management**:
   - Save issues before migration using `--output` to have a backup
   - Use `--summary-output` for a more compact representation of issues

5. **Migration Considerations**:
   - Migration preserves content, comments, labels, and open/closed status
   - User information is preserved as text references but not as GitHub user associations
   - Issue numbers will be different in the target repository

## Examples of File Output

### Full JSON (`--output`)
Contains complete issue data including all GitHub API fields, user information, and raw content.

### Summary JSON (`--summary-output`)
Contains just the essential information:
- Issue number, title, body
- Creation date and state
- Labels and author
- Comments with author, content, and date

## Troubleshooting

- **Authentication Errors**: Verify token permissions and validity
- **Not Found Errors**: Check repository name spelling and access permissions
- **Rate Limit Errors**: Wait for rate limit reset or use authenticated requests
- **Issues Not Migrating**: Check if target repository exists and you have write permissions

---

This tool makes it easy to archive, backup, or migrate GitHub issues while preserving all essential information and relationships.