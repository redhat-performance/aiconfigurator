# Release Process

## Overview

AIConfigurator uses git tags with the `deploy-api-v*` prefix to trigger production container builds. Tags must be created on the `deploy/api` branch.

## Branching Strategy

- **`main`** - Upstream commits from vLLM
- **`feat/deployment`** - Deployment-specific changes
- **`deploy/api`** - Integration branch (merges main + feat/deployment + other feature and fix branches)

## Creating a Release

1. **Ensure all changes are merged to `deploy/api`**:
   ```bash
   git checkout deploy/api
   git pull
   ```

2. **Determine the version and commit hash**:
   ```bash
   # Version is extracted from wheel metadata (typically 0.11.0, 0.11.1, etc.)
   # Hash is the short commit SHA
   HASH=$(git rev-parse --short HEAD)
   echo "deploy-api-v0.11.0+${HASH}"
   ```

3. **Create and push a signed git tag on `deploy/api`**:
   ```bash
   git tag -s deploy-api-v0.11.0+${HASH} -m "Release API v0.11.0+${HASH}"
   git push origin deploy-api-v0.11.0+${HASH}
   ```
   The `-s` flag creates a signed tag for release verification. The `-m` flag provides the tag message.

4. **GitHub Actions automatically**:
   - Triggers the build workflow (`.github/workflows/build-dev.yml`)
   - Builds the container image
   - Pushes to GHCR with tags:
     - `ghcr.io/redhat-performance/aiconfigurator:0.11.0+abc1234` (specific version)
     - `ghcr.io/redhat-performance/aiconfigurator:latest` (production)

## Container Tags

- **`latest`** - Most recent `deploy-api-v*` tagged release (production)
- **`dev`** - Latest commit from deploy/api branch (auto-updated on every push)
- **`0.11.0+abc1234`** - Specific version tags (from `deploy-api-v*` git tags)

## Container Registry

Published containers: https://github.com/redhat-performance/aiconfigurator/pkgs/container/aiconfigurator

## Deployment

**Production**:
```bash
podman pull ghcr.io/redhat-performance/aiconfigurator:latest
```

**Development**:
```bash
podman pull ghcr.io/redhat-performance/aiconfigurator:dev
```

## Rollback

To rollback, deploy a previous tagged version:
```bash
podman pull ghcr.io/redhat-performance/aiconfigurator:0.11.0+abc1234
```

## Python Wheels

Python wheel releases are handled separately by `.github/workflows/deploy-api-wheels.yml` and create GitHub Releases (not git tags) with the same `deploy-api-v*+hash` naming format. These wheel releases are automatically created on every `deploy/api` push and are independent of container releases.

Container releases use git tags with the same format to trigger production builds.
