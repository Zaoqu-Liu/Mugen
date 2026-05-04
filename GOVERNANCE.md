# Governance

## Model

Mugen uses a **BDFL (Benevolent Dictator for Life)** governance model.

**BDFL:** Zaoqu Liu ([@Zaoqu-Liu](https://github.com/Zaoqu-Liu))

The BDFL has final authority on all project decisions, including architecture, releases, and community policy.

## Decision Process

| Change Type | Process |
|---|---|
| Bug fix / small improvement | PR review by maintainer |
| New feature | Issue discussion → PR |
| Breaking / architectural change | Issue discussion → RFC → Implementation |
| Release | BDFL decision |

### RFC Process

For major changes:

1. Open an issue with `[RFC]` prefix describing the motivation and proposed design
2. Allow at least 7 days for community feedback
3. BDFL makes final accept/reject decision
4. Implementation proceeds via standard PR flow

## Roles

### Maintainer

- Full commit and release access
- Reviews and merges PRs
- Triages issues
- Sets project direction
- See [MAINTAINERS.md](MAINTAINERS.md) for current list

### Contributor

- Submits PRs and participates in discussions
- Follows [CONTRIBUTING.md](CONTRIBUTING.md)
- May be invited to maintainer role based on sustained, high-quality contributions

### Community Member

- Opens issues and participates in discussions
- Uses the project and provides feedback

## Releases

Version releases are at the sole discretion of the BDFL. The project follows [Semantic Versioning](https://semver.org/):

- **Major**: Breaking API/ABI changes
- **Minor**: New features, backward-compatible
- **Patch**: Bug fixes and performance improvements
