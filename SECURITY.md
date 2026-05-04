# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Report security issues via:

- **Email:** [liuzaoqu@163.com](mailto:liuzaoqu@163.com) (subject: `[Mugen Security]`)
- **GitHub Security Advisories:** [Report here](https://github.com/Zaoqu-Liu/Mugen/security/advisories/new)

### Response Timeline

| Stage | Timeframe |
|---|---|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 7 days |
| Fix or mitigation | Within 90 days |

We will coordinate disclosure timing with you. Credit will be given in the release notes unless you prefer anonymity.

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact assessment

## Security Release Process

1. Vulnerability confirmed and fix developed privately
2. CVE assigned (if applicable)
3. Patch release published with advisory
4. Public disclosure after users have had time to update

## Security Model

Mugen is a local inference engine. Key security considerations:

- **Model files**: Mugen loads GGUF model files from disk. Only load models from trusted sources. Maliciously crafted GGUF files could potentially exploit parsing bugs.
- **Network**: The built-in HTTP server (`mugen serve`) listens on localhost by default. Binding to `0.0.0.0` exposes the server to the network. Use with caution.
- **Dependencies**: Mugen has zero third-party dependencies. It relies only on Apple's Metal, Foundation, and POSIX frameworks.

## Best Practices

- Keep macOS and Xcode up to date
- Only download model files from trusted sources (HuggingFace verified publishers)
- Run `mugen serve` on localhost unless you understand the network implications
