# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in Mugen, please report it privately rather than opening a public issue.

**Do not open a public issue for security vulnerabilities.**

Please send a detailed report including:
- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact

We aim to acknowledge reports within 48 hours and provide an initial assessment within one week.

## Security Model

Mugen is a local inference engine. Key security considerations:

- **Model files**: Mugen loads GGUF model files from disk. Only load models from trusted sources. Maliciously crafted GGUF files could potentially exploit parsing bugs.
- **Network**: The built-in HTTP server (`mugen serve`) listens on localhost by default. Binding to `0.0.0.0` exposes the server to the network. Use with caution.
- **Dependencies**: Mugen has zero third-party dependencies. It relies only on Apple's Metal, Foundation, and POSIX frameworks.

## Best Practices

- Keep macOS and Xcode up to date
- Only download model files from trusted sources (HuggingFace verified publishers)
- Run `mugen serve` on localhost unless you understand the network implications
