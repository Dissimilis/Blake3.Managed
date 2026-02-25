# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Blake3.Managed, please report it privately to maintainer directly rather than opening a public issue.

**Contacting maintainer:** Create a private security advisory on GitHub via the repository's Security tab.

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |

## Security Considerations

This is a **community-maintained, pure managed C#** implementation of BLAKE3. It has **not been independently audited** by a professional cryptographers.

The implementation:
- Follows the [BLAKE3 specification](https://github.com/BLAKE3-team/BLAKE3/blob/master/blake3.pdf) and passes all official test vectors
- Uses constant-time equality comparison for hash values (`CryptographicOperations.FixedTimeEquals`)
- Zeros key material on `Dispose()` for keyed hash and key derivation modes
- Uses `System.Runtime.Intrinsics` (not hand-written assembly), which inherits the .NET runtime's security properties
