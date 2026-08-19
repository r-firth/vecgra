# Security policy

Vecgra is pre-1.0 experimental software. Only the latest revision on `main` is
currently supported with security fixes; there is not yet a stable on-disk or
public API compatibility promise.

Please do not open a public issue for a suspected vulnerability. Use GitHub's
private vulnerability reporting for this repository. Include the affected
revision, platform, reproduction steps, impact, and any proposed mitigation.
Do not include live credentials or private data.

The highest-risk trust boundaries are untrusted `.vg` files, import formats,
remote-provider responses, and graph queries whose size is controlled by an
external caller. Reports involving malformed storage, out-of-bounds access,
denial of service, unsafe-code invariants, or accidental credential persistence
are especially useful.

Reports are triaged on a best-effort basis; there is no response-time service
level while the project is experimental. Please allow validation and a
coordinated fix before public disclosure.
