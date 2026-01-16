# 🔒 NeuralOps Security Hardening Guide

## Overview
This document outlines the comprehensive security measures implemented to make the NeuralOps system as secure as possible against unauthorized access and attacks.

## Security Layers Implemented

### 1. **Content Security Policy (CSP) with Nonces**
- **Status**: ✅ Implemented
- **Location**: `runtime/security-hardening.js`
- **Features**:
  - Removed `unsafe-inline` and `unsafe-eval` from CSP
  - Uses cryptographic nonces for script execution
  - Strict policy that only allows trusted sources
- **How it works**: Each request generates a unique nonce that must match between the CSP header and script tags

### 2. **Anomaly Detection System**
- **Status**: ✅ Implemented
- **Location**: `runtime/security-hardening.js`
- **Detects**:
  - SQL injection attempts
  - XSS (Cross-Site Scripting) attempts
  - Path traversal attempts
  - Command injection attempts
  - Oversized requests
  - Suspicious user agents
  - Excessive query parameters
- **Response**: Blocks high-severity attacks immediately, logs medium/low severity

### 3. **IP Reputation System**
- **Status**: ✅ Implemented
- **Location**: `runtime/security-hardening.js`
- **Features**:
  - Tracks suspicious activity per IP
  - Automatically blocks IPs with 5+ suspicious activities
  - 24-hour rolling window for reputation
- **Response**: Blocks suspicious IPs automatically

### 4. **Request Fingerprinting**
- **Status**: ✅ Implemented
- **Location**: `runtime/security-hardening.js`
- **Features**:
  - Creates unique fingerprint from IP, User-Agent, Accept headers
  - Used for rate limiting and anomaly detection
  - Helps identify bot traffic and automated attacks

### 5. **Secure HTML Rendering**
- **Status**: ✅ Implemented
- **Location**: `runtime/secure-html-renderer.js`
- **Features**:
  - Replaces all `innerHTML` usage with safe alternatives
  - Uses DOMPurify for HTML sanitization
  - Provides `safeSetText()` for plain text (prevents XSS)
  - Provides `safeRenderHTML()` for trusted HTML content

### 6. **Enhanced Rate Limiting**
- **Status**: ✅ Implemented
- **Location**: `runtime/security-hardening.js`, `runtime/rate-limit.js`
- **Limits**:
  - Authentication: 10 attempts per 15 minutes
  - API endpoints: 100 requests per 5 minutes
  - Uploads: 100 per hour
  - Per IP and fingerprint combination

### 7. **Dependency Vulnerability Scanning**
- **Status**: ✅ Implemented
- **Location**: `scripts/security-scan.js`
- **Features**:
  - Automated npm audit scanning
  - Python pip-audit scanning (if available)
  - Environment variable validation
  - Security configuration checks
- **Usage**: `node scripts/security-scan.js` or `npm run security-scan`

## Security Headers

All responses include:
- `X-Content-Type-Options: nosniff` - Prevents MIME type sniffing
- `X-Frame-Options: DENY` - Prevents clickjacking
- `X-XSS-Protection: 1; mode=block` - Browser XSS protection
- `Strict-Transport-Security` - Forces HTTPS (production)
- `Content-Security-Policy` - Strict CSP with nonces
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Cross-Origin-Embedder-Policy: require-corp`
- `Cross-Origin-Opener-Policy: same-origin`

## Authentication & Authorization

### JWT Security
- ✅ Tokens signed with strong secret (32+ characters)
- ✅ HTTP-only cookies in production
- ✅ Secure flag in production
- ✅ SameSite=Strict to prevent CSRF
- ✅ Token expiration enforced
- ✅ Database role verification for owner access

### Account Protection
- ✅ Account lockout after 5 failed login attempts
- ✅ 15-minute lockout period
- ✅ Password strength requirements (8+ chars, uppercase, lowercase, numbers)
- ✅ Rate limiting on authentication endpoints

## Input Validation & Sanitization

### All User Input
- ✅ HTML sanitization (removes scripts, dangerous tags)
- ✅ SQL injection prevention (parameterized queries via Supabase)
- ✅ Path traversal prevention
- ✅ XSS prevention (multiple layers)
- ✅ Length limits (DoS prevention)
- ✅ Type validation

### Sanitization Functions
- `sanitizeText()` - Removes HTML, scripts, dangerous characters
- `sanitizeHTML()` - Sanitizes HTML while preserving safe formatting
- `sanitizeSQL()` - Removes SQL injection patterns
- `sanitizeFilename()` - Prevents path traversal in file operations

## Data Protection

### Encryption
- ✅ AES-256-GCM encryption at rest
- ✅ HTTPS/TLS in transit (enforced)
- ✅ Encryption keys stored in environment variables
- ✅ Fail-secure: Returns null if encryption fails (prevents data leakage)

### Database Security
- ✅ Row-Level Security (RLS) enabled on all tables
- ✅ User isolation (conversations filtered by user_id)
- ✅ Parameterized queries (no raw SQL)
- ✅ Service role key never exposed to client

### PII Protection
- ✅ PII redaction before AI training
- ✅ Email, phone, SSN, credit card patterns detected and redacted
- ✅ IP addresses redacted in logs

## Network Security

### CORS
- ✅ Specific allowed origins (no wildcards)
- ✅ Credentials only for trusted origins
- ✅ Preflight request validation

### CSRF Protection
- ✅ CSRF tokens required for state-changing operations
- ✅ Double-submit cookie pattern
- ✅ Token validation on every POST/PUT/DELETE

## Monitoring & Logging

### Security Event Logging
- ✅ All authentication failures logged
- ✅ All blocked requests logged
- ✅ All anomalies detected logged
- ✅ IP reputation changes logged
- ✅ Rate limit violations logged

### Log Files
- `logs/security.log` - All security events
- `logs/critical-security.log` - High-severity events only
- `logs/app.log` - General application logs
- `logs/error.log` - Error logs

## Regular Security Tasks

### Daily
- ✅ Review security logs for anomalies
- ✅ Check for blocked IPs
- ✅ Monitor rate limit violations

### Weekly
- ✅ Run dependency scans: `npm run security-scan`
- ✅ Review and update blocked IP list
- ✅ Check for new security advisories

### Monthly
- ✅ Run full security audit: `npm audit` and `pip-audit`
- ✅ Update dependencies with security patches
- ✅ Review and update security policies
- ✅ Test incident response procedures

## Attack Vectors - Protection Status

| Attack Vector | Protection | Status |
|--------------|-----------|--------|
| SQL Injection | Parameterized queries, input sanitization | ✅ Protected |
| XSS (Cross-Site Scripting) | CSP nonces, HTML sanitization, input validation | ✅ Protected |
| CSRF | CSRF tokens, SameSite cookies | ✅ Protected |
| Brute Force Login | Account lockout, rate limiting | ✅ Protected |
| DDoS | Rate limiting, request size limits | ✅ Protected |
| Path Traversal | Filename sanitization, path validation | ✅ Protected |
| Command Injection | Input sanitization, no shell execution | ✅ Protected |
| Session Hijacking | HTTP-only cookies, HTTPS, JWT expiration | ✅ Protected |
| Dependency Exploits | Automated scanning, regular updates | ✅ Protected |

## Security Best Practices

### For Developers
1. **Never use `innerHTML` with user content** - Use `safeSetText()` or `safeRenderHTML()`
2. **Always validate and sanitize input** - Use functions from `runtime/sanitize.js`
3. **Never log sensitive data** - Passwords, tokens, keys should never appear in logs
4. **Use environment variables** - Never hardcode secrets
5. **Run security scans** - Before deploying, run `npm run security-scan`

### For Deployment
1. **Set all required environment variables** - See `SECURITY_HARDENING.md`
2. **Enable HTTPS** - Never run in production without TLS
3. **Set strong secrets** - JWT_SECRET and ENCRYPTION_KEY should be 32+ random characters
4. **Monitor logs** - Set up alerts for security events
5. **Keep dependencies updated** - Run `npm audit fix` regularly

## Incident Response

### If Attack Detected
1. **Check security logs** - `logs/security.log` and `logs/critical-security.log`
2. **Review blocked IPs** - Check if IP reputation system blocked the attacker
3. **Review anomalies** - Check what patterns were detected
4. **Check rate limits** - See if rate limiting prevented the attack
5. **Update security rules** - If new attack pattern detected, add to anomaly detection

### Emergency Actions
- **Block IP immediately**: Add to `ipReputation.suspiciousIPs` in `security-hardening.js`
- **Increase rate limits**: Temporarily lower limits in `rate-limit.js`
- **Disable affected endpoints**: Comment out routes in `server.js` if needed
- **Rotate secrets**: Change JWT_SECRET and ENCRYPTION_KEY immediately

## Security Checklist

Before deploying to production:
- [ ] All environment variables set
- [ ] Security scan passed: `npm run security-scan`
- [ ] No vulnerabilities in dependencies
- [ ] HTTPS enabled
- [ ] Strong secrets configured (32+ characters)
- [ ] Security logging enabled
- [ ] Rate limiting configured
- [ ] CSP nonces working (check browser console)
- [ ] All `innerHTML` replaced with safe alternatives
- [ ] Input sanitization tested
- [ ] CSRF protection tested

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CSP Nonce Guide](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Security-Policy/script-src#unsafe_inline_script)
- [npm audit documentation](https://docs.npmjs.com/cli/v8/commands/npm-audit)
- [pip-audit documentation](https://pypi.org/project/pip-audit/)

## Support

For security concerns or to report vulnerabilities, contact the security team immediately.

---

**Last Updated**: 2024
**Version**: 1.0
**Status**: Production Ready

