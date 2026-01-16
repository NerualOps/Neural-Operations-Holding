// security-hardening.js
// Comprehensive security hardening middleware and utilities
const crypto = require('crypto');
const { logSecurityEvent } = require('./logging');

// ============================================================================
// 1. CSP NONCE GENERATION (Removes need for unsafe-inline)
// ============================================================================
const generateNonce = () => {
  return crypto.randomBytes(16).toString('base64');
};

// Store nonce per request (middleware will attach to req)
const attachNonce = (req, res, next) => {
  req.nonce = generateNonce();
  res.locals.nonce = req.nonce;
  next();
};

// ============================================================================
// 2. REQUEST FINGERPRINTING (Detect suspicious patterns)
// ============================================================================
const requestFingerprint = (req) => {
  const components = [
    req.ip,
    req.get('user-agent') || '',
    req.get('accept-language') || '',
    req.get('accept-encoding') || '',
    req.get('accept') || ''
  ];
  
  const fingerprint = crypto
    .createHash('sha256')
    .update(components.join('|'))
    .digest('hex');
  
  return fingerprint;
};

// ============================================================================
// 3. ANOMALY DETECTION (Detect suspicious behavior)
// ============================================================================
const suspiciousPatterns = {
  sqlInjection: [
    /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION)\b)/gi,
    /(--|\/\*|\*\/|;)/g,
    /(\bOR\b\s+\d+\s*=\s*\d+)/gi,
    /(\bAND\b\s+\d+\s*=\s*\d+)/gi
  ],
  xss: [
    /<script[^>]*>/gi,
    /javascript:/gi,
    /on\w+\s*=/gi,
    /<iframe[^>]*>/gi,
    /<object[^>]*>/gi,
    /<embed[^>]*>/gi
  ],
  pathTraversal: [
    /\.\.\//g,
    /\.\.\\/g,
    /\/etc\/passwd/gi,
    /\/proc\/self\/environ/gi,
    /windows\/system32/gi
  ],
  commandInjection: [
    /[;&|`$(){}[\]]/g,
    /\b(cat|ls|pwd|whoami|id|uname|ps|netstat)\b/gi
  ]
};

const detectAnomalies = (req) => {
  const anomalies = [];
  const url = req.url || '';
  const body = JSON.stringify(req.body || {});
  const query = JSON.stringify(req.query || {});
  const combined = url + body + query;
  
  // Check for suspicious patterns
  Object.entries(suspiciousPatterns).forEach(([type, patterns]) => {
    patterns.forEach(pattern => {
      if (pattern.test(combined)) {
        anomalies.push({
          type,
          pattern: pattern.toString(),
          severity: type === 'sqlInjection' || type === 'commandInjection' ? 'high' : 'medium'
        });
      }
    });
  });
  
  // Check for unusual request characteristics
  if (req.headers['content-length'] && parseInt(req.headers['content-length']) > 10 * 1024 * 1024) {
    anomalies.push({ type: 'oversizedRequest', severity: 'medium' });
  }
  
  if (req.url && req.url.length > 2048) {
    anomalies.push({ type: 'oversizedUrl', severity: 'medium' });
  }
  
  // Check for missing or suspicious user agent
  const userAgent = req.get('user-agent') || '';
  if (!userAgent || userAgent.length < 10) {
    anomalies.push({ type: 'suspiciousUserAgent', severity: 'low' });
  }
  
  // Check for too many query parameters (potential DoS)
  if (Object.keys(req.query || {}).length > 50) {
    anomalies.push({ type: 'excessiveQueryParams', severity: 'medium' });
  }
  
  return anomalies;
};

// ============================================================================
// 4. IP REPUTATION CHECK (Basic - can be enhanced with external services)
// ============================================================================
const ipReputation = {
  suspiciousIPs: new Map(), // IP -> { count, firstSeen, lastSeen, reasons }
  threshold: 5 // Number of suspicious activities before blocking
};

const checkIPReputation = (ip) => {
  const record = ipReputation.suspiciousIPs.get(ip);
  if (!record) return { suspicious: false, score: 0 };
  
  const score = record.count;
  const suspicious = score >= ipReputation.threshold;
  
  return { suspicious, score, reasons: record.reasons || [] };
};

const recordSuspiciousActivity = (ip, reason) => {
  const now = Date.now();
  const record = ipReputation.suspiciousIPs.get(ip) || {
    count: 0,
    firstSeen: now,
    lastSeen: now,
    reasons: []
  };
  
  record.count++;
  record.lastSeen = now;
  if (!record.reasons.includes(reason)) {
    record.reasons.push(reason);
  }
  
  ipReputation.suspiciousIPs.set(ip, record);
  
  // Clean up old records (older than 24 hours)
  const oneDayAgo = now - (24 * 60 * 60 * 1000);
  if (record.lastSeen < oneDayAgo && record.count < ipReputation.threshold) {
    ipReputation.suspiciousIPs.delete(ip);
  }
};

// ============================================================================
// 5. COMPREHENSIVE SECURITY MIDDLEWARE
// ============================================================================
const securityHardening = (req, res, next) => {
  const ip = req.ip || req.connection.remoteAddress || 'unknown';
  const fingerprint = requestFingerprint(req);
  
  // Check IP reputation
  const ipCheck = checkIPReputation(ip);
  if (ipCheck.suspicious) {
    logSecurityEvent('BLOCKED_SUSPICIOUS_IP', {
      ip,
      fingerprint,
      score: ipCheck.score,
      reasons: ipCheck.reasons,
      path: req.path
    }, 'error');
    
    return res.status(403).json({
      error: 'Access denied',
      message: 'Your request has been blocked due to suspicious activity.'
    });
  }
  
  // Detect anomalies
  const anomalies = detectAnomalies(req);
  if (anomalies.length > 0) {
    const highSeverity = anomalies.filter(a => a.severity === 'high');
    
    if (highSeverity.length > 0) {
      // High severity - block immediately
      logSecurityEvent('BLOCKED_ANOMALY_HIGH', {
        ip,
        fingerprint,
        path: req.path,
        anomalies: highSeverity,
        url: req.url
      }, 'error');
      
      recordSuspiciousActivity(ip, 'high_severity_anomaly');
      
      return res.status(403).json({
        error: 'Access denied',
        message: 'Your request contains suspicious patterns and has been blocked.'
      });
    } else {
      // Medium/low severity - log and continue
      logSecurityEvent('DETECTED_ANOMALY', {
        ip,
        fingerprint,
        path: req.path,
        anomalies,
        url: req.url
      }, 'warn');
      
      anomalies.forEach(anomaly => {
        if (anomaly.severity === 'medium') {
          recordSuspiciousActivity(ip, anomaly.type);
        }
      });
    }
  }
  
  // Attach security context to request
  req.securityContext = {
    ip,
    fingerprint,
    anomalies: anomalies.length,
    ipReputation: ipCheck.score
  };
  
  next();
};

// ============================================================================
// 6. ENHANCED CSP HEADER (With nonces, no unsafe-inline)
// ============================================================================
const generateCSPHeader = (nonce) => {
  const nonceValue = nonce ? `'nonce-${nonce}'` : '';
  
  // Strict CSP without unsafe-inline or unsafe-eval
  // Only allow scripts with nonces or from trusted sources
  const csp = [
    "default-src 'self'",
    `script-src 'self' ${nonceValue} blob: https://www.googletagmanager.com https://www.google-analytics.com`,
    `script-src-elem 'self' ${nonceValue} blob: https://www.googletagmanager.com https://www.google-analytics.com`,
    `script-src-attr 'self' ${nonceValue}`,
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com", // Styles need unsafe-inline for dynamic styles
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data: blob: https://*.supabase.co https://www.google-analytics.com",
    `connect-src 'self' blob: https://fonts.googleapis.com https://fonts.gstatic.com ${process.env.SUPABASE_URL || ''} https://*.supabase.co https://www.google-analytics.com https://www.googletagmanager.com`,
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "upgrade-insecure-requests"
  ].filter(Boolean).join('; ');
  
  return csp;
};

// ============================================================================
// 7. SECURE HEADERS MIDDLEWARE (Enhanced)
// ============================================================================
const secureHeaders = (req, res, next) => {
  // Generate nonce for this request
  const nonce = generateNonce();
  req.nonce = nonce;
  res.locals.nonce = nonce;
  
  // Basic security headers
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  res.setHeader('X-Permitted-Cross-Domain-Policies', 'none');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  
  // Enhanced CSP with nonce
  const csp = generateCSPHeader(nonce);
  res.setHeader('Content-Security-Policy', csp);
  
  // HSTS in production
  if (process.env.NODE_ENV === 'production') {
    res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
    res.setHeader('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');
  }
  
  // Hide server information
  res.removeHeader('X-Powered-By');
  
  next();
};

// ============================================================================
// 8. RATE LIMITING ENHANCEMENT (Per IP + Fingerprint)
// ============================================================================
const enhancedRateLimit = (windowMs, maxRequests) => {
  const requests = new Map(); // fingerprint -> { count, resetTime }
  
  return (req, res, next) => {
    const fingerprint = requestFingerprint(req);
    const now = Date.now();
    const record = requests.get(fingerprint);
    
    if (!record || now > record.resetTime) {
      // New window
      requests.set(fingerprint, {
        count: 1,
        resetTime: now + windowMs
      });
      return next();
    }
    
    if (record.count >= maxRequests) {
      logSecurityEvent('RATE_LIMIT_EXCEEDED', {
        ip: req.ip,
        fingerprint,
        path: req.path
      }, 'warn');
      
      recordSuspiciousActivity(req.ip, 'rate_limit_exceeded');
      
      return res.status(429).json({
        error: 'Too many requests',
        message: `Rate limit exceeded. Please try again in ${Math.ceil((record.resetTime - now) / 1000)} seconds.`,
        retryAfter: Math.ceil((record.resetTime - now) / 1000)
      });
    }
    
    record.count++;
    next();
  };
};

module.exports = {
  generateNonce,
  attachNonce,
  requestFingerprint,
  detectAnomalies,
  checkIPReputation,
  recordSuspiciousActivity,
  securityHardening,
  generateCSPHeader,
  secureHeaders,
  enhancedRateLimit
};

