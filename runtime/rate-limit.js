// rate-limit.js
const rateLimit = require('express-rate-limit');
const { logSecurityEvent } = require('./logging');

const isDevelopment = process.env.NODE_ENV !== 'production';

// Helper to check if request is from localhost
const isLocalhost = (req) => {
  // Use clientIP if available (set by middleware), otherwise check all sources
  const ip = req.clientIP || 
             req.ip || 
             req.connection?.remoteAddress || 
             req.socket?.remoteAddress ||
             (req.headers['x-forwarded-for']?.split(',')[0]?.trim()) ||
             req.headers['x-real-ip'] ||
             req.headers['cf-connecting-ip'] ||
             'unknown';
  
  // Check all possible localhost variations
  const localhostIPs = [
    '127.0.0.1',
    '::1',
    '::ffff:127.0.0.1',
    'localhost'
  ];
  
  // Check exact match or if IP starts with localhost pattern
  // Only skip if we can confirm it's localhost (don't skip for unknown IPs)
  return localhostIPs.some(localIP => 
    ip === localIP || 
    ip?.includes(localIP) ||
    ip?.startsWith('127.') ||
    ip?.startsWith('::1')
  );
};

// Auth rate limiter - stricter limits for authentication endpoints
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: isDevelopment ? 1000 : 10, // Higher limit in dev
  standardHeaders: true,
  legacyHeaders: false,
  skip: (req) => isDevelopment && isLocalhost(req),
  handler: (req, res, next, options) => {
    logSecurityEvent('RATE_LIMIT_EXCEEDED_AUTH', {
      path: req.path,
      ip: req.ip
    }, 'warn');
    res.status(options.statusCode).send('Too many authentication attempts, please try again later.');
  }
});

// API rate limiter - general API endpoints
// Disabled in development, high limit in production
const apiLimiter = rateLimit({
  windowMs: 5 * 60 * 1000, // 5 minutes
  max: isDevelopment ? 100000 : 100, // Unlimited in dev (100000), 100 in production
  standardHeaders: true,
  legacyHeaders: false,
  skip: (req) => isDevelopment && isLocalhost(req),
  handler: (req, res, next, options) => {
    logSecurityEvent('RATE_LIMIT_EXCEEDED_API', {
      path: req.path,
      ip: req.ip
    }, 'warn');
    res.status(options.statusCode).send('Too many requests, please try again later.');
  }
});

// Upload rate limiter - for document uploads
// Disabled in development, very high limit in production
const uploadLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  max: isDevelopment ? 10000 : 100, // Unlimited in dev (10000), 100/hour in production
  standardHeaders: true,
  legacyHeaders: false,
  skip: (req) => isDevelopment && isLocalhost(req),
  handler: (req, res, next, options) => {
    logSecurityEvent('RATE_LIMIT_EXCEEDED_UPLOAD', {
      path: req.path,
      ip: req.ip
    }, 'warn');
    res.status(options.statusCode).json({ 
      error: 'Too many document uploads, please try again later.'
    });
  }
});

// Sensitive route limiter - for sensitive endpoints like obfuscated files
const sensitiveRouteLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: isDevelopment ? 10000 : 50, // Higher limit in dev
  standardHeaders: true,
  legacyHeaders: false,
  skip: (req) => isDevelopment && isLocalhost(req),
  handler: (req, res, next, options) => {
    logSecurityEvent('RATE_LIMIT_EXCEEDED', {
      path: req.path,
      ip: req.ip
    }, 'warn');
    res.status(options.statusCode).send('Too many requests from this IP, please try again later');
  }
});

module.exports = { authLimiter, apiLimiter, uploadLimiter, sensitiveRouteLimiter };
