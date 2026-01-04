// csrf-protection.js
const crypto = require('crypto');
const { parseCookies } = require('./auth-middleware');
const { logSecurityEvent } = require('./logging');

// Generate a CSRF token
const generateCSRFToken = () => {
  return crypto.randomBytes(16).toString('hex');
};

// CSRF protection middleware
const csrfProtection = (req, res, next) => {
  // Skip CSRF check for GET, HEAD, OPTIONS requests
  if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
    // For GET requests, set a CSRF token cookie if it doesn't exist
    if (req.method === 'GET') {
      const cookies = parseCookies(req.headers.cookie || '');
      if (!cookies.csrfToken) {
        const token = generateCSRFToken();
        const cookieFlags = [`Path=/`, `SameSite=Strict`];
        if (process.env.NODE_ENV === 'production') {
          cookieFlags.push('Secure');
        }
        res.setHeader('Set-Cookie', `csrfToken=${token}; ${cookieFlags.join('; ')}`);
        res.setHeader('X-CSRF-Token', token);
      }
    }
    
    next();
    return;
  }
  
  // For all other methods, validate the CSRF token
  const cookies = parseCookies(req.headers.cookie || '');
  const cookieToken = cookies.csrfToken;
  
  // For multipart/form-data (file uploads), token may be in form body (parsed by multer)
  // For JSON requests, token should be in header
  const headerToken = req.headers['x-csrf-token'] || req.headers['X-CSRF-Token'];
  const formToken = req.body?._csrf || req.body?.csrfToken;
  
  // Accept token from header OR form data (for file uploads)
  const providedToken = headerToken || formToken;
  
  // In development, be more lenient - allow requests with header token OR cookie token
  // This helps with token synchronization issues during development
  const isDevelopment = process.env.NODE_ENV !== 'production';
  
  // In development: allow if header token OR cookie token exists (don't require both to match)
  // In production: require both cookie and header token to exist and match
  if (isDevelopment) {
    // Development: Allow if we have ANY token (header, form, or cookie)
    if (headerToken || formToken || cookieToken) {
      req.csrfValid = true;
      next();
      return;
    }
    // No tokens at all - log but still allow in development
    logSecurityEvent('CSRF_VALIDATION_FAILED', {
      path: req.path,
      ip: req.ip,
      method: req.method,
      hasCookieToken: !!cookieToken,
      hasHeaderToken: !!headerToken,
      hasFormToken: !!formToken,
      isDevelopment: true
    }, 'warn');
    req.csrfValid = false;
    next();
    return;
  }
  
  // Production: Require cookie token AND provided token to match
  if (!cookieToken || !providedToken || cookieToken !== providedToken) {
    // Log the CSRF validation failure
    logSecurityEvent('CSRF_VALIDATION_FAILED', {
      path: req.path,
      ip: req.ip,
      method: req.method,
      hasCookieToken: !!cookieToken,
      hasHeaderToken: !!headerToken,
      hasFormToken: !!formToken,
      isDevelopment: false
    }, 'warn');
    
    req.csrfValid = false;
    // In production, still allow the request but set flag (routes can decide)
    next();
    return;
  }
  
  // CSRF validation passed
  req.csrfValid = true;
  next();
};

module.exports = { csrfProtection, generateCSRFToken };
