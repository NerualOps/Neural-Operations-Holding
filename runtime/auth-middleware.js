const jwt = require('jsonwebtoken');

// Parse cookies helper (defined first to avoid hoisting issues)
const parseCookies = (cookieHeader) => {
  const cookies = {};
  if (cookieHeader) {
    cookieHeader.split(';').forEach(cookie => {
      const [name, value] = cookie.trim().split('=');
      if (name && value) {
        cookies[name] = decodeURIComponent(value);
      }
    });
  }
  return cookies;
};

// Verify authentication middleware - UPDATED WITH REDIRECTS
// NOTE: This is a simpler version. runtime/server.js has an enhanced verifyAuth with security logging.
// This version is kept for backward compatibility but is not currently used.
const verifyAuth = (requiredRole) => {
  return (req, res, next) => {
    try {
      // Get token from cookie
      const cookies = parseCookies(req.headers.cookie || '');
      const token = cookies.authToken;
      
      if (!token) {
        // CHANGED: Redirect instead of JSON response
        // Silent - no console.log
        return res.redirect('/login?unauthorized=true');
      }
      
      // Verify token
      try {
        // Security: Check JWT_SECRET is configured
        if (!process.env.JWT_SECRET) {
          console.error('JWT_SECRET not configured');
          return res.redirect('/login?error=config');
        }
        // Silent - no console.log
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        // Silent - no console.log
        
        // Check role if required
        if (requiredRole && decoded.role !== requiredRole && 
            !(requiredRole === 'client' && decoded.role === 'owner')) {
          // CHANGED: Redirect instead of JSON response
          // Silent - no console.log
          return res.redirect('/login?permission=false');
        }
        
        // Add user info to request
        req.user = {
          id: decoded.userId,
          email: decoded.email,
          role: decoded.role
        };
        
        // Silent - no console.log
        next();
      } catch (jwtError) {
        // CHANGED: Redirect for JWT verification errors
        console.error('JWT verification error:', jwtError.message);
        return res.redirect('/login?invalid=true');
      }
    } catch (error) {
      console.error('Auth middleware error:', error.message);
      // CHANGED: Redirect for any other errors
      return res.redirect('/login?error=true');
    }
  };
};

module.exports = { verifyAuth, parseCookies };
