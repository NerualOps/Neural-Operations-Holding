// auth-utils.js
const jwt = require('jsonwebtoken');
const { logSecurityEvent } = require('./logging');

// Set authentication cookie
const setAuthCookie = (res, userId, email, role) => {
  if (!process.env.JWT_SECRET) {
    console.error('[SECURITY] JWT_SECRET not configured');
    return false;
  }
  
  try {
    const token = jwt.sign(
      { 
        userId, 
        email, 
        role,
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + (7 * 24 * 60 * 60) // 7 days
      }, 
      process.env.JWT_SECRET
    );
    
    // Set secure HTTP-only cookie
    res.setHeader('Set-Cookie', `authToken=${token}; Path=/; HttpOnly; SameSite=Strict${process.env.NODE_ENV === 'production' ? '; Secure' : ''}; Max-Age=${7 * 24 * 60 * 60}`);
    
    logSecurityEvent('AUTH_COOKIE_SET', {
      userId,
      email,
      role
    });
    
    return true;
  } catch (error) {
    console.error('Error setting auth cookie:', error);
    
    logSecurityEvent('AUTH_COOKIE_ERROR', {
      userId,
      email,
      error: error.message
    }, 'error');
    
    return false;
  }
};

// Clear authentication cookie
const clearAuthCookie = (res) => {
  res.setHeader('Set-Cookie', `authToken=; Path=/; HttpOnly; SameSite=Strict${process.env.NODE_ENV === 'production' ? '; Secure' : ''}; Max-Age=0`);
  
  logSecurityEvent('AUTH_COOKIE_CLEARED', {});
  
  return true;
};

// Verify JWT token
const verifyToken = (token) => {
  if (!process.env.JWT_SECRET) {
    console.error('[SECURITY] JWT_SECRET not configured');
    return null;
  }
  
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    return decoded;
  } catch (error) {
    console.error('JWT verification error:', error.message);
    return null;
  }
};

module.exports = { setAuthCookie, clearAuthCookie, verifyToken };
