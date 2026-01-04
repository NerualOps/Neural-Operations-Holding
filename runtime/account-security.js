// account-security.js
const { logSecurityEvent } = require('./logging');

// In-memory store for login attempts (would use Redis in production)
const loginAttempts = {};

// Check login attempts for an account
const checkLoginAttempts = (identifier) => {
  if (!identifier) return { allowed: true, remainingAttempts: 5 };
  
  const now = Date.now();
  const attempts = loginAttempts[identifier];
  
  if (!attempts) {
    return { allowed: true, remainingAttempts: 5 };
  }
  
  // Clean up old attempts (older than 15 minutes)
  const recentAttempts = attempts.filter(timestamp => now - timestamp < 15 * 60 * 1000);
  loginAttempts[identifier] = recentAttempts;
  
  // If 5 or more recent attempts, account is locked
  if (recentAttempts.length >= 5) {
    const oldestAttempt = Math.min(...recentAttempts);
    const lockReleaseTime = oldestAttempt + (15 * 60 * 1000);
    const remainingTime = Math.max(0, lockReleaseTime - now);
    
    logSecurityEvent('ACCOUNT_LOCKED', {
      identifier,
      attempts: recentAttempts.length,
      remainingTime
    }, 'warn');
    
    return { 
      locked: true,
      allowed: false, 
      remainingAttempts: 0,
      remainingTime: remainingTime
    };
  }
  
  return { 
    locked: false,
    allowed: true, 
    remainingAttempts: 5 - recentAttempts.length
  };
};

// Record a login attempt (success or failure)
const recordLoginAttempt = (identifier, success = false, ip = null) => {
  if (!identifier) return { attempts: 0, locked: false };
  
  const now = Date.now();
  
  if (success) {
    // Reset attempts on successful login
    if (loginAttempts[identifier]) {
      delete loginAttempts[identifier];
    }
    
    logSecurityEvent('LOGIN_SUCCESS', {
      identifier,
      ip
    });
    
    return { attempts: 0, locked: false };
  }
  
  // Record failed attempt
  if (!loginAttempts[identifier]) {
    loginAttempts[identifier] = [];
  }
  
  loginAttempts[identifier].push(now);
  
  // Clean up old attempts
  const recentAttempts = loginAttempts[identifier].filter(timestamp => now - timestamp < 15 * 60 * 1000);
  loginAttempts[identifier] = recentAttempts;
  
  const attempts = recentAttempts.length;
  const locked = attempts >= 5;
  
  logSecurityEvent('FAILED_LOGIN_ATTEMPT', {
    identifier,
    attempts,
    ip
  }, attempts >= 3 ? 'warn' : 'info');
  
  // If this attempt caused a lockout, log it
  if (locked) {
    const oldestAttempt = Math.min(...recentAttempts);
    const lockReleaseTime = oldestAttempt + (15 * 60 * 1000);
    const remainingTime = Math.max(0, lockReleaseTime - now);
    
    logSecurityEvent('ACCOUNT_LOCKOUT', {
      identifier,
      lockDuration: '15 minutes',
      remainingTime
    }, 'warn');
  }
  
  return { attempts, locked, remainingTime: locked ? Math.max(0, (Math.min(...recentAttempts) + (15 * 60 * 1000)) - now) : 0 };
};

// Reset login attempts for an account after successful login
const resetLoginAttempts = (identifier) => {
  if (!identifier) return;
  
  if (loginAttempts[identifier]) {
    delete loginAttempts[identifier];
    
    logSecurityEvent('LOGIN_ATTEMPTS_RESET', {
      identifier
    });
  }
};

module.exports = { 
  checkLoginAttempts, 
  recordLoginAttempt,
  resetLoginAttempts
};
