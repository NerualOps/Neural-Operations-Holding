// logging.js
const fs = require('fs');
const path = require('path');

// Create logs directory if it doesn't exist
const logsDir = path.join(__dirname, 'logs');
if (!fs.existsSync(logsDir)) {
  try {
    fs.mkdirSync(logsDir);
  } catch (err) {
    console.error('Failed to create logs directory:', err);
  }
}

// Logger for general application logs
const logger = {
  info: (message) => {
    const timestamp = new Date().toISOString();
    const logMessage = typeof message === 'object' 
      ? `[${timestamp}] INFO: ${JSON.stringify(message)}\n`
      : `[${timestamp}] INFO: ${message}\n`;
    
    // Silent - no console.log (only write to file)
    
    try {
      fs.appendFileSync(path.join(logsDir, 'app.log'), logMessage);
    } catch (err) {
      console.error('Failed to write to log file:', err);
    }
  },
  
  error: (message, error) => {
    const timestamp = new Date().toISOString();
    const errorStack = error && error.stack ? error.stack : '';
    const logMessage = typeof message === 'object'
      ? `[${timestamp}] ERROR: ${JSON.stringify(message)}\n${errorStack}\n`
      : `[${timestamp}] ERROR: ${message}\n${errorStack}\n`;
    
    console.error(message, error || '');
    
    try {
      fs.appendFileSync(path.join(logsDir, 'error.log'), logMessage);
    } catch (err) {
      console.error('Failed to write to error log file:', err);
    }
  },
  
  warn: (message) => {
    const timestamp = new Date().toISOString();
    const logMessage = typeof message === 'object'
      ? `[${timestamp}] WARN: ${JSON.stringify(message)}\n`
      : `[${timestamp}] WARN: ${message}\n`;
    
    console.warn(message);
    
    try {
      fs.appendFileSync(path.join(logsDir, 'app.log'), logMessage);
    } catch (err) {
      console.error('Failed to write to log file:', err);
    }
  }
};

// Security event logger
const logSecurityEvent = (eventType, data, level = 'info') => {
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] SECURITY ${eventType}: ${JSON.stringify(data)}\n`;
  
  // Log to console based on level (only errors and warnings)
  if (level === 'error') {
    console.error(`SECURITY ${eventType}:`, data);
  } else if (level === 'warn') {
    console.warn(`SECURITY ${eventType}:`, data);
  } else {
    // Silent - no console.log for info level security events
  }
  
  // Write to security log file
  try {
    fs.appendFileSync(path.join(logsDir, 'security.log'), logMessage);
  } catch (err) {
    console.error('Failed to write to security log file:', err);
  }
  
  // For critical security events, also log to a separate file
  if (level === 'error' || eventType.includes('ATTACK') || eventType.includes('BREACH')) {
    try {
      fs.appendFileSync(path.join(logsDir, 'critical-security.log'), logMessage);
    } catch (err) {
      console.error('Failed to write to critical security log file:', err);
    }
  }
};

module.exports = { logger, logSecurityEvent };
