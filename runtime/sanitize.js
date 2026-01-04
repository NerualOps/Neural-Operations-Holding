// sanitize.js - Enhanced security version
const path = require('path');

// Sanitize HTML content - ENHANCED
const sanitizeHTML = (html) => {
  if (!html) return '';
  if (typeof html !== 'string') return '';
  
  let sanitized = html
    // Remove script tags and their contents
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    // Remove event handlers
    .replace(/on\w+\s*=\s*["'][^"']*["']/gi, '')
    // Remove dangerous tags
    .replace(/<iframe/gi, '&lt;iframe')
    .replace(/<object/gi, '&lt;object')
    .replace(/<embed/gi, '&lt;embed')
    .replace(/<link/gi, '&lt;link')
    .replace(/<meta/gi, '&lt;meta')
    // Escape remaining HTML
    .replace(/</g, '<')
    .replace(/>/g, '>')
    .replace(/"/g, '"')
    .replace(/'/g, '&#39;');
  
  // Limit length to prevent DoS
  if (sanitized.length > 100000) {
    sanitized = sanitized.substring(0, 100000);
  }
  
  return sanitized;
};

// Sanitize text content - ENHANCED
const sanitizeText = (text) => {
  if (!text) return '';
  if (typeof text !== 'string') return '';
  
  let sanitized = text
    // Remove HTML tags
    .replace(/<[^>]*>/g, '')
    // Remove script tags
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    // Remove event handlers
    .replace(/on\w+\s*=\s*["'][^"']*["']/gi, '')
    // Remove control characters except newlines and tabs
    .replace(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]/g, '')
    // Escape special characters
    .replace(/</g, '<')
    .replace(/>/g, '>')
    .replace(/"/g, '"')
    .replace(/'/g, '&#39;')
    // Remove null bytes
    .replace(/\0/g, '');
  
  // Limit length to prevent DoS
  if (sanitized.length > 50000) {
    sanitized = sanitized.substring(0, 50000);
  }
  
  return sanitized.trim();
};

// Sanitize SQL input to prevent injection
// NOTE: This is a basic sanitizer. For production, always use parameterized queries.
// This function only removes obvious SQL injection patterns from user-provided strings
// that might be used in dynamic queries (which should be avoided).
const sanitizeSQL = (input) => {
  if (!input) return '';
  if (typeof input !== 'string') return '';
  
  // Remove SQL comment patterns
  let sanitized = input
    .replace(/--/g, '')  // SQL comments
    .replace(/\/\*/g, '')  // Multi-line comment start
    .replace(/\*\//g, '')  // Multi-line comment end
    .replace(/;/g, '');  // Statement terminators
  
  // Remove dangerous SQL keywords when they appear in injection patterns
  // (only remove if they appear as part of injection attempts, not legitimate text)
  sanitized = sanitized
    .replace(/union\s+select/gi, '')
    .replace(/;\s*drop\s+table/gi, '')
    .replace(/;\s*drop\s+database/gi, '')
    .replace(/;\s*delete\s+from/gi, '')
    .replace(/;\s*truncate\s+table/gi, '')
    .replace(/exec\s*\(/gi, '')
    .replace(/execute\s*\(/gi, '')
    .replace(/xp_cmdshell/gi, '')
    .replace(/sp_executesql/gi, '');
  
  // Remove null bytes
  sanitized = sanitized.replace(/\0/g, '');
  
  // Limit length to prevent DoS
  if (sanitized.length > 10000) {
    sanitized = sanitized.substring(0, 10000);
  }
  
  return sanitized.trim();
};

// Sanitize filename to prevent path traversal - ENHANCED
const sanitizeFilename = (filename) => {
  if (!filename) return '';
  if (typeof filename !== 'string') return '';
  
  const sanitized = path.basename(filename)
    .replace(/[\/\\]/g, '')
    .replace(/\.\./g, '')
    .replace(/\s+/g, '_')
    .replace(/[^a-zA-Z0-9_\-\.]/g, '')
    // Remove null bytes
    .replace(/\0/g, '')
    // Limit length
    .substring(0, 255);
  
  return sanitized;
};

// Validate and sanitize user input - NEW
const validateUserInput = (input, options = {}) => {
  if (!input) return '';
  if (typeof input !== 'string') return '';
  
  const {
    maxLength = 5000,
    allowHTML = false,
    allowSpecialChars = true
  } = options;
  
  let sanitized = input;
  
  if (!allowHTML) {
    sanitized = sanitizeText(sanitized);
  } else {
    sanitized = sanitizeHTML(sanitized);
  }
  
  if (!allowSpecialChars) {
    sanitized = sanitized.replace(/[^\w\s.,?!@:;()\-"']/g, '');
  }
  
  if (sanitized.length > maxLength) {
    sanitized = sanitized.substring(0, maxLength);
  }
  
  return sanitized.trim();
};

module.exports = { 
  sanitizeHTML, 
  sanitizeText, 
  sanitizeSQL,
  sanitizeFilename,
  validateUserInput
};
