// client-security.js
// Client-side security utilities for safe HTML rendering
// Include this in HTML files: <script src="/runtime/client-security.js"></script>

(function() {
  'use strict';
  
  // Check if DOMPurify is available (should be loaded separately)
  const hasDOMPurify = typeof window !== 'undefined' && window.DOMPurify;
  
  /**
   * Safely set text content (use instead of innerHTML for user content)
   */
  window.safeSetText = function(element, text) {
    if (!element) return;
    if (typeof text !== 'string') text = String(text || '');
    element.textContent = text;
  };
  
  /**
   * Safely set HTML content (sanitizes HTML before setting)
   */
  window.safeSetHTML = function(element, html, options) {
    if (!element) return;
    if (typeof html !== 'string') html = String(html || '');
    
    if (hasDOMPurify) {
      // Use DOMPurify for sanitization
      const config = {
        ALLOWED_TAGS: (options && options.allowedTags) || ['p', 'br', 'strong', 'em', 'u', 'a', 'span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li'],
        ALLOWED_ATTR: (options && options.allowedAttrs) || ['href', 'class', 'id', 'target', 'rel'],
        FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form', 'input', 'button'],
        FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover', 'onfocus', 'onblur', 'onchange']
      };
      
      const sanitized = window.DOMPurify.sanitize(html, config);
      element.innerHTML = sanitized;
    } else {
      // Fallback: escape and use textContent
      element.textContent = html;
    }
  };
  
  /**
   * Safely create and append element
   */
  window.safeCreateElement = function(parent, tag, attributes, content) {
    if (!parent || !tag) return null;
    
    const element = document.createElement(tag);
    
    // Set attributes safely
    if (attributes) {
      Object.entries(attributes).forEach(function([key, value]) {
        if (key === 'class') {
          element.className = String(value || '');
        } else if (key === 'id') {
          element.id = String(value || '');
        } else if (key.startsWith('data-')) {
          element.setAttribute(key, String(value || ''));
        } else {
          element.setAttribute(key, String(value || ''));
        }
      });
    }
    
    // Set content safely
    if (content) {
      window.safeSetText(element, content);
    }
    
    parent.appendChild(element);
    return element;
  };
  
  /**
   * Escape HTML entities (for use in attributes or text)
   */
  window.escapeHtml = function(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
  };
  
  // Override console methods in production to prevent information leakage
  if (typeof window !== 'undefined' && window.location && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    // In production, sanitize console output
    const originalLog = console.log;
    const originalError = console.error;
    
    console.log = function() {
      // Only log in development
      if (window.location.search.includes('debug=true')) {
        originalLog.apply(console, arguments);
      }
    };
    
    console.error = function() {
      // Always log errors, but sanitize sensitive data
      const sanitized = Array.from(arguments).map(function(arg) {
        if (typeof arg === 'string') {
          // Remove potential secrets
          return arg.replace(/(password|secret|key|token)=[^&\s]+/gi, '$1=***');
        }
        return arg;
      });
      originalError.apply(console, sanitized);
    };
  }
  
  console.log('[SECURITY] Client-side security utilities loaded');
})();

