// secure-html-renderer.js
// Safe HTML rendering utilities - replaces innerHTML usage
const DOMPurify = require('dompurify');
const { JSDOM } = require('jsdom');

// Create a DOM environment for DOMPurify (server-side)
const window = new JSDOM('').window;
const purify = DOMPurify(window);

// ============================================================================
// SECURE HTML RENDERING
// ============================================================================

/**
 * Safely render HTML content (replaces innerHTML)
 * @param {string} html - HTML string to render
 * @param {object} options - Sanitization options
 * @returns {string} - Sanitized HTML
 */
const safeRenderHTML = (html, options = {}) => {
  if (!html || typeof html !== 'string') return '';
  
  const defaultOptions = {
    ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'a', 'span', 'div'],
    ALLOWED_ATTR: ['href', 'class', 'id', 'target', 'rel'],
    ALLOW_DATA_ATTR: false,
    FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form', 'input', 'button'],
    FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover', 'onfocus', 'onblur']
  };
  
  const config = { ...defaultOptions, ...options };
  
  // Sanitize HTML
  const sanitized = purify.sanitize(html, config);
  
  return sanitized;
};

/**
 * Safely set text content (replaces innerHTML for plain text)
 * @param {string} text - Plain text to set
 * @returns {string} - Escaped text safe for textContent
 */
const safeSetText = (text) => {
  if (!text || typeof text !== 'string') return '';
  
  // Escape HTML entities
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
};

/**
 * Safely create HTML element with content
 * @param {string} tag - HTML tag name
 * @param {object} attributes - Element attributes
 * @param {string} content - Element content (will be sanitized)
 * @returns {string} - Safe HTML string
 */
const safeCreateElement = (tag, attributes = {}, content = '') => {
  const safeTag = safeSetText(tag).toLowerCase();
  if (!safeTag.match(/^[a-z][a-z0-9]*$/)) {
    throw new Error('Invalid HTML tag');
  }
  
  let attrs = '';
  Object.entries(attributes).forEach(([key, value]) => {
    const safeKey = safeSetText(key);
    const safeValue = safeSetText(String(value));
    attrs += ` ${safeKey}="${safeValue}"`;
  });
  
  const safeContent = safeRenderHTML(content);
  
  return `<${safeTag}${attrs}>${safeContent}</${safeTag}>`;
};

/**
 * Safely update element content (client-side helper)
 * This function should be used in frontend JavaScript
 */
const clientSafeUpdate = {
  /**
   * Safely set text content (use instead of innerHTML for user content)
   */
  setText: (element, text) => {
    if (!element) return;
    if (typeof text !== 'string') text = String(text || '');
    element.textContent = text;
  },
  
  /**
   * Safely set HTML content (use for trusted HTML only)
   */
  setHTML: (element, html, options = {}) => {
    if (!element) return;
    if (typeof html !== 'string') html = String(html || '');
    
    // Use DOMPurify if available (client-side)
    if (typeof window !== 'undefined' && window.DOMPurify) {
      const sanitized = window.DOMPurify.sanitize(html, {
        ALLOWED_TAGS: options.allowedTags || ['p', 'br', 'strong', 'em', 'u', 'a', 'span', 'div'],
        ALLOWED_ATTR: options.allowedAttrs || ['href', 'class', 'id', 'target', 'rel'],
        FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form'],
        FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover']
      });
      element.innerHTML = sanitized;
    } else {
      // Fallback: escape and use textContent
      element.textContent = html;
    }
  },
  
  /**
   * Safely create and append element
   */
  createAndAppend: (parent, tag, attributes = {}, content = '') => {
    if (!parent) return null;
    
    const element = document.createElement(tag);
    
    // Set attributes safely
    Object.entries(attributes).forEach(([key, value]) => {
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
    
    // Set content safely
    if (content) {
      clientSafeUpdate.setText(element, content);
    }
    
    parent.appendChild(element);
    return element;
  }
};

module.exports = {
  safeRenderHTML,
  safeSetText,
  safeCreateElement,
  clientSafeUpdate
};

