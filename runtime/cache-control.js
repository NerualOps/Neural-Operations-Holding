// cache-control.js
const crypto = require('crypto');
const path = require('path');
const fs = require('fs'); // Added missing import for fs

// Enhanced cache control middleware
const setupCacheControl = (app, BUILD_ID) => {
  // Cache control middleware
  app.use((req, res, next) => {
    const p = req.path || '';
    
    // HTML files - short cache to prevent infinite loops
    if (p.endsWith('.html') || (req.headers.accept && req.headers.accept.includes('text/html'))) {
      res.setHeader('Cache-Control', 'public, max-age=300, must-revalidate'); // 5 minutes
      res.setHeader('Pragma', 'no-cache');
      res.setHeader('Expires', new Date(Date.now() + 300000).toUTCString());
      return next();
    }
    
    // JS, CSS, JSON files - versioned caching with longer duration
    if (/\.(js|css|mjs|json)$/i.test(p)) {
      // Use file hash instead of BUILD_ID to prevent unnecessary refreshes
      // SECURITY: Prevent path traversal attacks by normalizing and checking path
      const normalizedPath = path.normalize(p.startsWith('/') ? p.slice(1) : p);
      if (normalizedPath.includes('..') || normalizedPath.startsWith('/')) {
        // Path traversal attempt detected - skip cache control for this request
        return next();
      }
      const filePath = path.join(__dirname, normalizedPath);
      
      try {
        if (fs.existsSync(filePath)) {
          const stats = fs.statSync(filePath);
          const fileHash = crypto.createHash('md5').update(stats.mtime.toISOString() + stats.size).digest('hex');
          
          res.setHeader('ETag', `"${fileHash}"`);
          res.setHeader('Cache-Control', 'public, max-age=86400, must-revalidate'); // 24 hours
          res.setHeader('X-File-Hash', fileHash);
          
          const ifNoneMatch = req.headers['if-none-match'];
          if (ifNoneMatch === `"${fileHash}"`) {
            return res.status(304).end();
          }
        }
      } catch (error) {
        // If file doesn't exist, use BUILD_ID as fallback
        res.setHeader('ETag', `"${BUILD_ID}"`);
        res.setHeader('Cache-Control', 'public, max-age=86400, must-revalidate');
      }
      
      return next();
    }
    
    // Static assets - longer caching
    if (/\.(png|jpg|jpeg|gif|svg|webp|ico|woff2?|ttf|otf)$/i.test(p)) {
      res.setHeader('Cache-Control', 'public, max-age=604800'); // 7 days
      return next();
    }
    
    next();
  });
};

// Generate ETag for a file
const generateETag = (content) => {
  return crypto.createHash('md5').update(content).digest('hex');
};

// Version-injecting function for HTML files
const injectVersionToHTML = (html, BUILD_ID) => {
  if (!html.includes('meta name="build-id"')) {
    html = html.replace(/<\/head>/i, `<meta name="build-id" content="${BUILD_ID}">\n</head>`);
  }
  
  html = html.replace(/(<meta name="build-id" content=")([^"]*)(")/i, `$1${BUILD_ID}$3`);
  
  // Replace CDN Supabase script with local version
  html = html.replace(
    /<script src="https:\/\/cdn\.jsdelivr\.net\/npm\/@supabase\/supabase-js@2"><\/script>/,
    `<script src="/services/libs/supabase.js"></script>`
  );
    
  return html;
};

module.exports = { setupCacheControl, generateETag, injectVersionToHTML };
