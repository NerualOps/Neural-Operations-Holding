// protected-files.js
// Blocks ALL source code access in production
const { logSecurityEvent } = require('./logging');
const isProduction = process.env.NODE_ENV === 'production';

// Protected files middleware - AGGRESSIVE in production
const protectSourceFiles = (req, res, next) => {
  if (!isProduction) {
    return next();
  }
  
  // In production, block ALL source code files
  const protectedPatterns = [
    /\.js$/i,                // All JavaScript files (must use obfuscated routes)
    /\.ts$/i,                // TypeScript files
    /\.tsx$/i,               // TypeScript React files
    /\.jsx$/i,               // JavaScript React files
    /\.json$/i,              // All JSON files (except allowed)
    /\.env/i,                // Environment files
    /\.config/i,             // Config files
    /\.md$/i,                // Markdown files
    /package(-lock)?\.json/i, // Package files
    /\.map$/i,               // Source maps
    /\.log$/i,               // Log files
    /\.git/i,                // Git files
    /\.eslintrc/i,           // ESLint config
    /\.prettierrc/i,         // Prettier config
    /tsconfig\.json$/i,      // TypeScript config
    /webpack\.config/i,      // Webpack config
    /\.test\.js$/i,          // Test files
    /\.spec\.js$/i           // Spec files
  ];
  
  // Only allow obfuscated routes and specific public files
  // NOTE: JavaScript files should NOT be in allowedPaths - they must go through obfuscated routes
  const allowedPaths = [
    '/obfuscated/',
    '/node_modules/',
    '/cdn/',
    '/assets/',
    '/Imagies/',
    '/images/',
    '/img/',
    '/fonts/',
    '/favicon.ico',
    '/robots.txt',
    '/api/'
    // REMOVED: '/services/libs/supabase.js' - Must use obfuscated route
  ];
  
  const reqPath = req.path;
  const pathWithoutQuery = reqPath.split('?')[0];
  
  // Allow obfuscated and public paths
  if (allowedPaths.some(allowed => pathWithoutQuery.startsWith(allowed) || pathWithoutQuery === allowed.replace('/', ''))) {
    return next();
  }
  
  // Block ALL JavaScript files in production (must use obfuscated routes)
  if (pathWithoutQuery.endsWith('.js')) {
    logSecurityEvent('BLOCKED_JS_FILE_ACCESS', {
      path: req.path,
      ip: req.ip,
      userAgent: req.headers['user-agent']
    }, 'warn');
    
    return res.status(403).send('Access denied - Use obfuscated routes');
  }
  
  // Block all other protected patterns
  const isProtected = protectedPatterns.some(pattern => pattern.test(pathWithoutQuery));
  
  if (isProtected) {
    logSecurityEvent('BLOCKED_SOURCE_FILE_ACCESS', {
      path: req.path,
      ip: req.ip,
      userAgent: req.headers['user-agent']
    }, 'warn');
    
    return res.status(403).send('Access denied - Source files are protected');
  }
  
  next();
};

module.exports = { protectSourceFiles };
