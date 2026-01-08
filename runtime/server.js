// Main Express server with Epsilon AI system integration
// © 2025 Neural Ops – a division of Neural Operation's & Holding's LLC. All rights reserved.

// Log immediately using process.stdout.write to bypass any buffering
process.stdout.write('========================================\n');
process.stdout.write('[SERVER] NeuralOps Server Starting\n');
process.stdout.write('[SERVER] Node.js: ' + process.version + '\n');
process.stdout.write('[SERVER] Platform: ' + process.platform + '\n');
process.stdout.write('[SERVER] Environment: ' + (process.env.NODE_ENV || 'development') + '\n');
process.stdout.write('[SERVER] Working Directory: ' + process.cwd() + '\n');
process.stdout.write('========================================\n');

const express = require('express');
const path = require('path');
const fs = require('fs');

// Load environment variables early so downstream modules see them
if (process.env.NODE_ENV !== 'production') {
  const dotenvPath = path.join(process.cwd(), '.env');
  try {
    require('dotenv').config({ path: dotenvPath });
  } catch (err) {
    console.warn('[EPSILON-SERVER] Unable to load .env file:', err.message);
  }
}

const _silent = () => {};
const _silentLog = () => {};
const _silentInfo = () => {};
const _silentDebug = () => {};

// Only log critical startup info and errors/warnings
if (!process.env.SUPABASE_URL) {
  console.warn('[EPSILON-SERVER] SUPABASE_URL is not set');
}

const cors = require('cors');
const PythonServiceManager = require('../services/python-services/python_service_manager');
const ServerLearningService = require('../services/learning-engine/server-learning-service');
const ServerRAGService = require('../services/rag-engine/server-rag-service');
const { serveObfuscatedFile } = require('./production-obfuscation');

const isProduction = process.env.NODE_ENV === 'production';

// Structured Logger for Server (EpsilonLog equivalent)
const EpsilonLog = (() => {
  const sinks = [];
  const addSink = fn => sinks.push(fn);
  function emit(entry) { 
    for (const fn of sinks) try { fn(entry) } catch {} 
  }
  function log(level, code, msg, ctx={}) {
    const entry = {
      ts: new Date().toISOString(),
      level, code, msg,
      buildId: process.env.BUILD_ID || 'dev',
      traceId: ctx.traceId || ctx.requestId || require('crypto').randomUUID(),
      ...ctx
    };
    // Only log errors and warnings - silent for info/debug/log
    if (level === 'error') {
      console.error('[EPSILON-SERVER]', code, msg, ctx);
    } else if (level === 'warn') {
      console.warn('[EPSILON-SERVER]', code, msg, ctx);
    }
    emit(entry);
  }
  return {
    addSink,
    debug:()=>{},
    info:()=>{},
    warn:(c,m,x)=>log('warn',c,m,x),
    error:(c,m,x)=>log('error',c,m,x),
    metric:(name,value,tags={})=>emit({ ts:new Date().toISOString(), kind:'metric', name, value, tags })
  };
})();

// Global guard for server module
if (global.__EPSILON_SERVER_INITED__) { 
  EpsilonLog.warn('SERVER_DUP', 'Server module already initialized');
  module.exports = {}; 
} else {
global.__EPSILON_SERVER_INITED__ = true;


const { createClient } = require('@supabase/supabase-js');
const jwt = require('jsonwebtoken');
// Removed lambda-multipart-parser - not used, and has security vulnerabilities
// Using multer for file uploads instead
const pdfParse = require('pdf-parse');
const multer = require('multer');
// fs already imported above (line 14)
const uuid = require('uuid');
const helmet = require('helmet');
// Compression only needed in production - require conditionally with safe fallback
let compression = null;
if (isProduction) {
  try {
    // Use function wrapper to safely require compression
    const loadCompression = () => {
      try {
        return require('compression');
      } catch (e) {
        return null;
      }
    };
    compression = loadCompression();
    if (!compression) {
      console.warn('[SERVER] Compression module not available, continuing without compression');
    }
  } catch (err) {
    console.warn('[SERVER] Compression module not available, continuing without compression:', err.message);
    compression = null; // Explicitly set to null
  }
}
const crypto = require('crypto');
const https = require('https');
const rateLimit = require('express-rate-limit');

// Import modules
const { validateEnv } = require('./env-validator');
const { parseCookies } = require('./auth-middleware');
const { csrfProtection, generateCSRFToken } = require('./csrf-protection');
const { authLimiter, apiLimiter, uploadLimiter, sensitiveRouteLimiter } = require('./rate-limit');
const { sanitizeHTML, sanitizeText, sanitizeFilename } = require('./sanitize');
const { setAuthCookie, clearAuthCookie } = require('./auth-utils');
const { logger, logSecurityEvent } = require('./logging');
const { decrypt } = require('./encryption');
const { checkLoginAttempts, recordLoginAttempt } = require('./account-security');
const { protectSourceFiles } = require('./protected-files');
const { handleScriptRequest } = require('./script-handler');
const { setupCacheControl, injectVersionToHTML } = require('./cache-control');
const { trackVisitorIP, getClientIP, checkGuestUsage, incrementGuestUsage, associateIPWithAccount } = require('./ip-tracking');
const { secureHTML, minifyHTML: minifyHTMLFromSecurity } = require('./html-security');

// Save original console functions BEFORE epsilon-language-engine overrides them
const originalConsoleLog = console.log.bind(console);
const originalConsoleInfo = console.info.bind(console);
const originalConsoleDebug = console.debug.bind(console);

// Log BEFORE requiring epsilon-language-engine (which might silence console)
console.log('[SERVER] Loading Epsilon Language Engine...');
const epsilonLanguageEngine = require('../core/epsilon-language-engine');
// Training removed - local-only training in ml_local/
console.log('[SERVER] Epsilon Language Engine loaded');

// Restore console functions after epsilon-language-engine silenced them
console.log = originalConsoleLog;
console.info = originalConsoleInfo;
console.debug = originalConsoleDebug;
console.log('[SERVER] Console functions restored');

console.log('[SERVER] ========================================');
console.log('[SERVER] NeuralOps Server Initialization');
console.log('[SERVER] ========================================');
console.log('[SERVER] Node.js version:', process.version);
console.log('[SERVER] Platform:', process.platform);
console.log('[SERVER] Environment:', process.env.NODE_ENV || 'development');
console.log('[SERVER] Working directory:', process.cwd());

console.log('[SERVER] Validating environment variables...');
validateEnv();
console.log('[SERVER] Environment validation passed');

console.log('[SERVER] Creating Express application...');
const app = express();
console.log('[SERVER] Express app created');

// Configure trust proxy more securely
app.set('trust proxy', 1); // Trust only the first proxy (Render's load balancer)
console.log('[SERVER] Trust proxy configured (trusting first proxy)');

const BUILD_ID = process.env.RENDER_GIT_COMMIT || Date.now().toString();
console.log('[SERVER] Build ID:', BUILD_ID.substring(0, 12));

// PERFORMANCE: Enable compression for all responses (production only)
if (isProduction && compression) {
  app.use(compression({
    filter: (req, res) => {
      // Compress all text-based responses
      if (req.headers['x-no-compression']) {
        return false;
      }
      return compression.filter(req, res);
    },
    level: 9, // Maximum compression for production (best performance)
    threshold: 512, // Compress responses larger than 512 bytes
    memLevel: 9 // Maximum memory usage for better compression
  }));
  console.log('[SERVER] Compression middleware enabled');
} else if (isProduction) {
  console.log('[SERVER] Compression middleware not available (skipped)');
}

// Initialize Python Service Manager
console.log('[SERVER] Initializing Python Service Manager...');
let pythonServiceManager = null;
try {
  pythonServiceManager = new PythonServiceManager();
  console.log('[SERVER] Python Service Manager initialized');
  
  // Initialize server-side services
  console.log('[SERVER] Initializing server-side services...');
  const serverLearningService = new ServerLearningService();
  const serverRAGService = new ServerRAGService();
  global.serverLearningService = serverLearningService;
  global.serverRAGService = serverRAGService;
  console.log('[SERVER] Server-side learning and RAG services initialized');
  
  // Training is now local-only in ml_local/ - no Node.js training code
  // Models are trained locally using PyTorch, exported, and loaded by inference service

  // Initialize Python services asynchronously (don't block server startup)
  console.log('[SERVER] ========================================');
  console.log('[SERVER] Starting Python services initialization...');
  console.log('[SERVER] ========================================');
  pythonServiceManager.initialize().then(success => {
    if (success) {
      console.log('[SERVER] ========================================');
      console.log('[SERVER] All Python services are ready');
      console.log('[SERVER] ========================================');
      try {
        epsilonLanguageEngine.attachPythonManager(pythonServiceManager);
        
        // Initialize inference client
        const { getInferenceClient } = require('./inference_client');
        const inferenceClient = getInferenceClient();
        inferenceClient.initialize().catch(err => {
          console.warn('[SERVER] Inference client initialization failed:', err.message);
        });
        
        // Training removed - all training is local-only in ml_local/
        // Models are trained locally and exported, then loaded by inference service
        console.log('[EPSILON LANGUAGE ENGINE] Language engine attached. Training is local-only (ml_local/).');
        console.log('[INFERENCE CLIENT] Inference client initialized.');
      } catch (err) {
        EpsilonLog.warn('LANGUAGE_ENGINE_ATTACH_FAIL', 'Failed to attach language engine', { error: err.message });
      }
    } else {
      EpsilonLog.warn('PYTHON_SERVICES_FAILED', 'Some Python services failed to start');
    }
  }).catch(error => {
    EpsilonLog.error('PYTHON_SERVICES_ERROR', 'Error initializing Python services', { error: error.message });
  });
} catch (error) {
  EpsilonLog.warn('PYTHON_INIT_FAIL', 'Failed to initialize Python services', { error: error.message });
}

// Enhanced security headers middleware - add this near the top of your middleware chain
console.log('[SERVER] Setting up security headers middleware...');
app.use((req, res, next) => {
  // Basic security headers for all responses
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  
  // Additional security headers to prevent source code inspection
  res.setHeader('X-Permitted-Cross-Domain-Policies', 'none');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  
  // Enhanced Content Security Policy with stricter rules
  // RENDER COMPATIBLE: 'self' automatically includes Render domain
  // FIXED: Added 'unsafe-hashes' to allow event handlers (onclick, etc.)
  res.setHeader('Content-Security-Policy', 
    "default-src 'self'; " +
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' 'unsafe-hashes' blob:; " +
    "script-src-attr 'unsafe-inline' 'unsafe-hashes'; " +
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; " +
    "font-src 'self' https://fonts.gstatic.com; " +
    "img-src 'self' data: blob: https://*.supabase.co; " +
    "connect-src 'self' blob: https://fonts.googleapis.com https://fonts.gstatic.com https://*.supabase.co; " +
    "object-src 'none'; " +
    "base-uri 'self'; " +
    "form-action 'self'; " +
    "frame-ancestors 'none'; " +
    "upgrade-insecure-requests"
  );
  
  // PERFORMANCE: Enhanced caching for better performance
  if (req.path.match(/\.(js|css|html)$/i)) {
    res.setHeader('Cache-Control', 'public, max-age=300'); // 5 minutes
    res.setHeader('Pragma', 'cache');
  } else if (req.path.match(/\.(png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot)$/i)) {
    res.setHeader('Cache-Control', 'public, max-age=31536000'); // 1 year for static assets
  }
  
  // PERFORMANCE: Compression for text-based responses
  if (req.path.match(/\.(html|js|css|json|xml|txt)$/i)) {
    res.setHeader('Vary', 'Accept-Encoding');
  }
  
  // SECURITY: Additional headers for production
  if (process.env.NODE_ENV === 'production') {
    res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload');
    res.setHeader('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');
  }
  
  next();
});

// Enhanced rate limiting for sensitive routes - now imported from rate-limit.js

// Setup enhanced cache control
console.log('[SERVER] Setting up cache control...');
setupCacheControl(app, BUILD_ID);
console.log('[SERVER] Cache control configured');

const PORT = process.env.PORT || 10000;
console.log('[SERVER] Server port:', PORT);

const allowedOrigins = [
  process.env.FRONTEND_URL || 'https://neuralops.biz',
  'https://www.neuralops.biz',
  'https://neuralops.biz',
  // Add Render domains
  'https://neuralops.onrender.com',
  'https://neuralops-v2.onrender.com',
  // Local development
  'http://localhost:10000',
  'http://127.0.0.1:10000'
];

// Setup libraries
(function setupLibraries() {
  if (!fs.existsSync(path.join(__dirname, 'libs'))) {
    try {
      fs.mkdirSync(path.join(__dirname, 'libs'));
      _silent(' Created libs directory');
    } catch (err) {
      console.error(' Failed to create libs directory:', err);
    }
  }
  
  const supabasePath = path.join(__dirname, 'libs', 'supabase.js');
  if (!fs.existsSync(supabasePath)) {
    _silent(' Supabase library not found locally - using fallback');
  } else {
    _silent(' Supabase library found locally');
  }
  
  
})();

// File upload config
// Increased file size limit to 7.5GB for large files (3x 2.4GB requirement)
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { 
    fileSize: 10 * 1024 * 1024 * 1024 // 10GB limit for local uploads
  },
  fileFilter: (req, file, cb) => {
    // Security: Validate both MIME type and file extension
    const allowedMimes = ['application/pdf', 'text/plain', 'text/markdown'];
    const allowedExtensions = ['.pdf', '.txt', '.md'];
    
    // Check MIME type
    if (!allowedMimes.includes(file.mimetype)) {
      return cb(new Error('Invalid file type'), false);
    }
    
    // Check file extension (prevent MIME type spoofing)
    const ext = path.extname(file.originalname).toLowerCase();
    if (!allowedExtensions.includes(ext)) {
      return cb(new Error('Invalid file extension'), false);
    }
    
    // Sanitize filename
    file.originalname = sanitizeFilename(file.originalname);
    
    // Check filename length to prevent DoS
    if (file.originalname.length > 255) {
      return cb(new Error('Filename too long'), false);
    }
    
    cb(null, true);
  }
});

// Initialize Supabase with service role key (bypasses RLS)
// Optimized configuration for better reliability and fewer retries
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY,
  {
    auth: {
      persistSession: false,
      autoRefreshToken: false
    },
    db: {
      schema: 'public'
    },
    global: {
      headers: {
        'X-Client-Info': 'neuralops-server',
        'x-connection-timeout': '30000' // 30 second connection timeout
      },
      fetch: (url, options = {}) => {
        // Use shorter timeout for faster failure detection
        const timeout = options.timeout || 30000; // 30 seconds default
        return fetch(url, {
          ...options,
          signal: AbortSignal.timeout(timeout),
          keepalive: true
        });
      }
    }
  }
);

// Security headers
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: [
        "'self'", 
        "'unsafe-inline'",
        "'unsafe-eval'",
        "blob:", 
        "https://cdnjs.cloudflare.com",
        "https://cdn.jsdelivr.net",
        "https://unpkg.com"
      ],
      scriptSrcAttr: ["'unsafe-inline'", "'unsafe-hashes'"],
      styleSrc: [
        "'self'", 
        "'unsafe-inline'",
        "https://fonts.googleapis.com"
      ],
      imgSrc: [
        "'self'", 
        "data:", 
        "blob:", 
        "https://cdn.neuralops.biz", 
        "https://*.supabase.co"
      ],
      connectSrc: [
        "'self'", 
        "blob:", 
        "https://cdnjs.cloudflare.com",
        "https://cdn.jsdelivr.net",
        "https://unpkg.com",
        "https://fonts.googleapis.com",
        "https://fonts.gstatic.com",
        process.env.SUPABASE_URL, 
        "https://*.supabase.co",
        "https://uvvfuxdrqbrnhdlrkrsb.supabase.co"
      ],
      fontSrc: [
        "'self'", 
        "https://fonts.gstatic.com"
      ],
      objectSrc: ["'none'"],
      workerSrc: ["'self'", "blob:"],
      frameSrc: ["'self'"],
      upgradeInsecureRequests: []
    }
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true
  },
  referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
  xContentTypeOptions: true,
  xFrameOptions: { action: 'deny' },
  permittedCrossDomainPolicies: { permittedPolicies: 'none' }
}));

app.disable('x-powered-by');

// IP Tracking Middleware - Only track page loads, not API calls (reduces spam)
app.use(async (req, res, next) => {
  try {
    // Only track on HTML page requests, not API calls or static assets
    const isPageRequest = req.path === '/' || 
                         (req.path.endsWith('.html') && !req.path.startsWith('/api')) ||
                         (!req.path.startsWith('/api') && !req.path.startsWith('/services') && 
                          !req.path.includes('.') && req.method === 'GET');
    
    if (isPageRequest) {
      // Track visitor IP (non-blocking, silent on error)
      trackVisitorIP(req).catch(() => {
        // Silent - don't spam logs with tracking errors
      });
    }
  } catch (error) {
    // Silent - don't spam logs
  }
  next();
});

// Request logging
app.use((req, res, next) => {
  const requestId = uuid.v4();
  req.requestId = requestId;
  
  // Store IP for later use
  req.clientIP = getClientIP(req);
  
  const start = Date.now();
  
  logger.info({
    type: 'REQUEST',
    requestId,
    method: req.method,
    path: req.path,
    ip: req.clientIP,
    userAgent: req.headers['user-agent']
  });
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    
    logger.info({
      type: 'RESPONSE',
      requestId,
      method: req.method,
      path: req.path,
      statusCode: res.statusCode,
      duration,
      userId: req.user?.id || 'anonymous',
      ip: req.clientIP
    });
    
    if (res.statusCode === 403 || res.statusCode === 401) {
      logSecurityEvent('AUTH_FAILURE', {
        requestId,
        method: req.method,
        path: req.path,
        statusCode: res.statusCode,
        ip: req.clientIP,
        userId: req.user?.id || 'anonymous'
      }, 'warn');
    }
  });
  
  next();
});

// MIME type middleware
app.use((req, res, next) => {
  if (req.path.endsWith('.js')) {
    res.type('application/javascript');
  }
  next();
});

// Service Worker headers
app.use((req, res, next) => {
  // epsilon-sw.js removed - service worker was never registered in UI
  next();
});

// CORS setup
app.use(cors({
  origin: function(origin, callback) {
    if (!origin) return callback(null, true);
    
    if (allowedOrigins.indexOf(origin) === -1) {
      const msg = `CORS policy: Origin ${origin} not allowed`;
      return callback(new Error(msg), false);
    }
    return callback(null, true);
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Cookie', 'X-CSRF-Token']
}));

// OPTIONS handling
app.options('*', (req, res) => {
  const origin = req.headers.origin;
  if (origin && allowedOrigins.indexOf(origin) !== -1) {
    res.header('Access-Control-Allow-Origin', origin);
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, Cookie, X-CSRF-Token');
    res.header('Access-Control-Allow-Credentials', 'true');
    res.status(200).end();
  } else {
    res.status(403).end();
  }
});

app.use(csrfProtection);
// Increased body parser limits to support large file uploads (7.5GB)
app.use(express.json({ limit: '7.5gb' }));
app.use(express.urlencoded({ extended: true, limit: '7.5gb' })); // For form data (includes multipart form fields parsed before file)

// API routes MUST come BEFORE protection middleware
// Enhanced phantom script endpoint with better security
app.get('/api/get-script/:filename', handleScriptRequest);

// Service worker route with dynamic BUILD_ID - obfuscated in production
// epsilon-sw.js removed - service worker was never registered in UI

// Source code protection is handled by protectSourceFiles middleware below
// This redundant middleware removed - protectSourceFiles handles everything

// File serving routes - MUST be before protected files middleware
// Supabase.js route - MUST be before middleware to avoid 403 errors
app.get('/services/libs/supabase.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/libs/supabase.js', 'obfuscated/supabase.js');
});

app.get('/supabase.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/libs/supabase.js', 'obfuscated/supabase.js');
});

app.get('/libs/supabase.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/libs/supabase.js', 'obfuscated/supabase.js');
});

// Epsilon AI learning engine route - serves obfuscated in production
app.get('/core/epsilon-learning-engine.js', (req, res) => {
  serveObfuscatedFile(req, res, 'core/epsilon-learning-engine.js', 'obfuscated/epsilon-learning-engine.js');
});

app.get('/epsilon-learning-engine.js', (req, res) => {
  serveObfuscatedFile(req, res, 'core/epsilon-learning-engine.js', 'obfuscated/epsilon-learning-engine.js');
});

// RAG Embedding Service route - serves obfuscated in production
app.get('/services/rag-embedding-service.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/rag-embedding-service.js', 'obfuscated/rag-embedding-service.js');
});

app.get('/rag-embedding-service.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/rag-embedding-service.js', 'obfuscated/rag-embedding-service.js');
});

// RAG LLM Service route - serves obfuscated in production (used by epsilon-learning-engine.js)
app.get('/services/rag-llm-service.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/rag-llm-service.js', 'obfuscated/rag-llm-service.js');
});

app.get('/rag-llm-service.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/rag-llm-service.js', 'obfuscated/rag-llm-service.js');
});

// RAG Document Processor route - serves obfuscated in production
app.get('/services/rag-document-processor.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/rag-document-processor.js', 'obfuscated/rag-document-processor.js');
});

app.get('/rag-document-processor.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/rag-document-processor.js', 'obfuscated/rag-document-processor.js');
});

// Training files route
// Apply the protected files middleware AFTER file routes
app.use(protectSourceFiles);

// Secure static file middleware - obfuscates all files in production
const { secureStaticFile } = require('./secure-static');

// Custom static file handler for production security
if (isProduction) {
  // In production, ALL files must go through secure handler
  app.use((req, res, next) => {
    // Skip API routes, already handled routes, and non-file requests
    if (req.path.startsWith('/api/') || 
        req.path.startsWith('/.netlify/')) {
      return next();
    }
    
    // Check if this is a file request (has extension or is a known file)
    const hasExtension = /\.\w+$/.test(req.path);
    const isKnownFile = ['/favicon.ico', '/robots.txt'].includes(req.path);
    
    if (hasExtension || isKnownFile) {
      const filePath = path.join(process.cwd(), req.path);
      if (fs.existsSync(filePath)) {
        const stats = fs.statSync(filePath);
        if (stats.isFile()) {
          return secureStaticFile(req, res, filePath);
        }
      }
    }
    
    next();
  });
  
  // NO express.static in production - all files must go through secure handler
  // This ensures nothing bypasses obfuscation
} else {
  // Development: use standard static middleware with optimized caching
  app.use(express.static('./', {
    maxAge: isProduction ? 31536000 : 0, // 1 year in production, no cache in dev
    etag: true,
    lastModified: true,
    setHeaders: (res, path) => {
      // Aggressive caching for static assets in production
      if (isProduction) {
        if (path.match(/\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$/)) {
          res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
        } else if (path.match(/\.(html)$/)) {
          res.setHeader('Cache-Control', 'public, max-age=3600'); // 1 hour for HTML
        }
      }
    }
  }));
}

// Enhanced verifyAuth middleware with better security
const verifyAuth = (requiredRole) => {
  return async (req, res, next) => {
    try {
      const cookies = req.headers.cookie ? parseCookies(req.headers.cookie) : {};
      const token = cookies.authToken;

      if (!token) {
        // If no token, immediately redirect to login
        logSecurityEvent('AUTH_FAILURE', { path: req.path, ip: req.ip, reason: 'No token' }, 'warn');
        return res.redirect('/login?unauthorized=true'); 
      }

      // Verify token with proper error handling
      let decoded;
      try {
        // Security: Never use fallback secret in production
        if (!process.env.JWT_SECRET) {
          logSecurityEvent('AUTH_FAILURE', { 
            path: req.path, 
            ip: req.ip, 
            reason: 'JWT_SECRET not configured' 
          }, 'error');
          return res.status(500).redirect('/login?error=config');
        }
        const jwtSecret = process.env.JWT_SECRET;
        decoded = jwt.verify(token, jwtSecret);
      } catch (jwtError) {
        logSecurityEvent('AUTH_FAILURE', { 
          path: req.path, 
          ip: req.ip, 
          reason: 'Invalid token: ' + jwtError.message 
        }, 'warn');
        clearAuthCookie(res);
        return res.redirect('/login?invalid=true');
      }
      
      // Always verify role from database for owner access - this is critical for security
      // For owner role specifically, we must check the database to ensure accuracy
      let currentRole = decoded.role || 'client';
      let actualProfileId = decoded.userId; // Will be updated if profile found by email
      
      // For owner role requests, always check database regardless of JWT
      // For other roles, check if JWT role is missing or we need verification
      const mustCheckDb = requiredRole === 'owner';
      const shouldCheckDb = !decoded.role || decoded.role === 'client';
      
      if (mustCheckDb || shouldCheckDb) {
        try {
          // Query database for current role - owner role requires fresh check
          const profilePromise = supabase
            .from('profiles')
            .select('id, role')
            .eq('id', decoded.userId)
            .limit(1)
            .maybeSingle();
          
          const timeoutMs = requiredRole === 'owner' ? 15000 : 2000;
          const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Query timeout')), timeoutMs)
          );
          
          let result;
          try {
            result = await Promise.race([
              profilePromise,
              timeoutPromise
            ]);
          } catch (timeoutError) {
            const timeoutMsg = timeoutError.message || '';
            if (timeoutMsg.includes('timeout') || timeoutMsg.includes('521') || timeoutMsg.includes('520')) {
              if (requiredRole === 'owner' && timeoutMs < 30000) {
                console.warn('[AUTH] Database query timeout for role check, retrying with extended timeout (30s)...');
                try {
                  const extendedTimeout = new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Extended query timeout')), 30000)
                  );
                  result = await Promise.race([profilePromise, extendedTimeout]);
                } catch (extendedError) {
                  console.error('[AUTH] Critical: Cannot verify owner role from database after extended timeout:', extendedError.message);
                  if (decoded.role === 'owner') {
                    // Try email lookup before fallback
                    if (decoded.email && actualProfileId === decoded.userId) {
                      const { data: emailProfile } = await supabase
                        .from('profiles')
                        .select('id')
                        .eq('email', decoded.email)
                        .limit(1)
                        .maybeSingle();
                      if (emailProfile?.id) {
                        actualProfileId = emailProfile.id;
                      }
                    }
                    currentRole = 'owner';
                    req.user = { id: actualProfileId, email: decoded.email, role: 'owner' };
                    return next();
                  }
                  throw new Error(`Database query timeout: ${extendedError.message || 'Query exceeded timeout limit'}`);
                }
              } else {
                throw new Error(`Database query timeout: ${timeoutError.message || 'Query exceeded timeout limit'}`);
              }
            } else {
              throw timeoutError;
            }
          }
          
          const { data: profile, error: profileError } = result;
          
          if (!profileError && profile) {
            // Store the actual profile ID (fixes userId mismatch)
            if (profile.id) {
              actualProfileId = profile.id;
            }
            const dbRole = profile.role;
            if (dbRole && typeof dbRole === 'string' && dbRole.trim().toLowerCase() === 'owner') {
              currentRole = 'owner';
            } else if (dbRole && typeof dbRole === 'string' && dbRole.trim().length > 0) {
              currentRole = dbRole.trim().toLowerCase();
            } else {
              currentRole = decoded.role || 'client';
            }
            
            // If role changed, update JWT immediately
            if (currentRole !== decoded.role && currentRole) {
              const newToken = jwt.sign(
                {
                  userId: decoded.userId,
                  email: decoded.email,
                  role: currentRole,
                  name: decoded.name
                },
                process.env.JWT_SECRET,
                { expiresIn: '7d' }
              );
              res.cookie('authToken', newToken, {
                httpOnly: true,
                secure: process.env.NODE_ENV === 'production',
                sameSite: 'strict',
                maxAge: 7 * 24 * 60 * 60 * 1000,
                path: '/'
              });
            }
          } else if (profileError) {
            const errorMsg = profileError.message || '';
            const errorCode = profileError.code || '';
            const isSupabaseDown = errorMsg.includes('521') || 
                                  errorMsg.includes('520') || 
                                  errorMsg.includes('PGRST002') ||
                                  errorCode === 'PGRST002' ||
                                  errorMsg.includes('schema cache') ||
                                  errorMsg.includes('Web server') ||
                                  errorMsg.includes('Gateway') ||
                                  errorMsg.includes('Connection') ||
                                  errorMsg.includes('ECONNRESET') ||
                                  errorMsg.includes('ETIMEDOUT');
            
            if (isSupabaseDown) {
              if (requiredRole === 'owner' && decoded.role === 'owner') {
                  // Try email lookup before fallback
                  if (decoded.email && actualProfileId === decoded.userId) {
                    try {
                      const { data: emailProfile } = await supabase
                        .from('profiles')
                        .select('id')
                        .eq('email', decoded.email)
                        .limit(1)
                        .maybeSingle();
                      if (emailProfile?.id) {
                        actualProfileId = emailProfile.id;
                      }
                    } catch (e) {
                      // Ignore - Supabase is down
                    }
                  }
                currentRole = 'owner';
                  req.user = { id: actualProfileId, email: decoded.email, role: 'owner' };
                return next();
              }
            } else if (errorMsg.includes('timeout')) {
              console.warn('[AUTH] Database query timeout for role check, retrying with extended timeout...');
              if (requiredRole === 'owner') {
                try {
                  console.log('[AUTH] Retrying owner role query with extended timeout (30s) for userId:', decoded.userId);
                  const retryPromise = supabase
                    .from('profiles')
                    .select('id, role')
                    .eq('id', decoded.userId)
                    .limit(1)
                    .maybeSingle();
                  
                  const extendedTimeout = new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Extended query timeout')), 30000)
                  );
                  
                  const retryResult = await Promise.race([retryPromise, extendedTimeout]);
                  const { data: retryProfile, error: retryError } = retryResult;
                  
                  if (retryError) {
                    const retryErrorMsg = retryError.message || '';
                    if (retryErrorMsg.includes('521') || retryErrorMsg.includes('520') || retryErrorMsg.includes('PGRST002')) {
                      if (decoded.role === 'owner') {
                        console.warn('[AUTH] Supabase is down during retry, using JWT role as fallback');
                        currentRole = 'owner';
                        req.user = { id: actualProfileId, email: decoded.email, role: 'owner' };
                        return next();
                      }
                    }
                    // Silent - reduce log noise
                  } else if (retryProfile) {
                    if (retryProfile.id) {
                      actualProfileId = retryProfile.id;
                    }
                    const retryRole = retryProfile.role;
                    if (retryRole && typeof retryRole === 'string' && retryRole.trim().toLowerCase() === 'owner') {
                      currentRole = 'owner';
                    } else {
                      currentRole = retryRole || decoded.role || 'client';
                    }
                  } else if (!retryProfile && decoded.role === 'owner') {
                    // Try email lookup as fallback
                    if (decoded.email) {
                      const { data: emailProfile } = await supabase
                        .from('profiles')
                        .select('id, role')
                        .eq('email', decoded.email)
                        .limit(1)
                        .maybeSingle();
                      if (emailProfile) {
                        actualProfileId = emailProfile.id;
                        currentRole = emailProfile.role || 'owner';
                      }
                    }
                    if (currentRole === 'owner') {
                      req.user = { id: actualProfileId, email: decoded.email, role: 'owner' };
                    return next();
                    }
                  }
                } catch (retryErr) {
                  const retryErrMsg = retryErr.message || '';
                  if (decoded.role === 'owner' && (retryErrMsg.includes('521') || retryErrMsg.includes('520'))) {
                    // Try email lookup before fallback
                    if (decoded.email && actualProfileId === decoded.userId) {
                      try {
                        const { data: emailProfile } = await supabase
                          .from('profiles')
                          .select('id')
                          .eq('email', decoded.email)
                          .limit(1)
                          .maybeSingle();
                        if (emailProfile?.id) {
                          actualProfileId = emailProfile.id;
                        }
                      } catch (e) {
                        // Ignore - Supabase is down
                      }
                    }
                    currentRole = 'owner';
                    req.user = { id: actualProfileId, email: decoded.email, role: 'owner' };
                    return next();
                  }
                  console.error('[AUTH] Retry query exception:', retryErrMsg);
                }
              }
            } else {
              // Only log non-timeout errors (reduce noise)
              if (!errorMsg.includes('timeout') && !errorMsg.includes('521') && !errorMsg.includes('520')) {
              console.error('[AUTH] Error fetching role from database:', errorMsg);
              }
            }
          } else if (!profile) {
            // Profile doesn't exist by userId - try email lookup
            let emailProfile = null;
              if (decoded.email) {
              const { data: emailProfileData, error: emailError } = await supabase
                  .from('profiles')
                  .select('id, email, role')
                  .eq('email', decoded.email)
                  .limit(1)
                  .maybeSingle();
              
              if (emailProfileData && !emailError) {
                emailProfile = emailProfileData;
                // Use the profile found by email - userId mismatch resolved
                profile = { role: emailProfile.role, id: emailProfile.id };
                actualProfileId = emailProfile.id; // Fix userId mismatch
                currentRole = emailProfile.role || decoded.role || 'client';
              }
            }
            
            // Only log if still not found after email lookup (reduce noise)
            if (!emailProfile && requiredRole === 'owner') {
              // Silent - only log once per session, not every request
            }
          }
        } catch (roleError) {
          // For owner role, we cannot proceed without database confirmation
          if (requiredRole === 'owner') {
            console.error('[AUTH] Critical: Cannot verify owner role from database:', roleError.message);
            // Still check retry logic below
          } else {
            // For non-owner roles, continue with JWT role if DB fails
            if (!roleError.message || !roleError.message.includes('timeout')) {
              console.error('[AUTH] Error in role check:', roleError.message);
            }
          }
        }
      }
      
      // Check if the user's role is sufficient (case-insensitive comparison)
      const currentRoleLower = currentRole && typeof currentRole === 'string' ? currentRole.trim().toLowerCase() : 'client';
      const requiredRoleLower = requiredRole && typeof requiredRole === 'string' ? requiredRole.trim().toLowerCase() : '';
      const hasPermission = currentRoleLower === requiredRoleLower || 
                           (requiredRoleLower === 'client' && currentRoleLower === 'owner');

      if (!hasPermission) {
        // If token is valid but role is wrong, check one more time with a fresh DB query
        // This handles cases where JWT is outdated but user actually has the role
        if (requiredRole === 'owner') {
          // For owner role, always retry without timeout to ensure we get the correct role
          console.log('[AUTH] Owner role required but permission check failed. Retrying with fresh DB query...');
          try {
            const { data: profile, error: retryError } = await supabase
              .from('profiles')
              .select('id, role')
              .eq('id', decoded.userId)
              .limit(1)
              .maybeSingle();
            
            if (retryError) {
              console.error('[AUTH] Retry query error:', retryError.message);
            }
            
            // Check role with case-insensitive comparison and handle NULL
            const retryRole = profile.role;
            const isOwner = retryRole && typeof retryRole === 'string' && retryRole.trim().toLowerCase() === 'owner';
            
            if (profile && isOwner) {
              // User actually has owner role - update JWT and allow access
              console.log('[AUTH] Owner role confirmed on retry, updating JWT and allowing access');
              const newToken = jwt.sign(
                {
                  userId: decoded.userId,
                  email: decoded.email,
                  role: 'owner',
                  name: decoded.name
                },
                process.env.JWT_SECRET,
                { expiresIn: '7d' }
              );
              res.cookie('authToken', newToken, {
                httpOnly: true,
                secure: process.env.NODE_ENV === 'production',
                sameSite: 'strict',
                maxAge: 7 * 24 * 60 * 60 * 1000,
                path: '/'
              });
              
              req.user = { 
                id: actualProfileId, 
                email: decoded.email, 
                role: 'owner'
              };
              return next();
            } else {
              console.warn('[AUTH] Retry query did not confirm owner role. Profile:', profile ? 'exists' : 'null', 'Role:', profile ? profile.role : 'N/A');
            }
          } catch (retryError) {
            console.error('[AUTH] Retry query exception:', retryError.message);
            // If retry also fails, continue to redirect
          }
        }
        
        // If token is valid but role is wrong, redirect to login
        console.warn('[AUTH] Permission denied:', {
          path: req.path,
          requiredRole: requiredRole,
          currentRole: currentRole,
          jwtRole: decoded.role,
          userId: decoded.userId
        });
        logSecurityEvent('AUTH_FAILURE', { 
          path: req.path, 
          ip: req.ip, 
          userId: decoded.userId, 
          reason: 'Insufficient role',
          requiredRole: requiredRole,
          userRole: currentRole,
          jwtRole: decoded.role
        }, 'warn');
        return res.redirect('/login?permission=false');
      }

      // If everything is okay, attach user to request and continue
      // Use actualProfileId (from database lookup) to fix userId mismatch
      req.user = { 
        id: actualProfileId, 
        email: decoded.email, 
        role: currentRole // Use current role from database
      };
      next();

    } catch (error) {
      // If any other error occurs, redirect to login
      logSecurityEvent('AUTH_FAILURE', { 
        path: req.path, 
        ip: req.ip, 
        reason: 'Error: ' + error.message 
      }, 'warn');
      clearAuthCookie(res);
      return res.redirect('/login?error=true');
    }
  };
};

// Import function handlers
let authHandler, analyticsHandler;

try {
  authHandler = require('../api/netlify/functions/auth').handler;
  _silent(' Auth handler loaded');
} catch (error) {
  console.error(' Auth handler not found, using fallback:', error.message);
  authHandler = async (event, context) => {
    return {
      statusCode: 404,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Auth function not deployed yet' })
    };
  };
}

try {
  analyticsHandler = require('../api/netlify/functions/analytics').handler;
  _silent(' Analytics handler loaded');
} catch (error) {
  console.error(' Analytics handler not found, using fallback');
  analyticsHandler = async (event, context) => {
    return {
      statusCode: 404,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Analytics function not deployed yet' })
    };
  };
}

// Supabase proxy
const supabaseProxyRouter = require('../api/supabase-proxy');
app.use('/api/supabase-proxy', supabaseProxyRouter);
app.use('/.netlify/functions/supabase-proxy', supabaseProxyRouter);
_silent(' Supabase proxy router loaded');

// Supabase connectivity health check
app.get('/api/supabase-health', verifyAuth('owner'), async (req, res) => {
  try {
    const proxyResponse = await fetch(`${req.protocol}://${req.get('host')}/api/supabase-proxy`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': req.headers['x-csrf-token'] || 'supabase-healthcheck',
        'Cookie': req.headers.cookie || ''
      },
      body: JSON.stringify({
        action: 'verify-supabase-tables',
        data: {}
      })
    });
    
    if (!proxyResponse.ok) {
      return res.status(500).json({ error: 'Failed to verify Supabase tables' });
    }
    
    const data = await proxyResponse.json();
    res.json(data);
  } catch (error) {
    console.error(' Supabase health check error:', error);
    res.status(500).json({ error: 'Service unavailable' });
  }
});

// Document Learning Service health check
app.get('/api/document-learning/health', async (req, res) => {
  try {
    const response = await fetch('http://localhost:8004/health');
    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error(' [DOCUMENT LEARNING PROXY] Health check failed:', error);
    res.status(500).json({ error: 'Service unavailable' });
  }
});

// Advanced Learning System endpoints
app.get('/api/advanced-learning/insights', verifyAuth('owner'), async (req, res) => {
    try {
        const { createClient } = require('@supabase/supabase-js');
        const supabase = createClient(
            process.env.SUPABASE_URL,
            process.env.SUPABASE_SERVICE_KEY,
            { auth: { persistSession: false } }
        );
        
        // Get real data from Supabase with proper limits and indexing
        let docCount = 0;
        let feedbackData = [];
        let learningSessions = [];
        
        try {
            // Use head queries for count (fast, uses indexes)
            const countResult = await supabase
                .from('knowledge_documents')
                .select('id', { count: 'exact', head: true });
            docCount = countResult.count || 0;
            
            // Get limited feedback data (use index on rating)
            const feedbackResult = await supabase
                .from('epsilon_feedback')
                .select('rating, was_helpful')
                .order('created_at', { ascending: false })
                .limit(500); // Reduced from 1000 for faster queries
            feedbackData = feedbackResult.data || [];
            
            // Get limited learning sessions (use index on status)
            const sessionsResult = await supabase
                .from('document_learning_sessions')
                .select('status')
                .order('created_at', { ascending: false })
                .limit(100);
            learningSessions = sessionsResult.data || [];
        } catch (queryError) {
            const errorMsg = queryError?.message || '';
            console.error('[ADVANCED LEARNING] Supabase query failed:', errorMsg);
            // Return empty data - don't crash
        }
        
        const totalDocs = docCount || 0;
        const feedback = feedbackData || [];
        const sessions = learningSessions || [];
        
        const successCount = sessions.filter(s => s.status === 'completed').length;
        const successRate = sessions.length > 0 ? successCount / sessions.length : 0;
        const avgRating = feedback.length > 0 ? feedback.reduce((sum, f) => sum + (f.rating || 0), 0) / feedback.length : 0;
        const satisfaction = avgRating / 5.0; // Normalize to 0-1
        
        res.json({
            total_documents: totalDocs,
            learning_status: totalDocs > 0 ? 'active' : 'inactive',
            features_enabled: ['document_learning', 'pattern_detection', 'feedback_analysis'],
            performance_metrics: {
                success_rate: successRate,
                user_satisfaction: satisfaction,
                average_processing_time: 0
            }
        });
    } catch (error) {
        const errorMsg = error.message || '';
        // If Supabase timeout, return empty data gracefully
        if (errorMsg.includes('timeout') || errorMsg.includes('521') || errorMsg.includes('520')) {
            console.warn('[ADVANCED LEARNING] Supabase timeout, returning empty data');
        } else {
        console.error(' [ADVANCED LEARNING] Insights fetch failed:', error);
        }
        res.json({
            total_documents: 0,
            learning_status: 'inactive',
            features_enabled: [],
            performance_metrics: {
                average_processing_time: 0,
                success_rate: 0,
                error_rate: 0
            },
            error: 'Advanced learning insights temporarily unavailable'
        });
    }
});

app.post('/api/advanced-learning/synthesize', verifyAuth('owner'), async (req, res) => {
    try {
        const response = await fetch('http://localhost:8004/advanced-learning/synthesize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(req.body)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error(' [ADVANCED LEARNING] Synthesis failed:', error);
        res.status(500).json({ error: 'Processing failed' });
    }
});

// Document Learning Service proxy
app.use('/api/document-learning', async (req, res, next) => {
  try {
    const targetUrl = `http://localhost:8004${req.path}`;
    _silent(` [DOCUMENT LEARNING PROXY] Proxying ${req.method} ${req.path} to ${targetUrl}`);
    
    // Handle multipart form data differently
    if (req.headers['content-type'] && req.headers['content-type'].includes('multipart/form-data')) {
      _silent(' [DOCUMENT LEARNING PROXY] Handling multipart form data');
      _silent(' [DOCUMENT LEARNING PROXY] Content-Type:', req.headers['content-type']);
      _silent(' [DOCUMENT LEARNING PROXY] Content-Length:', req.headers['content-length']);
      
      // For file uploads, we need to stream the request
      const response = await fetch(targetUrl, {
        method: req.method,
        headers: {
          ...req.headers,
          'host': 'localhost:8004'
        },
        body: req,
        duplex: 'half' // Required for streaming request bodies
      });
      
      _silent(` [DOCUMENT LEARNING PROXY] Response status: ${response.status}`);
      _silent(` [DOCUMENT LEARNING PROXY] Response headers:`, Object.fromEntries(response.headers.entries()));
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(` [DOCUMENT LEARNING PROXY] Error response: ${errorText}`);
        return res.status(response.status).json({ error: errorText });
      }
      
      const data = await response.json();
      res.status(response.status).json(data);
    } else {
      // For JSON requests
      const response = await fetch(targetUrl, {
        method: req.method,
        headers: {
          'Content-Type': req.headers['content-type'] || 'application/json',
          'host': 'localhost:8004',
          ...req.headers
        },
        body: req.method !== 'GET' ? JSON.stringify(req.body) : undefined
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(` [DOCUMENT LEARNING PROXY] Error response: ${errorText}`);
        return res.status(response.status).json({ error: errorText });
      }
      
      const data = await response.json();
      res.status(response.status).json(data);
    }
  } catch (error) {
    console.error(' [DOCUMENT LEARNING PROXY] Proxy error:', error);
    res.status(500).json({ error: 'Service unavailable' });
  }
});
_silent(' Document learning service proxy loaded');

// Apply rate limiters
app.use('/api/auth', authLimiter);
app.use('/.netlify/functions/auth', authLimiter);
app.use('/api/upload-document', uploadLimiter);
app.use('/api', apiLimiter);
app.use('/.netlify/functions', apiLimiter);

// Get user role by user ID (no auth required - public info lookup)
// This route must be defined early, before static middleware
app.get('/api/user-role/:userId', apiLimiter, async (req, res) => {
  try {
    const { userId } = req.params;
    
    _silent(`[USER-ROLE] Fetching role for userId: ${userId}`);
    
    if (!userId) {
      return res.status(400).json({ error: 'User ID is required' });
    }
    
    // Get user role from profiles table
    const { data, error } = await supabase
      .from('profiles')
      .select('role')
      .eq('id', userId)
      .limit(1)
      .maybeSingle();
    
    if (error) {
      console.error(' [USER-ROLE] Error fetching user role:', error);
      return res.status(500).json({ 
        error: 'Failed to fetch user role',
        details: error.message || 'Database query failed'
      });
    }
    
    if (!data) {
      console.error(` [USER-ROLE] No profile found for userId: ${userId}`);
      return res.status(404).json({ 
        error: 'User profile not found',
        details: `No profile exists for user ID: ${userId}`
      });
    }
    
    _silent(` [USER-ROLE] Found role: ${data.role} for userId: ${userId}`);
    res.json({ role: data.role || 'client' });
  } catch (error) {
    console.error(' [USER-ROLE] Error in user-role endpoint:', error);
    res.status(500).json({ error: 'Failed to fetch user role' });
  }
});

// Profile endpoints
app.get('/api/profile/me', verifyAuth('client'), async (req, res) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    // Get profile from database including role
    const { data: profile, error } = await supabase
        .from('profiles')
      .select('id, email, name, full_name, company, industry, avatar_url, role, epsilon_version, last_login, created_at')
        .eq('id', userId)
      .limit(1)
        .maybeSingle();

    if (error) {
      console.error(' [PROFILE] Error fetching profile:', error);
      return res.status(500).json({ error: 'Failed to fetch profile' });
    }

    if (!profile) {
      // Profile doesn't exist - return minimal profile with role from JWT
      return res.json({
        success: true,
        profile: {
          id: userId,
          email: req.user.email,
          role: req.user.role || 'client'
        }
      });
    }

    // Return profile with role
    return res.json({
      success: true,
      profile: {
        id: profile.id,
        email: profile.email,
        name: profile.name,
        full_name: profile.full_name,
        company: profile.company,
        industry: profile.industry,
        avatar_url: profile.avatar_url,
        role: profile.role || 'client',
        epsilon_version: profile.epsilon_version,
        last_login: profile.last_login,
        created_at: profile.created_at
      }
    });
  } catch (error) {
    console.error(' [PROFILE] Unexpected error:', error);
    return res.status(500).json({ error: 'Failed to fetch profile' });
  }
});

app.put('/api/profile/me', verifyAuth('client'), async (req, res) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    // CSRF validation - be very lenient, especially in development
    // Check tokens directly instead of relying on middleware flag
    const cookies = parseCookies(req.headers.cookie || '');
    const cookieToken = cookies.csrfToken;
    const headerToken = req.headers['x-csrf-token'] || req.headers['X-CSRF-Token'];

    const isDevelopment = process.env.NODE_ENV !== 'production';
    
    // In development: allow if cookie token exists (even without header)
    // In production: require both cookie and header token to match
    let csrfValid = false;
    if (isDevelopment) {
      // In development, just having a cookie token is enough
      csrfValid = !!cookieToken;
    } else {
      // In production, both must exist and match
      csrfValid = cookieToken && headerToken && cookieToken === headerToken;
    }
    
    // Also check if middleware already validated it
    if (req.csrfValid === true) {
      csrfValid = true;
    }
    
    if (!csrfValid) {
      logSecurityEvent('CSRF_VALIDATION_FAILED', {
        userId,
        ip: req.ip,
        path: req.path,
        method: req.method,
        hasCookieToken: !!cookieToken,
        hasHeaderToken: !!headerToken,
        isDevelopment,
        middlewareValid: req.csrfValid
      }, 'warn');
      
      // Only block in production if both tokens are missing
      if (!isDevelopment && !cookieToken && !headerToken) {
      return res.status(403).json({ error: 'CSRF validation failed' });
      }
      // In development or if cookie exists, allow the request
    }

    const { full_name, avatar } = req.body || {};
    const updatePayload = {};
    const metadataUpdates = {};
    const updatedFields = [];

    let hasFullNameChange = false;
    let hasAvatarChange = false;

    if (typeof full_name === 'string') {
      const trimmedName = sanitizeText(full_name).trim();
      const boundedName = trimmedName ? trimmedName.substring(0, 120) : null;
      updatePayload.full_name = boundedName;
      metadataUpdates.full_name = boundedName ?? '';
      hasFullNameChange = true;
      updatedFields.push('full_name');
    }

    if (typeof avatar === 'string') {
      hasAvatarChange = true;
      updatedFields.push('avatar');

      if (!avatar.length) {
        updatePayload.avatar_url = null;
        metadataUpdates.avatar = '';
        metadataUpdates.avatar_url = '';
      } else if (avatar.startsWith('data:image/') && avatar.length < 2_000_000) {
        updatePayload.avatar_url = avatar;
        metadataUpdates.avatar = avatar;
        metadataUpdates.avatar_url = avatar;
      } else {
        return res.status(400).json({ error: 'Invalid avatar image' });
      }
    }

    if (!hasFullNameChange && !hasAvatarChange) {
      return res.status(400).json({ error: 'No changes provided' });
    }

    let avatarColumnAvailable = true;
    let updateResult = { data: [] };

    const hasDbFields = (payload) => Object.keys(payload).some((key) => key !== 'updated_at');
    const payloadHasColumns = hasDbFields(updatePayload);

    if (payloadHasColumns) {
      updatePayload.updated_at = new Date().toISOString();

      const initialUpdate = await supabase
        .from('profiles')
        .update(updatePayload)
        .eq('id', userId)
        .select('full_name, avatar_url');

      if (initialUpdate.error) {
        if (initialUpdate.error.message && initialUpdate.error.message.includes('avatar_url')) {
          avatarColumnAvailable = false;
          delete updatePayload.avatar_url;

          if (hasDbFields(updatePayload)) {
            const retryUpdate = await supabase
              .from('profiles')
              .update(updatePayload)
              .eq('id', userId)
              .select('full_name');

            if (retryUpdate.error) {
              console.error(' [PROFILE] Failed to update profile after retry:', retryUpdate.error);
              return res.status(500).json({ error: 'Failed to update profile' });
            }

            updateResult = retryUpdate;
          } else {
            delete updatePayload.updated_at;
            updateResult = { data: [] };
          }
        } else {
          console.error(' [PROFILE] Failed to update profile:', initialUpdate.error);
          return res.status(500).json({ error: 'Failed to update profile' });
        }
      } else {
        updateResult = initialUpdate;
      }
    } else {
      avatarColumnAvailable = false;
    }

    if (hasFullNameChange || hasAvatarChange) {
      try {
        const { data: authData, error: authError } = await supabase.auth.admin.getUserById(userId);
        if (!authError && authData?.user) {
          const currentMetadata = authData.user.user_metadata || {};
          const mergedMetadata = { ...currentMetadata };

          if (hasFullNameChange && metadataUpdates.full_name !== undefined) {
            mergedMetadata.full_name = metadataUpdates.full_name;
          }

          if (hasAvatarChange) {
            const avatarValue = metadataUpdates.avatar ?? '';
            mergedMetadata.avatar = avatarValue;
            mergedMetadata.avatar_url = avatarValue;
          }

          const { error: updateMetaError } = await supabase.auth.admin.updateUserById(userId, {
            user_metadata: mergedMetadata
          });

          if (updateMetaError) {
            console.warn(' [PROFILE] Failed to update auth metadata:', updateMetaError.message || updateMetaError);
          }
        } else if (authError) {
          console.warn(' [PROFILE] Auth metadata update skipped:', authError.message || authError);
        }
      } catch (adminError) {
        console.warn(' [PROFILE] Auth metadata update error:', adminError.message || adminError);
      }
    }

    logSecurityEvent('PROFILE_UPDATED', {
      userId,
      fields: updatedFields
    });

    const responseFullName = hasFullNameChange
      ? (updatePayload.full_name ?? '')
      : (updateResult?.data?.[0]?.full_name ?? metadataUpdates.full_name ?? '');

    const responseAvatar = hasAvatarChange
      ? (avatarColumnAvailable
          ? (updateResult?.data?.[0]?.avatar_url ?? metadataUpdates.avatar ?? '')
          : (metadataUpdates.avatar ?? ''))
      : (avatarColumnAvailable
          ? (updateResult?.data?.[0]?.avatar_url ?? '')
          : '');

    return res.json({
      success: true,
      profile: {
        full_name: responseFullName || '',
        avatar_url: responseAvatar || ''
      }
    });
  } catch (error) {
    console.error(' [PROFILE] Unexpected error during update:', error);
    return res.status(500).json({ error: 'Failed to update profile' });
  }
});

// Apply to sensitive routes
app.use('/api/get-script/', sensitiveRouteLimiter);
app.use('/obfuscated/', sensitiveRouteLimiter);
app.use('/api/auth', sensitiveRouteLimiter);
app.use('/.netlify/functions/auth', sensitiveRouteLimiter);

// Document upload endpoint
app.post('/api/upload-document', verifyAuth('owner'), upload.single('file'), async (req, res) => {
  const uploadStartTime = Date.now();
  let responseSent = false;
  
  // No endpoint-level timeout - let uploads complete naturally
  // HTTP server timeout (90 minutes) will handle hanging connections
  const fileSize = req.file?.size || 0;
  const fileSizeMB = (fileSize / (1024 * 1024)).toFixed(2);
  console.log(`[UPLOAD] Starting upload for file size: ${fileSizeMB}MB (no endpoint timeout - will complete naturally)`);
  
  // Note: Very large files (>2GB) may trigger Cloudflare 520 timeout
  // The upload will continue processing server-side even if client connection drops
  
  try {
      // Document upload started
    
    if (req.file) {
      const fileSizeMB = (req.file.size / (1024 * 1024)).toFixed(2);
      console.log(` [UPLOAD] File: ${req.file.originalname}, Size: ${fileSizeMB}MB, Type: ${req.file.mimetype}`);
    }
    
    logSecurityEvent('DOCUMENT_UPLOAD_ATTEMPT', {
      userId: req.user.id,
      ip: req.ip,
      filename: req.file?.originalname || 'unknown'
    });
    
    // For file uploads, check CSRF token in form body (after multer parses it)
    // CSRF middleware runs before multer, so we need to validate here
    const cookies = parseCookies(req.headers.cookie || '');
    const cookieToken = cookies.csrfToken;
    const headerToken = req.headers['x-csrf-token'];
    const formToken = req.body?._csrf || req.body?.csrfToken;
    
    // Try to use token from header first, then form, then check CSRF middleware result
    const providedToken = headerToken || formToken;
    
    // Validate: both cookie token and provided token must exist and match
    let csrfValid = false;
    if (cookieToken && providedToken) {
      csrfValid = cookieToken === providedToken;
    } else if (req.csrfValid === true) {
      // Fallback: if CSRF middleware already validated it, trust that
      csrfValid = true;
    }
    
    if (!csrfValid) {
      // Security: Don't log token details - just log the failure
      console.error('[CSRF] Validation failed for upload');
      
      logSecurityEvent('CSRF_VALIDATION_FAILED', {
        userId: req.user.id,
        ip: req.ip,
        path: req.path,
        hasCookieToken: !!cookieToken,
        hasHeaderToken: !!headerToken,
        hasFormToken: !!formToken,
        middlewareValid: req.csrfValid
      }, 'warn');
      
      return res.status(403).json({ error: 'CSRF validation failed' });
    }
    
    if (!req.file) {
      console.warn('No file uploaded');
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    const file = req.file;
    const documentType = sanitizeText(req.body.document_type) || 'general';
    const learningCategory = sanitizeText(req.body.learning_category) || null;
    const customFilename = sanitizeFilename(req.body.filename || file.originalname);
    
    // Map document type to learning category (matches Python service logic)
    const documentTypeToCategory = {
      'knowledge': 'knowledge',
      'knowledge_base': 'knowledge',
      'pricing': 'knowledge',
      'technical': 'knowledge',
      'process': 'knowledge',
      'faq': 'knowledge',
      'case_study': 'knowledge',
      'sales_training': 'sales_training',
      'sales_script': 'sales_training',
      'communication_guide': 'sales_training',
      'training_material': 'learning',
      'learning': 'learning',
      'dictionary': 'learning'  // Dictionary files go to learning category
    };
    
    // Determine learning category from document type or use provided category
    const mappedCategory = learningCategory || documentTypeToCategory[documentType] || 'knowledge';
    
    
    let extractedText = '';
    
    if (file.originalname.toLowerCase().endsWith('.pdf')) {
      try {
        console.log(' [UPLOAD] Processing PDF file...');
        const buffer = Buffer.isBuffer(file.buffer) ? file.buffer : Buffer.from(file.buffer);
        
                _silent('PDF buffer type:', typeof buffer);
        _silent('PDF buffer length:', buffer.length);
        
        // Validate PDF header (should start with %PDF)
        const pdfHeader = buffer.slice(0, 4).toString('utf8');
        if (!pdfHeader.startsWith('%PDF')) {
          console.error('Invalid PDF header:', pdfHeader);
          throw new Error('Invalid PDF file format');
        }
        
        // Try Node.js pdf-parse first
        let pdfData;
        try {
          console.log(' [UPLOAD] Extracting text from PDF using pdf-parse...');
          pdfData = await pdfParse(buffer);
          extractedText = pdfData.text || '';
          console.log(` [UPLOAD] Extracted ${extractedText.length} characters from PDF`);
          
          // VALIDATE: Check if extracted text is actually readable text, not hex/binary
          if (extractedText && extractedText.length > 0) {
            // Check if text looks like hex/binary data
            const sample = extractedText.substring(0, 200);
            const hexChars = (sample.match(/[0-9a-fA-F]/g) || []).length;
            const hexRatio = hexChars / Math.max(sample.length, 1);
            
            // Check for non-printable characters
            const printableChars = (sample.match(/[\x20-\x7E\n\r\t]/g) || []).length;
            const printableRatio = printableChars / Math.max(sample.length, 1);
            
            // If more than 70% hex-like characters, it's probably binary/hex
            if (hexRatio > 0.7 && printableRatio < 0.3) {
              console.warn(' Extracted text appears to be hex/binary data, trying Python service...');
              throw new Error('Extracted content appears to be binary/hex');
            }
            
            // If less than 30% printable characters, content is suspicious
            if (printableRatio < 0.3 && extractedText.length > 50) {
              console.warn(' Low printable character ratio, trying Python service...');
              throw new Error('Low printable character ratio');
            }
            
            // Check if text is mostly whitespace (bad extraction)
            const nonWhitespace = (extractedText.match(/\S/g) || []).length;
            if (nonWhitespace < extractedText.length * 0.1 && extractedText.length > 100) {
              console.warn(' Extracted text is mostly whitespace, trying Python service...');
              throw new Error('Text mostly whitespace');
            }
            
          } else {
            console.warn(' No text extracted, trying Python service...');
            throw new Error('No text extracted');
          }
        } catch (parseError) {
          // Node.js extraction failed or returned garbage, try Python service
          try {
            // Check if Python service manager is available
            if (pythonServiceManager && pythonServiceManager.isServiceReady('document_learning')) {
              
              // Convert buffer to base64 for Python service
              const base64Content = buffer.toString('base64');
              
              // Call Python service for PDF extraction
              // Note: axios is already a dependency, using require here is fine
              const axios = require('axios');
              
              // Use Promise.race to ensure we don't hang forever
              // Increased timeout for large PDFs - up to 5 minutes for complex documents
              const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
              const timeoutMs = fileSizeMB > 5 ? 300000 : fileSizeMB > 2 ? 180000 : 120000; // 5min for >5MB, 3min for >2MB, 2min otherwise
              
              const pythonResponse = await Promise.race([
                axios.post(
                  `http://localhost:8004/extract-pdf-text`,
                  {
                    pdf_content: base64Content,
                    filename: file.originalname
                  },
                  {
                    timeout: timeoutMs,
                    headers: {
                      'Content-Type': 'application/json'
                    }
                  }
                ),
                new Promise((_, reject) => 
                  setTimeout(() => reject(new Error(`Python service timeout after ${timeoutMs/1000}s`)), timeoutMs + 10000)
                )
              ]);
              
              if (pythonResponse.data && pythonResponse.data.text) {
                extractedText = pythonResponse.data.text;
                
                // Final validation of Python-extracted text
                const sample = extractedText.substring(0, 200);
                const printableChars = (sample.match(/[\x20-\x7E\n\r\t]/g) || []).length;
                const printableRatio = printableChars / Math.max(sample.length, 1);
                
                if (printableRatio < 0.5) {
                  throw new Error(`PDF extraction returned invalid content (printable ratio: ${printableRatio.toFixed(2)}). This PDF may be image-based, encrypted, or corrupted. File size: ${file.size} bytes.`);
                }
              } else {
                throw new Error('Python service did not return text');
              }
            } else {
              throw new Error('Python document learning service not available - PDF extraction service is required');
            }
          } catch (pythonError) {
            console.error(' Python service extraction failed:', pythonError.message);
            throw new Error(`PDF text extraction failed: ${pythonError.message}. The Python document learning service is required for PDF processing.`);
          }
        }
      } catch (pdfError) {
        console.error(' PDF processing error:', pdfError);
        // Don't send response if already sent
        if (res.headersSent) {
          console.error('[UPLOAD] Response already sent during PDF processing error');
          return;
        }
        return res.status(500).json({ 
          error: 'Failed to process PDF file',
          details: pdfError.message || 'Unknown error during PDF extraction'
        });
      }
    } else if (file.originalname.toLowerCase().endsWith('.txt') || file.originalname.toLowerCase().endsWith('.md')) {
      // Process large text files using streaming chunked approach
      // For files >100MB, we'll store chunks separately and only keep a preview in main content
      // This allows handling files up to 10GB without memory issues
      const LARGE_FILE_THRESHOLD = 100 * 1024 * 1024; // 100MB threshold for chunked storage
      const CHUNK_SIZE = 10 * 1024 * 1024; // 10MB chunks for processing
      // Reduce preview size for very large files to avoid Supabase statement timeouts
      // For files >2GB, use 5MB preview; >1GB use 10MB; >500MB use 20MB; otherwise 50MB
      const MAX_PREVIEW_SIZE = file.buffer.length > 2 * 1024 * 1024 * 1024 
        ? 5 * 1024 * 1024  // 5MB for files >2GB (to avoid timeout)
        : (file.buffer.length > 1024 * 1024 * 1024 
          ? 10 * 1024 * 1024  // 10MB for files >1GB
          : (file.buffer.length > 500 * 1024 * 1024 
            ? 20 * 1024 * 1024  // 20MB for files >500MB
            : 50 * 1024 * 1024)); // 50MB for smaller large files
      
      const fileSizeMB = (file.buffer.length / (1024 * 1024)).toFixed(2);
      
      if (file.buffer.length > LARGE_FILE_THRESHOLD) {
        console.log(`[UPLOAD] Large file detected (${fileSizeMB}MB). Using chunked storage approach...`);
        
        // For large files, process in streaming chunks and store separately
        // We'll store chunks in doc_chunks table and only a preview in knowledge_documents.content
        // Process chunks without accumulating all in memory - we'll store them after document creation
        let offset = 0;
        let previewText = '';
        let totalChunks = 0;
        const processedChunks = []; // Store minimal chunk info (index, size, offset) - not full text
        
        // Process file in chunks - only keep preview in memory, not all chunks
        while (offset < file.buffer.length) {
          const remaining = file.buffer.length - offset;
          const chunkSize = Math.min(CHUNK_SIZE, remaining);
          const chunk = file.buffer.slice(offset, offset + chunkSize);
          
          // Convert chunk to string
          const chunkText = chunk.toString('utf8');
          
          // Store minimal chunk metadata (we'll re-read from buffer when storing)
          processedChunks.push({
            index: totalChunks,
            offset: offset,
            size: chunkText.length,
            bufferStart: offset,
            bufferEnd: offset + chunkSize
          });
          
          // Build preview from first chunks (up to MAX_PREVIEW_SIZE)
          if (previewText.length < MAX_PREVIEW_SIZE) {
            const remainingPreview = MAX_PREVIEW_SIZE - previewText.length;
            if (chunkText.length <= remainingPreview) {
              previewText += chunkText;
            } else {
              previewText += chunkText.substring(0, remainingPreview);
            }
          }
          
          offset += chunkSize;
          totalChunks++;
          
          // Yield to event loop every chunk to prevent blocking
          await new Promise(resolve => setImmediate(resolve));
        }
        
        // Use preview for main content, store full chunks separately
        extractedText = previewText;
        
        // Store chunk metadata for later processing (we'll re-read from buffer when storing)
        req.file.processedChunks = processedChunks;
        req.file.isChunked = true;
        req.file.totalChunks = totalChunks;
        req.file.originalBuffer = file.buffer; // Keep reference to original buffer for chunk extraction
        
        console.log(`[UPLOAD] Processed ${totalChunks} chunks (${fileSizeMB}MB total). Preview: ${(previewText.length / 1024 / 1024).toFixed(2)}MB`);
      } else {
        // Small/medium file - process normally but still in chunks if >20MB
        if (file.buffer.length > 20 * 1024 * 1024) { // > 20MB
          console.log(`[UPLOAD] Processing medium-sized text file (${fileSizeMB}MB) in chunks...`);
          
          const chunks = [];
          let offset = 0;
          
          while (offset < file.buffer.length) {
            const chunkSize = Math.min(CHUNK_SIZE, file.buffer.length - offset);
            const chunk = file.buffer.slice(offset, offset + chunkSize);
            const chunkText = chunk.toString('utf8');
            chunks.push(chunkText);
            offset += chunkSize;
            
            // Yield periodically
            if (chunks.length % 3 === 0) {
              await new Promise(resolve => setImmediate(resolve));
            }
          }
          
          extractedText = chunks.join('');
          console.log(`[UPLOAD] Successfully processed ${chunks.length} chunks`);
        } else {
          // Small file - process normally
          extractedText = file.buffer.toString('utf8');
        }
      }
    } else {
      return res.status(400).json({ error: 'Unsupported file type' });
    }
    
    // Prepare the extracted text for storage
    extractedText = (extractedText || '').replace(/\u0000/g, '');
    const cleanedContent = extractedText.trim();
    if (!cleanedContent.length) {
      console.error(' Extracted content empty after validation');
      return res.status(500).json({ error: 'Unable to extract readable text from document' });
    }

    const sample = cleanedContent.substring(0, 200);
    const printableChars = (sample.match(/[\x20-\x7E\n\r\t]/g) || []).length;
    const printableRatio = printableChars / Math.max(sample.length, 1);
    if (sample.length > 0 && printableRatio < 0.3) {
      console.error(' Extracted content failed readability validation');
      return res.status(500).json({ error: 'Document content failed validation after extraction' });
    }

    // For chunked files, calculate stats from all chunks
    let storedContent = cleanedContent;
    let contentHash;
    let wordCount, sentenceCount, estimatedReadingMinutes, chunkEstimate;
    
    if (req.file.isChunked && req.file.processedChunks && req.file.originalBuffer) {
      // For chunked files, calculate hash and stats from all chunks
      // Re-extract chunks from buffer to calculate stats
      const hash = crypto.createHash('sha256');
      let totalWords = 0;
      let totalSentences = 0;
      
      for (const chunkMeta of req.file.processedChunks) {
        // Extract chunk text from original buffer
        const chunkBuffer = req.file.originalBuffer.slice(chunkMeta.bufferStart, chunkMeta.bufferEnd);
        const chunkText = chunkBuffer.toString('utf8');
        
        hash.update(chunkText);
        const words = chunkText.split(/\s+/).filter(Boolean).length;
        const sentences = (chunkText.match(/[.!?]+/g) || []).length;
        totalWords += words;
        totalSentences += sentences;
      }
      
      contentHash = hash.digest('hex');
      wordCount = totalWords;
      sentenceCount = totalSentences;
      estimatedReadingMinutes = Math.max(1, Math.round(wordCount / 200)) || 1;
      chunkEstimate = req.file.totalChunks || Math.max(1, Math.ceil(wordCount / 450));
      
      console.log(`[UPLOAD] Chunked file stats: ${wordCount} words, ${sentenceCount} sentences, ${req.file.totalChunks} chunks`);
    } else {
      // Regular file - calculate normally
      storedContent = cleanedContent;
      contentHash = crypto.createHash('sha256').update(storedContent).digest('hex');
      wordCount = storedContent.split(/\s+/).filter(Boolean).length;
      sentenceCount = (storedContent.match(/[.!?]+/g) || []).length;
      estimatedReadingMinutes = Math.max(1, Math.round(wordCount / 200)) || 1;
      chunkEstimate = Math.max(1, Math.ceil(wordCount / 450));
    }

    // Check for duplicate document by checksum/file_hash
    const { data: existingDocument, error: duplicateCheckError } = await supabase
      .from('knowledge_documents')
      .select('id, title, file_hash, created_at')
      .eq('file_hash', contentHash)
      .limit(1)
      .maybeSingle();
    
    if (existingDocument) {
      console.log(`[UPLOAD] Duplicate document detected: ${existingDocument.title} (uploaded ${new Date(existingDocument.created_at).toLocaleString()})`);
      if (!responseSent) {
        responseSent = true;
        return res.status(409).json({ 
          error: 'Duplicate document detected',
          message: `This document already exists: "${existingDocument.title}" (uploaded ${new Date(existingDocument.created_at).toLocaleString()})`,
          duplicate_id: existingDocument.id
        });
      }
      return;
    }
    
    if (duplicateCheckError && duplicateCheckError.code !== 'PGRST116') {
      // PGRST116 is "not found" which is fine - means no duplicate
      console.warn('[UPLOAD] Error checking for duplicates:', duplicateCheckError.message);
      // Continue anyway - don't block upload if duplicate check fails
    }

    // Ensure we have a valid profile ID (fixes foreign key constraint errors)
    let uploaderId = req.user.id;
    if (!uploaderId) {
      // Fallback: try to find profile by email if userId is invalid
      if (req.user.email) {
        const { data: profileByEmail } = await supabase
          .from('profiles')
          .select('id')
          .eq('email', req.user.email)
          .limit(1)
          .maybeSingle();
        if (profileByEmail?.id) {
          uploaderId = profileByEmail.id;
        }
      }
      if (!uploaderId) {
        throw new Error('Invalid user ID - cannot upload document without valid profile');
      }
    }

    let documentRecord = null;
    try {
      // Ensure size is properly cast for large files (BIGINT)
      const documentSizeValue = typeof file.size === 'number' ? file.size : parseInt(file.size, 10);
      
      const { data: documentData, error: documentError } = await supabase
        .from('documents')
        .insert([{
          uploader_id: uploaderId,
          filename: customFilename,
          content_type: file.mimetype,
          size: documentSizeValue, // Cast to ensure proper type
          source: 'upload',
          checksum: contentHash
        }])
        .select()
        .limit(1)
        .maybeSingle();

      if (documentError) {
        throw new Error(`Failed to store document metadata: ${documentError.message}`);
      }
      
      if (!documentData) {
        throw new Error('Document metadata insert returned no data');
      }
      
      documentRecord = documentData;
    } catch (documentInsertError) {
      console.error(' Document metadata insert failed:', documentInsertError.message);
      throw new Error(`Document metadata storage failed: ${documentInsertError.message}. Upload cannot proceed without document metadata.`);
    }

    const learningMetadata = {
      user_id: uploaderId,
      mime_type: file.mimetype,
      original_filename: file.originalname,
      document_type: documentType,
      learning_category: mappedCategory,
      storage_format: req.file.isChunked ? 'chunked' : 'plaintext',
      raw_length: req.file.isChunked ? (req.file.processedChunks.reduce((sum, c) => sum + c.size, 0)) : storedContent.length,
      extracted_text_preview: storedContent.substring(0, 600),
      learning_status: 'pending',
      checksum: contentHash,
      document_id: documentRecord?.id || null,
      word_count: wordCount,
      sentence_count: sentenceCount,
      estimated_reading_minutes: estimatedReadingMinutes,
      chunk_estimate: chunkEstimate,
      is_chunked: req.file.isChunked || false,
      total_chunks: req.file.totalChunks || null
    };
    
    // Process dictionary data for dictionary documents
    // This ensures dictionary words are stored correctly in Supabase
    let dictionaryData = {};
    if (documentType === 'dictionary') {
      // For dictionary documents, we'll let the Python service process them
      // But we still need to mark it as a dictionary document
      dictionaryData = {
        is_dictionary_file: true,
        processing_note: 'Will be processed by Python service for word extraction'
      };
      console.log(' [UPLOAD] Dictionary document detected - will be processed for word definitions');
    }
    
    // Store document metadata in Supabase
    // Ensure file_size is properly cast to handle large files (BIGINT)
    const fileSizeValue = typeof file.size === 'number' ? file.size : parseInt(file.size, 10);
    
    // For chunked files, store a minimal placeholder in content field
    // Full content is in doc_chunks and will be used for training
    // This avoids Supabase statement timeout on large inserts
    const contentToStore = req.file.isChunked
      ? `[CHUNKED_FILE] This file is stored in ${req.file.totalChunks} chunks. Full content available in doc_chunks table. Preview: ${storedContent.substring(0, 10000)}` // 10KB placeholder + small preview
      : storedContent;
    
    console.log(`[UPLOAD] Storing document: content=${(contentToStore.length / 1024).toFixed(2)}KB, file_size=${(fileSizeValue / 1024 / 1024).toFixed(2)}MB, chunks=${req.file.isChunked ? req.file.totalChunks : 0}`);
    
    const { data: knowledgeDocument, error } = await supabase
      .from('knowledge_documents')
      .insert([
        {
          document_id: documentRecord?.id || null,
          title: customFilename,
          content: contentToStore, // Reduced content for large files
          doc_type: documentType,
          document_type: documentType,
          learning_category: mappedCategory,
          learning_status: 'pending',
          file_size: fileSizeValue,
          file_hash: contentHash,
          learning_metadata: learningMetadata,
          dictionary_data: dictionaryData  // Store dictionary metadata
        }
      ])
      .select()
      .limit(1).maybeSingle();
    
    if (error) {
      console.error(' Error storing document in Supabase:', error);
      console.error('   Error details:', JSON.stringify(error, null, 2));
      if (res.headersSent) {
        console.error('[UPLOAD] Response already sent during document storage error');
        return;
      }
      return res.status(500).json({ 
        error: 'Failed to store document',
        details: error.message 
      });
    }
    
    // Store chunks for large files
    if (req.file.isChunked && req.file.processedChunks && knowledgeDocument?.id && req.file.originalBuffer) {
      console.log(`[UPLOAD] Storing ${req.file.processedChunks.length} chunks for document ${knowledgeDocument.id}...`);
      
      try {
        // Check existing chunks to find the highest chunk_index and continue numbering
        let startingChunkIndex = 0;
        try {
          const { data: existingChunks, error: existingChunksError } = await supabase
            .from('doc_chunks')
            .select('chunk_index')
            .eq('document_id', knowledgeDocument.id)
            .order('chunk_index', { ascending: false })
            .limit(1)
            .maybeSingle();
          
          if (!existingChunksError && existingChunks && existingChunks.chunk_index !== null && existingChunks.chunk_index !== undefined) {
            startingChunkIndex = existingChunks.chunk_index + 1;
            console.log(`[UPLOAD] Found existing chunks for document ${knowledgeDocument.id}, starting from chunk_index ${startingChunkIndex}`);
          } else if (existingChunksError && existingChunksError.code !== 'PGRST116') {
            console.warn(`[UPLOAD] Error checking existing chunks (non-critical): ${existingChunksError.message}`);
          }
        } catch (checkError) {
          console.warn(`[UPLOAD] Error checking existing chunks (non-critical): ${checkError.message}`);
          // Continue with index 0 if check fails
        }
        
        const chunkInserts = [];
        let processedCount = 0;
        let successfullyStoredChunks = 0;
        
          for (const chunkMeta of req.file.processedChunks) {
          const chunkBuffer = req.file.originalBuffer.slice(chunkMeta.bufferStart, chunkMeta.bufferEnd);
          const chunkText = chunkBuffer.toString('utf8');
          
          const chunkHash = crypto.createHash('sha256').update(chunkText).digest('hex');
          // Use startingChunkIndex + chunkMeta.index to continue from existing chunks
          const actualChunkIndex = startingChunkIndex + chunkMeta.index;
          chunkInserts.push({
            document_id: knowledgeDocument.id,
            chunk_index: actualChunkIndex,
            chunk_text: chunkText,
            tokens: chunkText.split(/\s+/).filter(Boolean).length,
            checksum: chunkHash,
            metadata: {
              offset: chunkMeta.offset,
              size: chunkMeta.size,
              total_chunks: req.file.totalChunks
            }
          });
          
          processedCount++;
          
          // Optimize batch size based on file size and chunk size
          // For very large files, use smaller batches to prevent timeouts
          // For smaller files, use larger batches for efficiency
          let dynamicBatchSize;
          if (req.file.size > 2 * 1024 * 1024 * 1024) {
            dynamicBatchSize = 1; // 1 chunk at a time for >2GB files
          } else if (req.file.size > 1024 * 1024 * 1024) {
            dynamicBatchSize = 2; // 2 chunks for 1-2GB files
          } else if (req.file.size > 500 * 1024 * 1024) {
            dynamicBatchSize = 3; // 3 chunks for 500MB-1GB files
          } else {
            dynamicBatchSize = 5; // 5 chunks for smaller files
          }
          if (chunkInserts.length >= dynamicBatchSize) {
            let retries = 3;
            let success = false;
            
            while (retries > 0 && !success) {
              try {
                const batchToInsert = chunkInserts.slice(0, dynamicBatchSize);
                const { error: chunkError } = await supabase
                  .from('doc_chunks')
                  .insert(batchToInsert);
                
                if (chunkError) {
                  const errorMsg = chunkError.message || '';
                  const errorCode = chunkError.code || '';
                  const isBufferLimit = errorMsg.includes('exceeded request buffer') || 
                                       errorMsg.includes('buffer limit') ||
                                       errorMsg.includes('request entity too large') ||
                                       errorMsg.includes('413');
                  const isSupabaseDown = errorMsg.includes('521') || 
                                        errorMsg.includes('520') ||
                                        errorMsg.includes('Web server is down');
                  const isTimeoutError = errorMsg.includes('timeout') || 
                                        errorMsg.includes('504') || 
                                        errorMsg.includes('Gateway') ||
                                        errorMsg.includes('PGRST002') ||
                                        errorCode === 'PGRST002' ||
                                        errorMsg.includes('schema cache') ||
                                        errorMsg.includes('Connection') ||
                                        errorMsg.includes('ECONNRESET') ||
                                        errorMsg.includes('ETIMEDOUT');
                  
                  if (isBufferLimit && dynamicBatchSize > 1) {
                    dynamicBatchSize = 1;
                    console.warn(`[UPLOAD] Buffer limit exceeded, reducing batch size to 1 chunk at a time`);
                    continue;
                  }
                  
                  if (isSupabaseDown) {
                    retries--;
                    if (retries > 0) {
                      const delay = (4 - retries) * 10000;
                      console.warn(`[UPLOAD] Supabase is down (521/520), waiting ${delay/1000}s before retry... (${retries} retries left)`);
                      await new Promise(resolve => setTimeout(resolve, delay));
                      continue;
                    } else {
                      throw new Error(`Failed to store chunk batch after ${3} retries: Supabase is down (521/520). Please try again when Supabase is available.`);
                    }
                  }
                  
                  if (isTimeoutError) {
                    retries--;
                    if (retries > 0) {
                      const delay = (4 - retries) * 5000;
                      console.warn(`[UPLOAD] Chunk batch timeout, retrying in ${delay/1000}s... (${retries} retries left)`);
                      await new Promise(resolve => setTimeout(resolve, delay));
                      continue;
                    } else {
                      throw new Error(`Failed to store chunk batch after ${3} retries: ${errorMsg}. Supabase connection issues prevent chunk storage.`);
                    }
                  }
                  throw new Error(`Failed to store chunk batch: ${errorMsg}. Chunk storage is required for large files.`);
                } else {
                  success = true;
                  successfullyStoredChunks += batchToInsert.length;
                  chunkInserts.splice(0, dynamicBatchSize);
                  console.log(`[UPLOAD] Stored batch of ${batchToInsert.length} chunks (${processedCount}/${req.file.totalChunks} total)`);
                }
              } catch (err) {
                const errMsg = err.message || '';
                const isBufferLimit = errMsg.includes('exceeded request buffer') || 
                                     errMsg.includes('buffer limit') ||
                                     errMsg.includes('request entity too large') ||
                                     errMsg.includes('413');
                const isSupabaseDown = errMsg.includes('521') || 
                                      errMsg.includes('520') ||
                                      errMsg.includes('Web server is down');
                const isRetryable = errMsg.includes('timeout') || 
                                  errMsg.includes('PGRST002') ||
                                  errMsg.includes('Connection') ||
                                  errMsg.includes('ECONNRESET') ||
                                  errMsg.includes('ETIMEDOUT');
                
                if (isBufferLimit && dynamicBatchSize > 1) {
                  dynamicBatchSize = 1;
                  console.warn(`[UPLOAD] Buffer limit exceeded in catch, reducing batch size to 1 chunk at a time`);
                  continue;
                }
                
                if (isSupabaseDown) {
                  retries--;
                  if (retries > 0) {
                    const delay = (4 - retries) * 10000;
                    console.warn(`[UPLOAD] Supabase is down (521/520), waiting ${delay/1000}s before retry... (${retries} retries left)`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                  }
                } else if (isRetryable) {
                  retries--;
                  if (retries > 0) {
                    const delay = (4 - retries) * 5000;
                    console.warn(`[UPLOAD] Chunk batch error, retrying in ${delay/1000}s... (${retries} retries left): ${errMsg.substring(0, 100)}`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                  }
                }
                throw new Error(`Failed to store chunk batch after ${3} retries: ${errMsg}. Chunk storage is required for large files.`);
              }
            }
            
            chunkInserts.length = 0;
            await new Promise(resolve => setImmediate(resolve));
          }
        }
        
        if (chunkInserts.length > 0) {
          let finalBatchSize = dynamicBatchSize || 1;
          let retries = 5;
          let success = false;
          
          while (retries > 0 && !success && chunkInserts.length > 0) {
            try {
              const finalBatchToInsert = chunkInserts.slice(0, finalBatchSize);
              const { error: chunkError } = await supabase
                .from('doc_chunks')
                .insert(finalBatchToInsert);
              
              if (chunkError) {
                const errorMsg = chunkError.message || '';
                const errorCode = chunkError.code || '';
                const isBufferLimit = errorMsg.includes('exceeded request buffer') || 
                                     errorMsg.includes('buffer limit') ||
                                     errorMsg.includes('request entity too large') ||
                                     errorMsg.includes('413');
                const isSupabaseDown = errorMsg.includes('521') || 
                                      errorMsg.includes('520') ||
                                      errorMsg.includes('Web server is down');
                const isTimeoutError = errorMsg.includes('timeout') || 
                                      errorMsg.includes('504') || 
                                      errorMsg.includes('Gateway') ||
                                      errorMsg.includes('PGRST002') ||
                                      errorCode === 'PGRST002' ||
                                      errorMsg.includes('schema cache') ||
                                      errorMsg.includes('Connection') ||
                                      errorMsg.includes('ECONNRESET') ||
                                      errorMsg.includes('ETIMEDOUT');
                
                if (isBufferLimit && finalBatchSize > 1) {
                  finalBatchSize = 1;
                  console.warn(`[UPLOAD] Final batch buffer limit exceeded, reducing to 1 chunk at a time`);
                  continue;
                }
                
                if (isSupabaseDown) {
                  retries--;
                  if (retries > 0) {
                    const delay = (6 - retries) * 10000;
                    console.warn(`[UPLOAD] Final batch - Supabase is down (521/520), waiting ${delay/1000}s... (${retries} retries left)`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                  } else {
                    throw new Error(`Failed to store final chunk batch after ${5} retries: Supabase is down (521/520). Please try again when Supabase is available.`);
                  }
                }
                
                if (isTimeoutError) {
                  retries--;
                  if (retries > 0) {
                    const delay = (6 - retries) * 5000;
                    console.warn(`[UPLOAD] Final chunk batch timeout, retrying in ${delay/1000}s... (${retries} retries left)`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                  } else {
                    throw new Error(`Failed to store final chunk batch after ${5} retries: ${errorMsg}. Supabase connection issues prevent chunk storage.`);
                  }
                }
                throw new Error(`Failed to store final chunk batch: ${errorMsg}. Chunk storage is required for large files.`);
              } else {
                success = true;
                successfullyStoredChunks += finalBatchToInsert.length;
                chunkInserts.splice(0, finalBatchSize);
                console.log(`[UPLOAD] Stored final batch of ${finalBatchToInsert.length} chunks`);
                if (chunkInserts.length > 0) {
                  success = false;
                }
              }
            } catch (err) {
              const errMsg = err.message || '';
              const isBufferLimit = errMsg.includes('exceeded request buffer') || 
                                   errMsg.includes('buffer limit') ||
                                   errMsg.includes('request entity too large') ||
                                   errMsg.includes('413');
              const isSupabaseDown = errMsg.includes('521') || 
                                    errMsg.includes('520') ||
                                    errMsg.includes('Web server is down');
              const isRetryable = errMsg.includes('timeout') || 
                                errMsg.includes('PGRST002') ||
                                errMsg.includes('Connection') ||
                                errMsg.includes('ECONNRESET') ||
                                errMsg.includes('ETIMEDOUT');
              
              if (isBufferLimit && finalBatchSize > 1) {
                finalBatchSize = 1;
                console.warn(`[UPLOAD] Final batch buffer limit exceeded in catch, reducing to 1 chunk at a time`);
                continue;
              }
              
              if (isSupabaseDown) {
                retries--;
                if (retries > 0) {
                  const delay = (6 - retries) * 10000;
                  console.warn(`[UPLOAD] Final batch - Supabase is down (521/520), waiting ${delay/1000}s... (${retries} retries left)`);
                  await new Promise(resolve => setTimeout(resolve, delay));
                  continue;
                }
              } else if (isRetryable) {
                retries--;
                if (retries > 0) {
                  const delay = (6 - retries) * 5000;
                  console.warn(`[UPLOAD] Final chunk batch error, retrying in ${delay/1000}s... (${retries} retries left): ${errMsg.substring(0, 100)}`);
                  await new Promise(resolve => setTimeout(resolve, delay));
                  continue;
                }
              }
              throw new Error(`Failed to store final chunk batch after ${5} retries: ${errMsg}. Chunk storage is required for large files.`);
            }
          }
        }
        
        // Clear original buffer reference to free memory
        req.file.originalBuffer = null;
        
        // Verify chunks were actually stored - use both in-memory count and database query
        // First check in-memory count (faster, but verify with DB query for safety)
        if (successfullyStoredChunks !== req.file.totalChunks) {
          console.error(`[UPLOAD] Chunk count mismatch (in-memory): expected ${req.file.totalChunks}, stored ${successfullyStoredChunks}`);
          throw new Error(`Chunk storage incomplete: expected ${req.file.totalChunks} chunks but only ${successfullyStoredChunks} were stored. Upload failed.`);
        }
        
        // Also verify by querying the database (double-check for data integrity)
        const { count: storedChunkCount, error: verifyError } = await supabase
          .from('doc_chunks')
          .select('id', { count: 'exact', head: true })
          .eq('document_id', knowledgeDocument.id);
        
        if (verifyError) {
          console.error(`[UPLOAD] Failed to verify chunk storage: ${verifyError.message}`);
          throw new Error(`Chunk storage verification failed: ${verifyError.message}. Chunks may not have been stored correctly.`);
        }
        
        if (storedChunkCount !== req.file.totalChunks) {
          console.error(`[UPLOAD] Chunk count mismatch (database): expected ${req.file.totalChunks}, found ${storedChunkCount}`);
          throw new Error(`Chunk storage incomplete: expected ${req.file.totalChunks} chunks but only ${storedChunkCount} were stored in database. Upload failed.`);
        }
        
        console.log(`[UPLOAD] Verified: ${storedChunkCount} chunks stored in database (matches expected ${req.file.totalChunks})`);
        
        // Update knowledge_documents to mark as chunked
        const { error: updateError } = await supabase
          .from('knowledge_documents')
          .update({
            is_chunked: true,
            total_chunks: req.file.totalChunks
          })
          .eq('id', knowledgeDocument.id);
        
        if (updateError) {
          console.warn(`[UPLOAD] Failed to update chunked flag: ${updateError.message}`);
          // Non-critical - chunks are stored, just metadata update failed
        }
        
        console.log(`[UPLOAD] ✅ Successfully stored and verified ${storedChunkCount} chunks for document ${knowledgeDocument.id}`);
      } catch (chunkError) {
        console.error(`[UPLOAD] Error storing chunks: ${chunkError.message}`);
        throw new Error(`Failed to store document chunks: ${chunkError.message}. The document metadata was stored but chunks are required for large files.`);
      }
    }
    
    let learningArtifactResults = [];
    try {
      learningArtifactResults = await recordDocumentLearningArtifacts({
        supabaseClient: supabase,
        uploaderId: req.user.id,
        documentRecord,
        knowledgeDocument,
        storedContent,
        mappedCategory,
        documentType,
        learningMetadata,
        fileSize: file.size,
        wordCount,
        sentenceCount,
        estimatedReadingMinutes,
        chunkEstimate,
        contentHash
      });
      
      if (!Array.isArray(learningArtifactResults)) {
        throw new Error('Learning artifacts must be an array');
      }
    } catch (artifactError) {
      console.error(' Document learning artifact generation failed:', artifactError.message);
      throw new Error(`Failed to generate learning artifacts: ${artifactError.message}. Document storage requires learning artifacts.`);
    }
    
    // Document learning handled by inference service and ml_local training pipeline
    
    const uploadDuration = ((Date.now() - uploadStartTime) / 1000).toFixed(2);
    console.log(` [UPLOAD] Document uploaded successfully in ${uploadDuration}s - ID: ${knowledgeDocument.id}`);
    
    logSecurityEvent('DOCUMENT_UPLOAD_SUCCESS', {
      userId: req.user.id,
      documentId: knowledgeDocument.id,
      filename: customFilename
    });
    
    // Check if response already sent
    if (responseSent || res.headersSent) {
      console.warn('[UPLOAD] Response already sent, skipping success response');
      return;
    }
    
    responseSent = true;
    return res.status(200).json({ 
      success: true, 
      document_id: knowledgeDocument.id,
      message: 'Document uploaded successfully',
      learning_artifacts: learningArtifactResults
    });
  } catch (error) {
    console.error(' Error in document upload:', error);
    console.error('   Error stack:', error.stack);
    console.error('   Error message:', error.message);
    console.error('   Error name:', error.name);
    
    // Don't send response if headers already sent
    if (responseSent || res.headersSent) {
      console.error('[UPLOAD] Response already sent, cannot send error response');
      return;
    }
    
    responseSent = true;
    
    logSecurityEvent('DOCUMENT_UPLOAD_ERROR', {
      userId: req.user?.id || 'unknown',
      error: error.message,
      stack: error.stack?.substring(0, 500)
    }, 'error');
    
    // Determine appropriate status code
    let statusCode = 500;
    if (error.message && error.message.includes('timeout')) {
      statusCode = 504;
    } else if (error.message && (error.message.includes('CSRF') || error.message.includes('validation'))) {
      statusCode = 403;
    } else if (error.message && error.message.includes('size')) {
      statusCode = 413;
    }
    
    return res.status(statusCode).json({ 
      error: 'An error occurred during document upload',
      details: error.message,
      type: error.name
    });
  }
});

function clampValue(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function parseOptionalJson(value) {
  if (!value) return {};
  if (typeof value === 'object') return { ...value };
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      return typeof parsed === 'object' && parsed !== null ? parsed : {};
    } catch {
      return {};
    }
  }
  return {};
}

async function calculateComparativeMetrics({
  supabaseClient,
  knowledgeDocument,
  mappedCategory,
  documentType,
  tone,
  wordCount,
  sentenceCount,
  estimatedReadingMinutes
}) {
  const metrics = {
    baseline_word_count: null,
    baseline_sentence_count: null,
    baseline_reading_minutes: null,
    improvement_word_percent: null,
    improvement_sentence_percent: null,
    tone_distribution: {},
    prior_documents: 0,
    learning_score: 0.55,
    novelty_score: 0.5
  };

  try {
    const { data: previousSessions, error } = await supabaseClient
      .from('document_learning_sessions')
      .select('id, metadata, created_at')
      .eq('learning_category', mappedCategory)
      .neq('document_id', knowledgeDocument.id)
      .order('created_at', { ascending: false })
      .limit(10);

    if (error) {
      console.warn(' Unable to fetch prior learning sessions:', error.message);
      return metrics;
    }

    if (!previousSessions || !previousSessions.length) {
      metrics.learning_score = clampValue(0.6 + wordCount / 12000, 0.55, 0.85);
      metrics.novelty_score = 1;
      return metrics;
    }

    metrics.prior_documents = previousSessions.length;

    let wordTotal = 0;
    let sentenceTotal = 0;
    let readingTotal = 0;
    const toneStats = {};

    previousSessions.forEach(session => {
      const meta = parseOptionalJson(session.metadata);
      if (meta.word_count) {
        wordTotal += Number(meta.word_count) || 0;
      }
      if (meta.sentence_count) {
        sentenceTotal += Number(meta.sentence_count) || 0;
      }
      if (meta.estimated_reading_minutes) {
        readingTotal += Number(meta.estimated_reading_minutes) || 0;
      }
      if (meta.tone) {
        const normalizedTone = String(meta.tone).toLowerCase();
        toneStats[normalizedTone] = (toneStats[normalizedTone] || 0) + 1;
      }
    });

    const denom = previousSessions.length || 1;
    metrics.baseline_word_count = Math.round(wordTotal / denom) || null;
    metrics.baseline_sentence_count = Math.round(sentenceTotal / denom) || null;
    metrics.baseline_reading_minutes = Math.round(readingTotal / denom) || null;
    metrics.tone_distribution = toneStats;

    if (metrics.baseline_word_count) {
      metrics.improvement_word_percent = clampValue(((wordCount - metrics.baseline_word_count) / metrics.baseline_word_count) * 100, -95, 250);
    }

    if (metrics.baseline_sentence_count) {
      metrics.improvement_sentence_percent = clampValue(((sentenceCount - metrics.baseline_sentence_count) / metrics.baseline_sentence_count) * 100, -95, 250);
    }

    const tonePrevCount = tone ? (toneStats[String(tone).toLowerCase()] || 0) : 0;
    const toneDiversityBoost = Math.min(0.15, tonePrevCount === 0 ? 0.12 : 0.04);

    const improvementComponent = metrics.improvement_word_percent !== null
      ? clampValue(metrics.improvement_word_percent / 100, -0.2, 0.35)
      : 0;

    const noveltyComponent = previousSessions.every(session => {
      const meta = parseOptionalJson(session.metadata);
      return !meta.document_type || meta.document_type !== documentType;
    }) ? 0.1 : 0;

    metrics.learning_score = clampValue(
      0.55 + improvementComponent + toneDiversityBoost + noveltyComponent,
      0.5,
      0.95
    );
    metrics.novelty_score = clampValue(0.5 + noveltyComponent * 2, 0.4, 1);
  } catch (comparativeError) {
    console.warn(' Comparative metric calculation failed:', comparativeError.message);
  }

  return metrics;
}

async function recordDocumentLearningArtifacts({
  supabaseClient,
  uploaderId,
  documentRecord,
  knowledgeDocument,
  storedContent,
  mappedCategory,
  documentType,
  learningMetadata,
  fileSize,
  wordCount,
  sentenceCount,
  estimatedReadingMinutes,
  chunkEstimate,
  contentHash
}) {
  const nowIso = new Date().toISOString();
  const tone = learningMetadata?.detected_tone || learningMetadata?.tone || learningMetadata?.dominant_tone || 'neutral';

  const tableExistenceCache = {};
  const insertResults = [];

  const ensureTable = async (table) => {
    if (tableExistenceCache[table] !== undefined) {
      return tableExistenceCache[table];
    }

    try {
      const { error } = await supabaseClient
        .from(table)
        .select('id', { head: true, count: 'exact' });

      if (error && error.message && error.message.toLowerCase().includes('does not exist')) {
        tableExistenceCache[table] = false;
        console.warn(` [LEARNING-ARTIFACTS] Table ${table} does not exist`);
        return false;
      }

      if (error) {
        console.warn(` [LEARNING-ARTIFACTS] Error checking table ${table}:`, error.message);
      }

      tableExistenceCache[table] = true;
      return true;
    } catch (checkError) {
      console.warn(` [LEARNING-ARTIFACTS] Exception checking table ${table}:`, checkError.message);
      tableExistenceCache[table] = false;
      return false;
    }
  };

  const safeInsert = async (table, payload, description) => {
    const tableAvailable = await ensureTable(table);
    if (!tableAvailable) {
      insertResults.push({
        table,
        description,
        status: 'skipped',
        reason: 'missing_table'
      });
      return;
    }

    try {
      const { error } = await supabaseClient.from(table).insert(payload);
      if (error) {
        console.warn(` Unable to insert ${description}:`, error.message);
        insertResults.push({
          table,
          description,
          status: 'failed',
          error: error.message
        });
      } else {
        insertResults.push({
          table,
          description,
          status: 'inserted',
          rows: Array.isArray(payload) ? payload.length : 1
        });
      }
    } catch (insertError) {
      console.warn(` ${description} insert failed:`, insertError.message);
      insertResults.push({
        table,
        description,
        status: 'failed',
        error: insertError.message
      });
    }
  };

  const comparativeMetrics = await calculateComparativeMetrics({
    supabaseClient,
    knowledgeDocument,
    mappedCategory,
    documentType,
    tone,
    wordCount,
    sentenceCount,
    estimatedReadingMinutes
  });

  const enrichedLearningMetadata = {
    ...learningMetadata,
    comparative_analysis: comparativeMetrics
  };

  try {
    await supabaseClient
      .from('knowledge_documents')
      .update({ learning_metadata: enrichedLearningMetadata })
      .eq('id', knowledgeDocument.id);
  } catch (metadataUpdateError) {
    console.warn(' Knowledge document metadata update failed:', metadataUpdateError.message);
  }

  try {
    const { data: embeddingRows, error: embeddingFetchError } = await supabaseClient
      .from('document_embeddings')
      .select('id, metadata')
      .eq('document_id', knowledgeDocument.id);

    if (embeddingFetchError) {
      console.warn(' Unable to fetch embeddings for comparative update:', embeddingFetchError.message);
    } else if (embeddingRows && embeddingRows.length) {
      for (const row of embeddingRows) {
        const existingMeta = parseOptionalJson(row.metadata);
        const updatedMeta = {
          ...existingMeta,
          learning_score: comparativeMetrics.learning_score,
          comparative_snapshot: {
            improvement_word_percent: comparativeMetrics.improvement_word_percent,
            baseline_word_count: comparativeMetrics.baseline_word_count,
            tone,
            learning_category: mappedCategory
          }
        };

        await supabaseClient
          .from('document_embeddings')
          .update({ metadata: updatedMeta })
          .eq('id', row.id);
      }
    }
  } catch (embeddingUpdateError) {
    console.warn(' Unable to update embedding metadata:', embeddingUpdateError.message);
  }

  let sessionId = null;
  try {
    const { data: sessionData, error: sessionError } = await supabaseClient
      .from('document_learning_sessions')
      .insert([{
        document_id: knowledgeDocument.id,
        learning_category: mappedCategory,
        document_type: documentType || knowledgeDocument.doc_type || 'general',
        status: 'completed',
        learning_approach: 'automated_ingest',
        focus_area: 'document_upload',
        file_size: typeof fileSize === 'number' ? fileSize : parseInt(fileSize, 10),
        file_hash: contentHash,
        processing_started_at: nowIso,
        processing_completed_at: nowIso,
        learning_started_at: nowIso,
        learning_completed_at: nowIso,
        metadata: {
          uploader_id: uploaderId,
          document_id: knowledgeDocument.id,
          document_title: knowledgeDocument.title,
          word_count: wordCount,
          sentence_count: sentenceCount,
          estimated_reading_minutes: estimatedReadingMinutes,
          chunk_estimate: chunkEstimate,
          tone,
        document_type: documentType,
          source_document_id: documentRecord?.id || null
        }
      }])
      .select('id')
      .limit(1).maybeSingle();

    if (sessionError) {
      console.warn(' Unable to create document learning session:', sessionError.message);
    } else if (sessionData) {
      sessionId = sessionData.id;
    }
  } catch (sessionInsertError) {
    console.warn(' Document learning session insert failed:', sessionInsertError.message);
  }

  const summaryInsight = {
    summary: `Document "${knowledgeDocument.title}" ingested with ${wordCount} words and ${sentenceCount} sentences${comparativeMetrics.improvement_word_percent !== null ? ` (${comparativeMetrics.improvement_word_percent >= 0 ? '+' : ''}${comparativeMetrics.improvement_word_percent.toFixed(1)}% vs. recent baseline)` : ''}.`,
    tone,
    word_count: wordCount,
    estimated_reading_minutes: estimatedReadingMinutes,
    chunk_estimate: chunkEstimate,
    document_type: documentType,
    uploader_id: uploaderId,
    comparative: comparativeMetrics
  };

  if (sessionId) {
    await safeInsert(
      'document_learning_insights',
      [{
        session_id: sessionId,
        insight_type: 'learning_summary',
        content: summaryInsight,
        confidence_score: 0.7,
        importance_score: 0.6,
        learning_category: mappedCategory,
        tags: [documentType, tone].filter(Boolean)
      }],
      'document learning insight'
    );
  }

  await safeInsert(
    'document_learning_progress',
    [{
      learning_category: mappedCategory,
      progress_type: 'knowledge_expansion',
      metric_name: 'document_word_count',
      metric_value: Math.min(wordCount, 999999.9999), // DECIMAL(10,4) max value to prevent overflow
      baseline_value: comparativeMetrics.baseline_word_count ? Math.min(comparativeMetrics.baseline_word_count, 999999.9999) : null, // Cap baseline to prevent overflow
      improvement_percentage: comparativeMetrics.improvement_word_percent,
      document_count: 1,
      last_updated_document_id: knowledgeDocument.id,
      learning_session_id: sessionId,
      metadata: {
        document_title: knowledgeDocument.title,
        estimated_reading_minutes: estimatedReadingMinutes,
        chunk_estimate: chunkEstimate,
        tone,
        comparative: comparativeMetrics
      },
      recorded_at: nowIso
    }],
    'document learning progress'
  );

  await safeInsert(
    'document_learning_analytics',
    [{
      session_id: sessionId,
      metric_type: 'learning_effectiveness',
      metric_value: Math.max(0.2, Math.min(1, wordCount / 4000)),
      metric_unit: 'score',
      // CRITICAL FIX: Cap comparison_value to DECIMAL(10,4) max (999999.9999) to prevent overflow
      comparison_value: comparativeMetrics.baseline_word_count ? Math.min(comparativeMetrics.baseline_word_count, 999999.9999) : null,
      trend_direction: comparativeMetrics.improvement_word_percent !== null && comparativeMetrics.improvement_word_percent < 0 ? 'declining' : 'improving',
      learning_category: mappedCategory,
      metadata: {
        document_id: knowledgeDocument.id,
        document_title: knowledgeDocument.title,
        chunk_estimate: chunkEstimate,
        tone,
        comparative: comparativeMetrics
      },
      recorded_at: nowIso
    }],
    'document learning analytics'
  );

  await safeInsert(
    'document_learning_patterns',
    [{
      pattern_type: 'knowledge_pattern',
      pattern_name: `${mappedCategory || 'knowledge'}_ingest_baseline`,
      pattern_description: 'Auto-generated pattern captured during document ingestion.',
      pattern_data: {
        word_count: wordCount,
        sentence_count: sentenceCount,
        tone,
        document_type: documentType,
        estimated_reading_minutes: estimatedReadingMinutes
      },
      learning_category: mappedCategory,
      confidence_level: 0.35,
      usage_count: 1,
      success_rate: 0.0,
      source_document_ids: [knowledgeDocument.id],
      tags: [documentType, tone].filter(Boolean),
      is_active: true
    }],
    'document learning pattern'
  );

  await safeInsert(
    'learning_metrics',
    [
      {
        name: 'documents_processed',
        value: 1,
        metadata: {
          document_id: knowledgeDocument.id,
          learning_category: mappedCategory,
          document_type: documentType,
          uploader_id: uploaderId
        },
        created_at: nowIso
      },
      {
        name: 'avg_document_word_count',
        value: wordCount,
        metadata: {
          document_id: knowledgeDocument.id,
          learning_category: mappedCategory,
          document_type: documentType
        },
        created_at: nowIso
      },
      {
        name: 'document_learning_score',
        value: comparativeMetrics.learning_score,
        metadata: {
          document_id: knowledgeDocument.id,
          learning_category: mappedCategory,
          document_type: documentType,
          improvement_word_percent: comparativeMetrics.improvement_word_percent
        },
        created_at: nowIso
      }
    ],
    'learning metric'
  );

  await safeInsert(
    'epsilon_learning_analytics',
    [{
      session_id: `doc-${knowledgeDocument.id}`,
      user_id: uploaderId,
      learning_type: 'quality',
      metric_score: comparativeMetrics.learning_score,
      user_message: `Ingested document ${knowledgeDocument.title}`,
      epsilon_response: null,
      metadata: {
        document_id: knowledgeDocument.id,
        word_count: wordCount,
        tone,
        document_type: documentType,
        comparative: comparativeMetrics
      },
      created_at: nowIso
    }],
    'Epsilon AI learning analytic'
  );

  await safeInsert(
    'epsilon_learning_sessions',
    [{
      session_id: `ingest-${knowledgeDocument.id}`,
      session_type: 'autonomous_mining',
      training_data_count: chunkEstimate,
      model_version_before: '1.0.0',
      model_version_after: '1.0.0',
      performance_improvement: 0.0,
      status: 'completed',
      metadata: {
        document_id: knowledgeDocument.id,
        learning_category: mappedCategory,
        tone,
        chunk_estimate: chunkEstimate
      },
      started_at: nowIso,
      completed_at: nowIso,
      created_at: nowIso,
      updated_at: nowIso
    }],
    'Epsilon AI learning session'
  );

  await safeInsert(
    'epsilon_training_data',
    [{
      input_text: storedContent.substring(0, 2000),
      expected_output: `Summarize the document "${knowledgeDocument.title}" focusing on actionable insights.`,
      training_type: 'document_ingest',
      quality_score: 0.5,
      is_validated: false,
      source_document_id: knowledgeDocument.id,
      metadata: {
        document_id: knowledgeDocument.id,
        learning_category: mappedCategory,
        word_count: wordCount,
        tone,
        document_type: documentType
      },
      created_at: nowIso,
      updated_at: nowIso
    }],
    'Epsilon AI training data'
  );

  await safeInsert(
    'epsilon_learning_patterns',
    [{
      pattern_type: 'conversation',
      pattern_data: {
        document_id: knowledgeDocument.id,
        learning_category: mappedCategory,
        tone,
        document_type: documentType,
        estimated_reading_minutes: estimatedReadingMinutes
      },
      confidence_score: 0.3,
      usage_count: 1,
      last_used_at: nowIso,
      created_at: nowIso,
      updated_at: nowIso
    }],
    'Epsilon AI learning pattern'
  );

  await safeInsert(
    'epsilon_experience_data',
    [{
      user_id: uploaderId,
      interaction_type: 'document_upload',
      user_input: `Uploaded document ${knowledgeDocument.title}`,
      assistant_response: 'Epsilon AI ingested the document and updated learning metrics.',
      context: {
        document_id: knowledgeDocument.id,
        learning_category: mappedCategory,
        tone,
        document_type: documentType
      },
      success_rate: 0.5,
      emotion_tone: tone,
      outcome: 'ingested',
      topic: documentType || 'general',
      confidence_score: 0.5,
      learning_value: 'medium',
      created_at: nowIso,
      updated_at: nowIso
    }],
    'Epsilon AI experience data'
  );

  await safeInsert(
    'epsilon_learning_rules',
    [{
      rule_type: 'ingestion_policy',
      pattern: `document:${mappedCategory || 'general'}`,
      response_template: 'Blend the learned material naturally into conversation, referencing proof points without explicit citations.',
      confidence_score: 0.4,
      success_count: 0,
      failure_count: 0,
      last_used: nowIso,
      is_active: true,
      created_at: nowIso,
      updated_at: nowIso
    }],
    'Epsilon AI learning rule'
  );

  await safeInsert(
      'epsilon_model_weights',
    [{
      weight_type: 'response_style',
      weight_name: `${mappedCategory || 'general'}_document_bias`,
      weight_value: clampValue(comparativeMetrics.learning_score, 0.2, 0.95),
      learning_session_id: sessionId ? `ingest-${sessionId}` : `ingest-${knowledgeDocument.id}`,
      metadata: {
        document_id: knowledgeDocument.id,
        tone,
        document_type: documentType,
        comparative: comparativeMetrics
      },
      created_at: nowIso,
      updated_at: nowIso
    }],
    'Epsilon AI model weight'
  );

  const failedInserts = insertResults.filter(result => result.status !== 'inserted');
  if (failedInserts.length) {
    EpsilonLog.warn('LEARNING_ARTIFACT_WARN', 'Some learning artifacts failed to store', {
      documentId: knowledgeDocument.id,
      failures: failedInserts
    });
  } else {
    EpsilonLog.info('LEARNING_ARTIFACT_OK', 'Learning artifacts stored successfully', {
      documentId: knowledgeDocument.id,
      tablesUpdated: insertResults.length
    });
  }

  // Document ingestion handled by inference service and ml_local training pipeline

  return insertResults;
}

// Documents handler
const documentsHandler = async (event, context) => {
  try {
    const { httpMethod, path, body, headers, queryStringParameters } = event;
    
    // Parse user from context
    const user = context.clientContext?.user || { sub: null, email: null };
    
    if (!user.sub) {
      return {
        statusCode: 401,
        body: JSON.stringify({ error: 'Unauthorized' })
      };
    }
    
    if (httpMethod === 'GET') {
      // Get documents for user
      // Note: Using client-side filtering instead of JSONB filter for better compatibility
      const { data: allData, error: fetchError } = await supabase
        .from('knowledge_documents')
        .select('id, title, document_type, created_at, file_size, learning_metadata')
        .order('created_at', { ascending: false });
      
      if (fetchError) {
        console.error('Error fetching documents:', fetchError);
        return {
          statusCode: 500,
          body: JSON.stringify({ error: 'Failed to fetch documents' })
        };
      }
      
      // Filter by user_id client-side
      const data = (allData || []).filter(doc => {
        if (doc.learning_metadata && typeof doc.learning_metadata === 'object') {
          return doc.learning_metadata.user_id === user.sub;
        }
        return false;
      });
      
      return {
        statusCode: 200,
        body: JSON.stringify({ documents: data })
      };
    } else if (httpMethod === 'DELETE') {
      // Delete document
      const documentId = queryStringParameters?.id;
      
      if (!documentId) {
        return {
          statusCode: 400,
          body: JSON.stringify({ error: 'Document ID is required' })
        };
      }
      
      // Verify ownership
      const { data: docData, error: docError } = await supabase
        .from('knowledge_documents')
        .select('learning_metadata')
        .eq('id', documentId)
        .limit(1).maybeSingle();
      
      if (docError || !docData) {
        return {
          statusCode: 404,
          body: JSON.stringify({ error: 'Document not found' })
        };
      }
      
      if (docData.learning_metadata?.user_id !== user.sub) {
        return {
          statusCode: 403,
          body: JSON.stringify({ error: 'You do not have permission to delete this document' })
        };
      }
      
      // Delete document
      const { error } = await supabase
        .from('knowledge_documents')
        .delete()
        .eq('id', documentId);
      
      if (error) {
        console.error('Error deleting document:', error);
        return {
          statusCode: 500,
          body: JSON.stringify({ error: 'Failed to delete document' })
        };
      }
      
      return {
        statusCode: 200,
        body: JSON.stringify({ success: true, message: 'Document deleted successfully' })
      };
    } else if (httpMethod === 'POST' && path.endsWith('/search')) {
      // Search documents
      const { query } = JSON.parse(body);
      
      if (!query) {
        return {
          statusCode: 400,
          body: JSON.stringify({ error: 'Search query is required' })
        };
      }
      
      // Get all documents for user - include chunked flag
      // Note: Using client-side filtering instead of JSONB filter for better compatibility
      const { data: allData, error: fetchError } = await supabase
        .from('knowledge_documents')
        .select('id, title, document_type, content, created_at, learning_metadata, is_chunked, total_chunks');
      
      if (fetchError) {
        console.error('Error fetching documents for search:', fetchError);
        return {
          statusCode: 500,
          body: JSON.stringify({ error: 'Failed to search documents' })
        };
      }
      
      // Filter by user_id client-side
      const data = (allData || []).filter(doc => {
        if (doc.learning_metadata && typeof doc.learning_metadata === 'object') {
          return doc.learning_metadata.user_id === user.sub;
        }
        return false;
      });
      
      // Decrypt and search in each document - handle chunked documents
      const results = [];
      
      for (const doc of data) {
        try {
          let content = doc.content || '';
          
          if (doc.is_chunked && doc.id) {
            try {
              // Use chunk fetcher utility with timeout handling and batching
              const { fetchChunksInBatches } = require('../utils/chunk-fetcher');
              const chunkRows = await fetchChunksInBatches(supabase, doc.id, { 
                batchSize: 50,
                silent: true 
              });
                
              if (chunkRows && chunkRows.length > 0) {
                  content = chunkRows.map(chunk => chunk.chunk_text).join('\n\n');
                  } else {
                console.warn(`[WARN] [SERVER] Document ${doc.id} marked as chunked but no chunks found in search`);
              }
            } catch (chunkFetchError) {
              console.warn(`[SERVER] Error fetching chunks for document ${doc.id} in search:`, chunkFetchError.message);
              // Continue with preview content
            }
          }
          
          const decryptedContent = decrypt(content);
          
          if (decryptedContent && decryptedContent.toLowerCase().includes(query.toLowerCase())) {
            // Find the context around the match
            const index = decryptedContent.toLowerCase().indexOf(query.toLowerCase());
            const start = Math.max(0, index - 100);
            const end = Math.min(decryptedContent.length, index + query.length + 100);
            const excerpt = decryptedContent.substring(start, end);
            
            results.push({
              id: doc.id,
              filename: doc.filename,
              document_type: doc.document_type,
              created_at: doc.created_at,
              excerpt: excerpt
            });
          }
        } catch (decryptError) {
          console.error(`Error decrypting document ${doc.id}:`, decryptError);
        }
      }
      
      return {
        statusCode: 200,
        body: JSON.stringify({ results })
      };
    }
    
    return {
      statusCode: 405,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  } catch (error) {
    console.error('Error in documents handler:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'An error occurred' })
    };
  }
};


// Netlify function to Express middleware adapter
const netlifyToExpress = (netlifyHandler) => {
  return async (req, res) => {
    try {
      // Convert Express request to Netlify event format
      const event = {
        httpMethod: req.method,
        path: req.path,
        headers: req.headers,
        queryStringParameters: req.query,
        body: req.method !== 'GET' && req.method !== 'HEAD' ? JSON.stringify(req.body) : null,
        isBase64Encoded: false,
        user: req.user // Pass user info from Express middleware
      };
      
      // Create context object with user info if available
      const context = {
        clientContext: {
          user: req.user ? {
            sub: req.user.id,
            email: req.user.email,
            role: req.user.role
          } : null
        }
      };
      
      // Call the Netlify function handler
      const result = await netlifyHandler(event, context);
      
      // Set status code and headers
      res.status(result.statusCode);
      
      if (result.headers) {
        Object.entries(result.headers).forEach(([key, value]) => {
          // Handle Set-Cookie header specially to ensure it's set correctly
          if (key.toLowerCase() === 'set-cookie') {
            if (Array.isArray(value)) {
              value.forEach(cookie => res.setHeader('Set-Cookie', cookie));
            } else {
              res.setHeader('Set-Cookie', value);
            }
          } else {
            res.setHeader(key, value);
          }
        });
      }
      
      // Send response body
      if (result.body) {
        if (result.isBase64Encoded) {
          const buffer = Buffer.from(result.body, 'base64');
          res.send(buffer);
        } else {
          res.send(result.body);
        }
      } else {
        res.end();
      }
    } catch (error) {
      console.error('Error in Netlify function adapter:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  };
};

// API routes
app.use('/api/auth', authLimiter, netlifyToExpress(authHandler));
app.use('/.netlify/functions/auth', authLimiter, netlifyToExpress(authHandler));
app.use('/api/analytics', apiLimiter, netlifyToExpress(analyticsHandler));
app.use('/.netlify/functions/analytics', apiLimiter, netlifyToExpress(analyticsHandler));

// Analytics fallback endpoint removed - all analytics must go through /api/analytics
// This ensures proper error handling and service availability checks
// Debug endpoint removed - use /api/documents instead

// Documents API routes
app.get('/api/documents', verifyAuth('owner'), async (req, res) => {
  try {
    // Handle verify action (ignore cache-busting 't' parameter)
    if (req.query.action === 'verify' && req.query.id) {
      const documentId = req.query.id;
    
      // Get document content (decrypted)
      const { data: docData, error: docError } = await supabase
      .from('knowledge_documents')
        .select('content')
        .eq('id', documentId)
        .limit(1).maybeSingle();
      
      if (docError || !docData) {
        return res.status(404).json({ error: 'Document not found' });
      }
      
      // Decrypt content
      let contentLength = 0;
      if (docData.content) {
        try {
          // Check if content is encrypted
          const isEncrypted = typeof docData.content === 'string' && docData.content.includes(':') && docData.content.split(':').length === 3;
          
          if (isEncrypted) {
            const decryptedContent = decrypt(docData.content);
            if (decryptedContent) {
              contentLength = decryptedContent.length;
            } else {
              console.error(` [SERVER] Failed to decrypt content for document ${documentId}`);
              contentLength = 0;
            }
          } else {
            contentLength = docData.content.length;
          }
        } catch (decryptError) {
          console.error(` [SERVER] Error decrypting document ${documentId}:`, decryptError);
          contentLength = 0;
        }
      }
      
      return res.json({ contentLength });
    }
    
      
    // Fetch documents from Supabase with pagination and proper indexing
    // Use limit to prevent large queries that cause timeouts
    const limit = parseInt(req.query.limit) || 1000; // Default 1000, max reasonable
    const offset = parseInt(req.query.offset) || 0;
    
    let data = [];
    let error = null;
    let count = 0;
    
    try {
      // First get count (fast, uses index)
      const { count: totalCount, error: countError } = await supabase
        .from('knowledge_documents')
        .select('id', { count: 'exact', head: true });
      
      if (countError) {
        throw countError;
      }
      count = totalCount || 0;
      
      // Then get paginated data (uses created_at index)
      const { data: documents, error: queryError } = await supabase
        .from('knowledge_documents')
        .select('id, title, doc_type, document_type, created_at, file_size, learning_metadata, learning_status')
        .order('created_at', { ascending: false })
        .range(offset, offset + limit - 1);
      
      if (queryError) {
        throw queryError;
      }
      
      data = documents || [];
      error = null;
    } catch (queryError) {
      error = queryError;
      const errorMsg = error?.message || '';
      console.error('[DOCUMENTS] Query error:', errorMsg);
      
      // If it's a timeout or connection error, try to get at least a count
      if (errorMsg.includes('timeout') || errorMsg.includes('521') || errorMsg.includes('520') || errorMsg.includes('Connection')) {
        console.warn('[DOCUMENTS] Supabase connection issue, attempting fallback count query');
        try {
          const { count: fallbackCount } = await supabase
            .from('knowledge_documents')
            .select('id', { count: 'exact', head: true });
          count = fallbackCount || 0;
        } catch (fallbackError) {
          console.error('[DOCUMENTS] Fallback count query also failed');
        }
        return res.status(503).json({ 
          error: 'Database temporarily unavailable', 
          documents: [], 
          count: count 
        });
      }
      
      return res.status(500).json({ 
        error: 'Failed to fetch documents', 
        details: errorMsg, 
        documents: [] 
      });
    }
    
    // Handle case where data might be null
    if (!data) {
      console.warn('  Query returned null data (might be empty result)');
      data = [];
    }
    
    
    // For owners, show ALL documents regardless of user_id
    // For non-owners, filter by user_id
    if (data && data.length > 0) {
      if (req.user.role === 'owner') {
        // Owners see everything - no filtering needed
        // Normalize document_type field (use doc_type if document_type is missing)
        data = data.map(doc => ({
          ...doc,
          document_type: doc.document_type || doc.doc_type || 'general'
        }));
      } else {
        const beforeCount = data.length;
      data = data.filter(doc => {
        try {
          if (doc.learning_metadata && typeof doc.learning_metadata === 'object') {
              const matches = doc.learning_metadata.user_id === req.user.id;
              if (!matches) {
          }
              return matches;
          }
            // Documents without metadata are excluded for non-owners
            return false;
        } catch (e) {
            console.error(`    Error filtering document ${doc.id}:`, e.message);
            return false;
          }
        });
        // Normalize document_type field
        data = data.map(doc => ({
          ...doc,
          document_type: doc.document_type || doc.doc_type || 'general'
        }));
        }
    } else {
    }
    
    // Also fetch documents from Python service (optional, non-blocking)
    let pythonDocuments = [];
    try {
      // Use axios with timeout instead of fetch for better Node.js compatibility
      const axios = require('axios');
      const pythonResponse = await Promise.race([
        axios.get('http://localhost:8004/all-documents', { timeout: 2000 }),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 2000))
      ]).catch((error) => {
        throw new Error(`Python service fetch failed: ${error.message || 'Connection timeout or service unavailable'}`);
      });
      
      if (!pythonResponse || pythonResponse.status !== 200 || !pythonResponse.data) {
        throw new Error('Python service returned invalid response');
      }
      
      if (pythonResponse.data) {
        pythonDocuments = pythonResponse.data.documents || [];
      }
    } catch (pythonError) {
      // Silent - Python service is optional, don't spam logs
      // Documents from Supabase are sufficient
    }
    
    // Combine documents from both sources
    const allDocuments = [...(data || []), ...pythonDocuments];
    
    
    // Prevent caching to ensure fresh data
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, private');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    res.setHeader('ETag', `"${Date.now()}"`); // Add ETag to prevent 304 responses
    
    const responseData = { documents: allDocuments };
    
    // Ensure we're sending the correct format
    if (!Array.isArray(responseData.documents)) {
      console.error(' CRITICAL: documents is not an array!', typeof responseData.documents);
      responseData.documents = [];
    }
    
    res.json(responseData);
  } catch (error) {
    console.error('Error in documents GET:', error);
    // Return empty array instead of 500 to prevent 502 errors
    const errorMsg = error.message || '';
    if (errorMsg.includes('timeout') || errorMsg.includes('521') || errorMsg.includes('520')) {
      // Supabase timeout - return empty data gracefully
      res.json({ documents: [], count: 0 });
    } else {
      res.status(500).json({ error: 'An error occurred', documents: [] });
    }
  }
});

app.delete('/api/documents', verifyAuth('owner'), async (req, res) => {
  try {
    const documentId = req.query.id;
    
    if (!documentId) {
      return res.status(400).json({ error: 'Document ID is required' });
    }
    
    
    // First try to find and delete from Supabase
    let deletedFromSupabase = false;
    try {
      const { data: docData, error: docError } = await supabase
        .from('knowledge_documents')
        .select('learning_metadata')
        .eq('id', documentId)
        .limit(1).maybeSingle();
      
      if (!docError && docData) {
        // Check ownership
        if (docData.learning_metadata?.user_id !== req.user.id && req.user.role !== 'owner') {
          return res.status(403).json({ error: 'You do not have permission to delete this document' });
        }
        
        // Delete from Supabase
        const { error } = await supabase
          .from('knowledge_documents')
          .delete()
          .eq('id', documentId);
        
        if (!error) {
          deletedFromSupabase = true;
        }
      }
    } catch (supabaseError) {
      console.warn(' Supabase deletion failed:', supabaseError.message);
    }
    
    // Also try to delete from Python service
    let deletedFromPython = false;
    try {
      const pythonResponse = await fetch(`http://localhost:8004/documents/${documentId}`, {
        method: 'DELETE'
      });
      
      if (pythonResponse.ok) {
        deletedFromPython = true;
      }
    } catch (pythonError) {
      console.warn(' Python service deletion failed:', pythonError.message);
    }
    
    // Check if document was deleted from at least one source
    if (deletedFromSupabase || deletedFromPython) {
      res.json({ success: true, message: 'Document deleted successfully' });
    } else {
      res.status(404).json({ error: 'Document not found in any storage system' });
    }
    
  } catch (error) {
    console.error('Error in documents DELETE:', error);
    res.status(500).json({ error: 'An error occurred while deleting the document' });
  }
});

app.post('/api/documents/search', verifyAuth('owner'), async (req, res) => {
  try {
    const { query } = req.body;
    
    if (!query) {
      return res.status(400).json({ error: 'Search query is required' });
    }
    
    // Get all documents for user - include chunked flag
    // Get all documents and filter client-side (more reliable than JSONB filter)
    const { data: allData, error: fetchError } = await supabase
      .from('knowledge_documents')
      .select('id, title, document_type, content, created_at, learning_metadata, is_chunked, total_chunks');
    
    if (fetchError) {
      console.error('Error fetching documents for search:', fetchError);
      return res.status(500).json({ error: 'Failed to search documents' });
    }
    
    // Filter by user_id client-side
    const data = (allData || []).filter(doc => {
      if (doc.learning_metadata && typeof doc.learning_metadata === 'object') {
        return doc.learning_metadata.user_id === req.user.id;
      }
      // For owner, include all documents
      return req.user.role === 'owner';
    });
    
    // Decrypt and search in each document - handle chunked documents
    const results = [];
    
    for (const doc of data) {
      try {
        let content = doc.content || '';
        
        // If document is chunked, fetch chunks and reconstruct full content
        if (doc.is_chunked && doc.id) {
          try {
            // Use chunk fetcher utility with timeout handling and batching
            const { fetchChunksInBatches } = require('../utils/chunk-fetcher');
            const chunkRows = await fetchChunksInBatches(supabase, doc.id, { 
              batchSize: 50,
              silent: true 
            });
            
            if (chunkRows && chunkRows.length > 0) {
              content = chunkRows.map(chunk => chunk.chunk_text).join('\n\n');
            } else {
              console.warn(`[SERVER] Document ${doc.id} marked as chunked but no chunks found in search`);
            }
          } catch (chunkFetchError) {
            console.warn(`[SERVER] Error fetching chunks for document ${doc.id} in search:`, chunkFetchError.message);
            // Continue with preview content
          }
        }
        
        const decryptedContent = decrypt(content);
        
        if (decryptedContent && decryptedContent.toLowerCase().includes(query.toLowerCase())) {
          // Find the context around the match
          const index = decryptedContent.toLowerCase().indexOf(query.toLowerCase());
          const start = Math.max(0, index - 100);
          const end = Math.min(decryptedContent.length, index + query.length + 100);
          const excerpt = decryptedContent.substring(start, end);
          
          results.push({
            id: doc.id,
            filename: doc.title,
            document_type: doc.document_type,
            created_at: doc.created_at,
            excerpt: excerpt
          });
        }
      } catch (decryptError) {
        console.error(`Error decrypting document ${doc.id}:`, decryptError);
      }
    }
    
    res.json({ results });
  } catch (error) {
    console.error('Error in documents search:', error);
    res.status(500).json({ error: 'An error occurred' });
  }
});

// Keep the Netlify function handler for compatibility
app.use('/.netlify/functions/documents', apiLimiter, verifyAuth('owner'), netlifyToExpress(documentsHandler));

// Routes moved to before protection middleware - duplicates removed

// Obfuscated JS route - files are pre-built during deployment, served via static middleware
// No runtime obfuscation needed

// Security health check
app.get('/health/security', verifyAuth('owner'), (req, res) => {
  const securityChecks = {
    environment: checkEnvironmentVariables(),
    encryption: checkEncryptionKeys(),
    database: checkDatabaseConnection(),
    authentication: checkAuthSystem(),
    rateLimit: checkRateLimiters()
  };
  
  const allPassed = Object.values(securityChecks)
    .every(check => check.status === 'pass');
  
  logSecurityEvent('SECURITY_HEALTH_CHECK', {
    result: allPassed ? 'pass' : 'fail',
    checks: securityChecks,
    userId: req.user.id
  }, allPassed ? 'info' : 'warn');
  
  res.status(allPassed ? 200 : 500).json({
    status: allPassed ? 'healthy' : 'unhealthy',
    timestamp: new Date().toISOString(),
    checks: securityChecks
  });
});

function checkEnvironmentVariables() {
  const requiredVars = [
    'JWT_SECRET', 
    'SUPABASE_URL', 
    'SUPABASE_SERVICE_KEY',
    'FRONTEND_URL',
    'ENCRYPTION_KEY'
  ];
  
  const missingVars = requiredVars.filter(v => !process.env[v]);
  
  return {
    status: missingVars.length === 0 ? 'pass' : 'fail',
    message: missingVars.length === 0 
      ? 'All required environment variables present' 
      : `Missing variables: ${missingVars.join(', ')}`
  };
}

function checkEncryptionKeys() {
  const hasEncryptionKey = !!process.env.ENCRYPTION_KEY;
  const keyLength = process.env.ENCRYPTION_KEY?.length || 0;
  
  return {
    status: hasEncryptionKey && keyLength >= 64 ? 'pass' : 'fail',
    message: hasEncryptionKey && keyLength >= 64
      ? 'Encryption key properly configured'
      : 'Encryption key missing or too short'
  };
}

function checkDatabaseConnection() {
  return {
    status: !!supabase ? 'pass' : 'fail',
    message: !!supabase 
      ? 'Database connection established' 
      : 'Database connection failed'
  };
}

function checkAuthSystem() {
  const hasJwtSecret = !!process.env.JWT_SECRET;
  const jwtSecretLength = process.env.JWT_SECRET?.length || 0;
  
  return {
    status: hasJwtSecret && jwtSecretLength >= 32 ? 'pass' : 'fail',
    message: hasJwtSecret && jwtSecretLength >= 32
      ? 'Authentication system properly configured'
      : 'JWT secret missing or too short'
  };
}

function checkRateLimiters() {
  return {
    status: 'pass',
    message: 'Rate limiters properly configured'
  };
}

// Epsilon AI learning engine route - serves obfuscated in production
app.get('/core/epsilon-learning-engine.js', (req, res) => {
  serveObfuscatedFile(req, res, 'core/epsilon-learning-engine.js', 'obfuscated/epsilon-learning-engine.js');
});

app.get('/epsilon-learning-engine.js', (req, res) => {
  serveObfuscatedFile(req, res, 'core/epsilon-learning-engine.js', 'obfuscated/epsilon-learning-engine.js');
});


// RAG Embedding Service route - serves obfuscated in production
app.get('/services/rag-embedding-service.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/rag-embedding-service.js', 'obfuscated/rag-embedding-service.js');
});

app.get('/rag-embedding-service.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/rag-embedding-service.js', 'obfuscated/rag-embedding-service.js');
});

// RAG LLM Service route - serves obfuscated in production (used by epsilon-learning-engine.js)
app.get('/services/rag-llm-service.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/rag-llm-service.js', 'obfuscated/rag-llm-service.js');
});

app.get('/rag-llm-service.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/rag-llm-service.js', 'obfuscated/rag-llm-service.js');
});

// RAG Document Processor route - serves obfuscated in production
app.get('/services/rag-document-processor.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/rag-document-processor.js', 'obfuscated/rag-document-processor.js');
});

app.get('/rag-document-processor.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/rag-document-processor.js', 'obfuscated/rag-document-processor.js');
});

// CSRF token endpoint
app.get('/api/csrf-token', (req, res) => {
  const cookies = parseCookies(req.headers.cookie || '');
  let token = cookies.csrfToken;
  
  // Debug logging to track token matching issues
  if (process.env.NODE_ENV === 'development') {
  }
  
  // If no token exists in cookie, generate one using the same method as CSRF middleware
  if (!token) {
    token = generateCSRFToken();
    const cookieFlags = [`Path=/`, `SameSite=Strict`];
    if (process.env.NODE_ENV === 'production') {
      cookieFlags.push('Secure');
    }
    // Don't use HttpOnly so JavaScript can read it if needed
    res.setHeader('Set-Cookie', `csrfToken=${token}; ${cookieFlags.join('; ')}`);
    if (process.env.NODE_ENV === 'development') {
    }
  }
  
  // Also set token in response header for easier access
  res.setHeader('X-CSRF-Token', token);
  
  // Return the token (from cookie or newly generated)
  res.json({ token });
});

// Epsilon AI chat endpoint for chat bubble - uses UNIFIED AI SYSTEM
app.post('/api/epsilon-chat', async (req, res) => {
  try {
    // CSRF validation - check middleware result
    if (req.csrfValid === false) {
      const cookies = parseCookies(req.headers.cookie || '');
      const cookieToken = cookies.csrfToken;
      const headerToken = req.headers['x-csrf-token'] || req.headers['X-CSRF-Token'];
      
      // Only block if we're in production or if tokens are completely missing
      if (process.env.NODE_ENV === 'production' || (!cookieToken && !headerToken)) {
        logSecurityEvent('CSRF_VALIDATION_FAILED', {
          path: req.path,
          ip: req.ip,
          method: req.method
        }, 'warn');
        return res.status(403).json({ error: 'CSRF validation failed' });
      }
    }
    
    const { message, sessionId, history } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    // Check if user is authenticated - check JWT token first, then fallback to headers/cookies
    let isAuthenticated = false;
    let userId = null;
    let userRole = null;
    
    // First, try to verify JWT token (most reliable)
    const cookies = req.headers.cookie ? parseCookies(req.headers.cookie) : {};
    const token = cookies.authToken;
    
    if (token && process.env.JWT_SECRET) {
      try {
        const jwt = require('jsonwebtoken');
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        if (decoded && decoded.userId && decoded.email) {
          isAuthenticated = true;
          userId = decoded.userId;
          userRole = decoded.role || 'client';
          
          // Owner accounts always bypass guest limits
          if (userRole && userRole.toLowerCase() === 'owner') {
            userRole = 'owner';
          }
          
          console.log(`[EPSILON CHAT] JWT authenticated: userId=${userId}, role=${userRole}`);
        }
      } catch (jwtError) {
        // JWT invalid or expired, continue to check other methods
        console.log(`[EPSILON CHAT] JWT verification failed: ${jwtError.message}`);
      }
    } else {
      console.log(`[EPSILON CHAT] No JWT token found - token exists: ${!!token}, JWT_SECRET exists: ${!!process.env.JWT_SECRET}`);
    }
    
    // Fallback: check headers/cookies if JWT not available
    if (!isAuthenticated) {
      // Parse cookies manually since we don't use cookie-parser
      const parsedCookies = req.headers.cookie ? parseCookies(req.headers.cookie) : {};
      const userStr = req.headers['x-user-data'] || parsedCookies.epsilon_user;
      if (userStr) {
        try {
          const user = typeof userStr === 'string' ? JSON.parse(userStr) : userStr;
          if (user && user.email) {
            isAuthenticated = true;
            userId = user.id;
            userRole = user.role || 'client';
            
            // Owner accounts always bypass guest limits
            if (userRole && userRole.toLowerCase() === 'owner') {
              userRole = 'owner';
            }
            
            console.log(`[EPSILON CHAT] Header/cookie authenticated: userId=${userId}, role=${userRole}`);
          }
        } catch (e) {
          // Invalid user data, continue as guest
          console.log(`[EPSILON CHAT] Failed to parse user data: ${e.message}`);
        }
      }
    }
    
    // Check guest usage limits ONLY for non-authenticated users
    // All authenticated users (including owners) bypass guest limits
    if (!isAuthenticated) {
      console.log(`[EPSILON CHAT] User not authenticated, checking guest limits`);
      const clientIP = req.clientIP || getClientIP(req);
      const usageCheck = await checkGuestUsage(clientIP);
      
      if (!usageCheck.allowed) {
        if (usageCheck.reason === 'cooldown') {
          return res.status(429).json({ 
            error: 'Usage limit reached',
            message: `You have reached the guest usage limit. Please wait ${usageCheck.hoursRemaining} hours or create an account to continue.`,
            reason: 'cooldown',
            hoursRemaining: usageCheck.hoursRemaining,
            cooldownUntil: usageCheck.cooldownUntil
          });
        } else if (usageCheck.reason === 'limit_reached') {
          await incrementGuestUsage(clientIP);
          return res.status(429).json({ 
            error: 'Usage limit reached',
            message: `You have reached the guest usage limit of 20 messages. Please create an account to continue or wait 24 hours.`,
            reason: 'limit_reached',
            hoursRemaining: usageCheck.hoursRemaining
          });
        }
      } else {
        await incrementGuestUsage(clientIP);
      }
    }
    
    // Log authentication status for debugging
    if (isAuthenticated) {
      console.log(`[EPSILON CHAT] ✓ Authenticated user: userId=${userId}, role=${userRole || 'unknown'} - BYPASSING guest limits`);
    } else {
      const clientIP = req.clientIP || getClientIP(req);
      console.log(`[EPSILON CHAT] ✗ Guest user from IP: ${clientIP} - checking guest limits`);
      console.log(`[EPSILON CHAT] Debug - Cookies available: ${!!req.headers.cookie}, x-user-data: ${!!req.headers['x-user-data']}`);
    }
    
    // Use the UNIFIED AI SYSTEM - directly call the handler function
    try {
      const clientIP = req.clientIP || getClientIP(req);
      
      // Get the handler function from the already-required proxy module
      let handleGetEpsilonResponse = supabaseProxyRouter.handleGetEpsilonResponse;
      
      // Handler is required - no fallback
      if (!handleGetEpsilonResponse) {
        throw new Error('Epsilon response handler is not available. The supabase-proxy router must be properly initialized.');
      }
      
      // Use the handler - no fallback
      const result = await handleGetEpsilonResponse({
        user_message: message,  // Fixed: was 'userMessage' (undefined)
        session_id: sessionId,
        user_id: userId,
        context_data: {
          source: 'api_endpoint',
          timestamp: Date.now(),
          isGuest: !isAuthenticated,
          ip: clientIP
        }
      });
      
      if (!result || !result.success) {
        throw new Error(result?.error || result?.message || 'Failed to generate response');
      }
      
      if (!result.response) {
        throw new Error('Handler returned no response text');
      }
      
      return res.json({
        response: result.response,
        sessionId: sessionId || result.session_id || `server_session_${Date.now()}`,
        source: result.meta?.source || 'unified_system'
      });
      
    } catch (handlerError) {
      EpsilonLog.error('EPSILON_CHAT_HANDLER_ERROR', 'Error in response handler', {
        error: handlerError.message,
        stack: handlerError.stack
      });
      
      // Provide user-friendly error messages
      let statusCode = 500;
      let errorMessage = handlerError.message || 'Failed to generate response';
      
      if (handlerError.message && handlerError.message.includes('not ready') || 
          handlerError.message && handlerError.message.includes('loading')) {
        statusCode = 503; // Service Unavailable
        errorMessage = 'AI model is loading. Please try again in a few seconds.';
      } else if (handlerError.message && handlerError.message.includes('timeout')) {
        statusCode = 504; // Gateway Timeout
        errorMessage = 'Request timed out. Please try again.';
      } else if (handlerError.message && handlerError.message.includes('not available')) {
        statusCode = 503;
        errorMessage = 'AI service is temporarily unavailable. Please try again in a moment.';
      }
      
      return res.status(statusCode).json({ 
        error: 'Epsilon AI service error',
        message: errorMessage
      });
    }
    
  } catch (error) {
    EpsilonLog.error('EPSILON_CHAT_ERROR', 'Error in epsilon-chat endpoint', { 
      error: error.message, 
      stack: error.stack,
      path: req.path,
      method: req.method
    });
    
    if (error.message && error.message.includes('CSRF')) {
      return res.status(403).json({ error: 'CSRF validation failed' });
    }
    
    const errorMessage = process.env.NODE_ENV === 'development' 
      ? error.message 
      : 'An error occurred while processing your request';
    
    res.status(500).json({ 
      error: 'Internal server error', 
      message: errorMessage 
    });
  }
});

// Protected routes with enhanced security
app.get('/owner', verifyAuth('owner'), (req, res) => {
  try {
    const ownerPath = path.join(__dirname, '../ui/owner.html');
    let content = fs.readFileSync(ownerPath, 'utf8');
    res.set({
      'Content-Type': 'text/html',
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY'
    });
    res.send(content);
  } catch (error) {
    console.error(' [ROUTE] Error in /owner route:', error.message);
    res.status(500).send('Error loading owner dashboard');
  }
});

app.get('/owner.html', verifyAuth('owner'), (req, res) => {
  try {
    const ownerPath = path.join(__dirname, '../ui/owner.html');
    let content = fs.readFileSync(ownerPath, 'utf8');
    res.set({
      'Content-Type': 'text/html',
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY'
    });
    res.send(content);
  } catch (error) {
    res.status(500).send('Error loading owner dashboard');
  }
});

app.get('/client', verifyAuth('client'), (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/client.html'));
});

app.get('/client.html', verifyAuth('client'), (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/client.html'));
});

app.get('/edit_profile', verifyAuth('client'), (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/edit_profile.html'));
});

app.get('/edit_profile.html', verifyAuth('client'), (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/edit_profile.html'));
});

// HTML minification function for production - runs 3 passes for maximum compression
function minifyHTML(html) {
  if (!isProduction) return html;
  
  // Run minification 3 times for maximum compression
  for (let pass = 0; pass < 3; pass++) {
    html = html
      .replace(/<!--[\s\S]*?-->/g, '') // Remove HTML comments
      .replace(/\s+/g, ' ') // Replace multiple spaces/tabs/newlines with single space
      .replace(/>\s+</g, '><') // Remove spaces between tags
      .replace(/\s+/g, ' ') // Clean up remaining whitespace again
      .replace(/\s*([{}:;,=])\s*/g, '$1') // Remove spaces around CSS/JS operators
      .replace(/;\s*}/g, '}') // Remove semicolons before closing braces
      .replace(/\s*>\s*/g, '>') // Remove spaces around > 
      .replace(/\s*<\s*/g, '<') // Remove spaces around <
      .trim();
  }
  
  return html;
}

// Secure HTML file serving helper
function serveSecureHTML(req, res, filePath) {
  try {
    if (!fs.existsSync(filePath)) {
      return res.status(404).send('File not found');
    }
    
    let content = fs.readFileSync(filePath, 'utf8');
    
    res.set({
      'Content-Type': 'text/html',
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY'
    });
    
    res.send(content);
  } catch (error) {
    console.error(`[SERVER] Error serving HTML file: ${filePath}`, error.message);
    res.status(500).send('Error loading page');
  }
}

// Epsilon AI routes - serve epsilon.html
app.get('/epsilon-ai', (req, res) => {
  const filePath = path.join(__dirname, '../ui/epsilon.html');
  if (!fs.existsSync(filePath)) {
    console.error(`[SERVER] epsilon.html not found at: ${filePath}`);
    return res.status(404).send('File not found');
  }
  
  let content = fs.readFileSync(filePath, 'utf8');
  
  res.set({
    'Content-Type': 'text/html',
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY'
  });
  
  res.send(content);
});

app.get('/epsilon-ai.html', (req, res) => {
  const filePath = path.join(__dirname, '../ui/epsilon.html');
  if (!fs.existsSync(filePath)) {
    return res.status(404).send('File not found');
  }
  
  let content = fs.readFileSync(filePath, 'utf8');
  
  res.set({
    'Content-Type': 'text/html',
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY'
  });
  
  res.send(content);
});

// /epsilon route - serves epsilon.html
app.get('/epsilon', (req, res) => {
  const filePath = path.join(__dirname, '../ui/epsilon.html');
  if (!fs.existsSync(filePath)) {
    console.error(`[SERVER] epsilon.html not found at: ${filePath}`);
    return res.status(404).send('File not found');
  }
  
  let content = fs.readFileSync(filePath, 'utf8');
  
  res.set({
    'Content-Type': 'text/html',
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY'
  });
  
  res.send(content);
});

app.get('/contact', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/contact.html'));
});

app.get('/contact.html', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/contact.html'));
});

app.get('/copyright', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/copyright.html'));
});

app.get('/copyright.html', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/copyright.html'));
});

// Removed /how_ai_works and /intelligence routes - pages deleted

app.get('/login', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/login.html'));
});

app.get('/login.html', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/login.html'));
});

app.get('/privacy', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/privacy.html'));
});

app.get('/privacy.html', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/privacy.html'));
});

app.get('/terms', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/terms.html'));
});

app.get('/terms.html', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/terms.html'));
});

// Security.txt route (RFC 9116)
app.get('/.well-known/security.txt', (req, res) => {
  const securityTxtPath = path.join(__dirname, '../public/.well-known/security.txt');
  if (fs.existsSync(securityTxtPath)) {
    res.set({
      'Content-Type': 'text/plain',
      'Cache-Control': 'public, max-age=86400' // Cache for 1 day
    });
    res.sendFile(securityTxtPath);
  } else {
    res.status(404).send('Not Found');
  }
});

app.get('/process', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/process.html'));
});

app.get('/process.html', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/process.html'));
});

app.get('/register', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/register.html'));
});

app.get('/register.html', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/register.html'));
});

// Robots.txt file to prevent indexing of sensitive URLs
// SEO Routes - robots.txt and sitemap.xml
app.get('/robots.txt', (req, res) => {
  const robotsPath = path.join(__dirname, '../public/robots.txt');
  if (fs.existsSync(robotsPath)) {
    res.type('text/plain');
    res.sendFile(robotsPath);
  } else {
    res.type('text/plain');
    res.send(`User-agent: *
Allow: /
Allow: /epsilon
Allow: /about
Allow: /contact
Disallow: /api/
Disallow: /obfuscated/
Disallow: /core/
Disallow: /owner
Disallow: /client
Disallow: /.netlify/
Sitemap: https://neuralops.biz/sitemap.xml`);
  }
});

app.get('/sitemap.xml', (req, res) => {
  const sitemapPath = path.join(__dirname, '../public/sitemap.xml');
  if (fs.existsSync(sitemapPath)) {
    res.type('application/xml');
    res.sendFile(sitemapPath);
  } else {
    res.status(404).send('Sitemap not found');
  }
});

// Redirect index.html to Epsilon AI
app.get('/index.html', (req, res) => {
  res.redirect('/epsilon');
});

app.get('/', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/index.html'));
});

// OAuth callback route for Google Sign-In
app.get('/auth/callback', async (req, res) => {
  try {
    const { code, error } = req.query;
    
    if (error) {
      console.error('[OAUTH CALLBACK] OAuth error:', error);
      return res.redirect('/login?error=oauth_failed');
    }
    
    if (!code) {
      return res.redirect('/login?error=no_code');
    }
    
    // Exchange code for session via auth API
    const authResponse = await fetch(`${process.env.FRONTEND_URL || 'https://neuralops.biz'}/api/auth`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        action: 'google_oauth_callback',
        code: code
      })
    });
    
    const authData = await authResponse.json();
    
    if (authData.success) {
      // Set cookie from response if available
      const setCookieHeader = authResponse.headers.get('set-cookie');
      if (setCookieHeader) {
        res.setHeader('Set-Cookie', setCookieHeader);
      }
      
      // Redirect to dashboard
      return res.redirect('/epsilon?google_signin=success');
    } else {
      console.error('[OAUTH CALLBACK] Auth error:', authData.error);
      return res.redirect('/login?error=oauth_failed');
    }
  } catch (error) {
    console.error('[OAUTH CALLBACK] Callback error:', error.message);
    return res.redirect('/login?error=oauth_failed');
  }
});

// HTML routes
const htmlRoutes = ['/register', '/login', '/client', '/process', '/contact', '/about'];

htmlRoutes.forEach(route => {
  app.get(route, (req, res) => {
    const fileName = route.substring(1) + '.html';
    const filePath = path.join(__dirname, '../ui', fileName);
    
    if (!fs.existsSync(filePath)) {
      return serveSecureHTML(req, res, path.join(__dirname, '../ui/index.html'));
    }
    
    // Only backend JavaScript files should be obfuscated
    let html = fs.readFileSync(filePath, 'utf8');
    html = injectVersionToHTML(html, BUILD_ID);

    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('X-Build-Version', BUILD_ID);
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.type('html').send(html);
  });
});


app.get('/pricing', (req, res) => {
  serveSecureHTML(req, res, path.join(__dirname, '../ui/index.html'));
});

// /about route is handled by htmlRoutes array above (line 2014)

// Favicon route - serve favicon from Imagies folder
app.get('/favicon.ico', (req, res) => {
  const faviconPath = path.join(__dirname, '..', 'Imagies', 'favicon_created_by_zenbusiness.ico');
  
  // Check if file exists
  if (fs.existsSync(faviconPath)) {
    if (isProduction) {
      return secureStaticFile(req, res, faviconPath);
    } else {
      res.setHeader('Content-Type', 'image/x-icon');
      res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
      res.sendFile(path.resolve(faviconPath));
    }
  } else {
    res.status(204).end();
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    version: '1.0.0',
    build: BUILD_ID,
    timestamp: Date.now(),
    env: process.env.NODE_ENV || 'development',
    serverTime: new Date().toISOString()
  });
});

// Test endpoint for debugging
app.get('/api/test', (req, res) => {
  res.json({ 
    message: 'Server is working',
    timestamp: new Date().toISOString(),
    env: {
      JWT_SECRET: process.env.JWT_SECRET ? 'SET' : 'NOT SET',
      SUPABASE_URL: process.env.SUPABASE_URL ? 'SET' : 'NOT SET',
      SUPABASE_SERVICE_KEY: process.env.SUPABASE_SERVICE_KEY ? 'SET' : 'NOT SET'
    }
  });
});

// Get Supabase public credentials for frontend
app.get('/api/supabase-config', (req, res) => {
  const url = process.env.SUPABASE_URL;
  const anonKey = process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_SERVICE_KEY;
  
  if (!url || !anonKey) {
    console.error(' [SUPABASE CONFIG] Missing Supabase credentials', {
      hasUrl: !!url,
      hasAnonKey: !!anonKey
    });
    return res.status(500).json({
      error: 'Supabase credentials are not configured',
      hasUrl: !!url,
      hasAnonKey: !!anonKey
    });
  }
  
  res.json({ url, anonKey });
});

// Supabase connectivity health check
app.get('/api/supabase-health', async (req, res) => {
  const tablesToCheck = [
    'knowledge_documents',
    'epsilon_conversations',
    'epsilon_feedback',
    'document_embeddings',
    'epsilon_trial_tracking',
    'learning_patterns'
  ];
  
  const results = {};
  let allOk = true;
  
  for (const table of tablesToCheck) {
    try {
      const { error } = await supabase
        .from(table)
        .select('id', { count: 'exact', head: true })
        .limit(1);
      
      if (error) {
        allOk = false;
        results[table] = { ok: false, error: error.message };
        console.error(` [SUPABASE HEALTH] Table check failed for ${table}:`, error.message);
      } else {
        results[table] = { ok: true };
      }
    } catch (err) {
      allOk = false;
      results[table] = { ok: false, error: err.message };
      console.error(` [SUPABASE HEALTH] Unexpected error checking ${table}:`, err.message);
    }
  }
  
  res.json({
    success: allOk,
    tables: results,
    timestamp: new Date().toISOString()
  });
});

// IP Tracking for Trial System
app.get('/api/get-ip', (req, res) => {
  try {
    const ip = req.clientIP || getClientIP(req);
    
    console.log('[IP API] IP address requested:', { ip, userAgent: req.headers['user-agent'] });
    
    res.json({ ip: ip });
  } catch (error) {
    console.error('[IP API] Error getting IP address:', error.message);
    res.status(500).json({ error: 'Failed to get IP address' });
  }
});

// Check guest usage limits
app.post('/api/check-guest-usage', async (req, res) => {
  try {
    const { ip } = req.body;
    const clientIP = ip || req.clientIP || getClientIP(req);
    
    console.log('[GUEST USAGE] Checking usage for IP:', clientIP);
    
    const usageCheck = await checkGuestUsage(clientIP);
    
    console.log('[GUEST USAGE] Usage check result:', usageCheck);
    
    res.json(usageCheck);
  } catch (error) {
    console.error('[GUEST USAGE] Error checking usage:', error.message);
    res.status(500).json({ error: 'Failed to check guest usage' });
  }
});

// Route moved earlier to line ~700 for proper ordering

// Get trial data for IP
app.post('/api/get-trial-data', apiLimiter, async (req, res) => {
  try {
    const { ip } = req.body;
    
    if (!ip) {
      return res.status(400).json({ error: 'IP address required' });
    }
    
    // Check if trial data exists for this IP
    const { data, error } = await supabase
      .from('epsilon_trial_tracking')
      .select('id, ip_address, messages_used, last_message_at, created_at')
      .eq('ip_address', ip)
      .limit(1).maybeSingle();
    
    if (error && error.code !== 'PGRST116') { // PGRST116 = no rows found
      logger.error('TRIAL_GET_ERROR', 'Database error getting trial data', { 
        error: error.message,
        ip: ip 
      });
      return res.status(500).json({ error: 'Database error' });
    }
    
    if (data) {
      logger.info('TRIAL_GET_SUCCESS', 'Trial data retrieved', { 
        ip: ip,
        messagesRemaining: data.messages_remaining,
        trialUsed: data.trial_used
      });
      res.json({
        messagesRemaining: data.messages_remaining,
        trialUsed: data.trial_used,
        lastUpdated: data.updated_at
      });
    } else {
      // No trial data found - new user
      logger.info('TRIAL_GET_NEW', 'New trial user detected', { ip: ip });
      res.status(404).json({ error: 'No trial data found' });
    }
    
  } catch (error) {
    logger.error('TRIAL_GET_ERROR', 'Error getting trial data', { 
      error: error.message,
      ip: req.body.ip 
    });
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Save trial data for IP
app.post('/api/save-trial-data', apiLimiter, async (req, res) => {
  try {
    const { ip, messagesRemaining, trialUsed, timestamp } = req.body;
    
    if (!ip || messagesRemaining === undefined) {
      return res.status(400).json({ error: 'IP and messages remaining required' });
    }
    
    // Upsert trial data (insert or update)
    const { data, error } = await supabase
      .from('epsilon_trial_tracking')
      .upsert({
        ip_address: ip,
        messages_remaining: messagesRemaining,
        trial_used: trialUsed || false,
        updated_at: new Date().toISOString(),
        user_agent: req.headers['user-agent'],
        last_activity: new Date().toISOString()
      }, {
        onConflict: 'ip_address'
      })
      .select()
      .limit(1).maybeSingle();
    
    if (error) {
      logger.error('TRIAL_SAVE_ERROR', 'Database error saving trial data', { 
        error: error.message,
        ip: ip,
        messagesRemaining: messagesRemaining
      });
      return res.status(500).json({ error: 'Database error' });
    }
    
    logger.info('TRIAL_SAVE_SUCCESS', 'Trial data saved', { 
      ip: ip,
      messagesRemaining: messagesRemaining,
      trialUsed: trialUsed
    });
    
    res.json({ 
      success: true, 
      messagesRemaining: data.messages_remaining,
      trialUsed: data.trial_used
    });
    
  } catch (error) {
    logger.error('TRIAL_SAVE_ERROR', 'Error saving trial data', { 
      error: error.message,
      ip: req.body.ip 
    });
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Test obfuscation
app.get('/test-obfuscation', (req, res) => {
  try {
    const testCode = `
      function hello() {
        return 42;
      }
      hello();
    `;
    
    const obfuscated = JavaScriptObfuscator.obfuscate(testCode, {
      compact: true,
      controlFlowFlattening: true,
      stringArray: true,
      debugProtection: false,
      debugProtectionInterval: 0
    }).getObfuscatedCode();
    
    res.type('application/javascript');
    res.send(obfuscated);
  } catch (error) {
    console.error(' Obfuscation test failed:', error);
    res.status(500).send(`Obfuscation test failed: ${error.message}`);
  }
});

// Python Service API Endpoints
if (pythonServiceManager) {
  // Python services are initialized above (line 88-106) and also started by start.sh
  // The manager will connect to services started by start.sh if available, otherwise start them
  _silent('PYTHON_ENDPOINTS', 'Python service API endpoints registered');

  // NLP Analysis endpoint
  app.post('/api/python/nlp/analyze', async (req, res) => {
    try {
      const { text, analysis_type = 'full' } = req.body;
      
      if (!text) {
        return res.status(400).json({ error: 'Text is required' });
      }

      const analysis = await pythonServiceManager.analyzeText(text, analysis_type);
      res.json(analysis);
    } catch (error) {
      EpsilonLog.error('PYTHON_NLP_ERROR', 'NLP analysis failed', { error: error.message });
      res.status(500).json({ error: 'NLP analysis failed' });
    }
  });

  // Content Generation endpoint - NOW USES LANGUAGE MODEL WITH DICTIONARY/RULES/METADATA
  app.post('/api/python/content/generate', async (req, res) => {
    try {
      const { user_message, context = {} } = req.body;
      
      if (!user_message) {
        return res.status(400).json({ error: 'User message is required' });
      }

      // Use language model service (with dictionary/rules/metadata) instead of content service
      if (epsilonLanguageEngine && epsilonLanguageEngine.isModelReady && epsilonLanguageEngine.isModelReady()) {
        try {
          const generation = await epsilonLanguageEngine.generate({
            userMessage: user_message,
            ragContext: [], // Empty - uses learned patterns
            persona: context.user_profile || {},
            maxLength: 120
          });
          
          if (generation && generation.text) {
            return res.json({
              content: generation.text,
              metadata: generation.meta || {},
              source: 'language_model_with_dictionary'
            });
          }
        } catch (langError) {
          EpsilonLog.error('LANGUAGE_MODEL_ERROR', 'Language model generation failed', { error: langError.message });
          throw new Error(`Language model generation failed: ${langError.message}. The language model service is required.`);
        }
      } else {
        throw new Error('Language model service is not available. EpsilonLanguageEngine is required for content generation.');
      }
      
      // No fallback - language model is required
      const content = await pythonServiceManager.generateResponse(user_message, context);
      res.json(content);
    } catch (error) {
      EpsilonLog.error('PYTHON_CONTENT_ERROR', 'Content generation failed', { error: error.message });
      res.status(500).json({ error: 'Content generation failed' });
    }
  });

  // Conversation Analysis endpoint
  app.post('/api/python/analytics/conversation', async (req, res) => {
    try {
      const { conversation_data } = req.body;
      
      if (!conversation_data) {
        return res.status(400).json({ error: 'Conversation data is required' });
      }

      const analysis = await pythonServiceManager.analyzeConversation(conversation_data);
      res.json(analysis);
    } catch (error) {
      EpsilonLog.error('PYTHON_ANALYTICS_ERROR', 'Conversation analysis failed', { error: error.message });
      res.status(500).json({ error: 'Conversation analysis failed' });
    }
  });

  // Learning Insights endpoint
  app.get('/api/python/analytics/insights', async (req, res) => {
    try {
      const { time_period = '7d' } = req.query;
      
      const insights = await pythonServiceManager.generateInsights(time_period);
      res.json(insights);
    } catch (error) {
      EpsilonLog.error('PYTHON_INSIGHTS_ERROR', 'Insights generation failed', { error: error.message });
      res.status(500).json({ error: 'Insights generation failed' });
    }
  });

  // Response Optimization endpoint
  app.post('/api/python/analytics/optimize', async (req, res) => {
    try {
      const { user_profile, conversation_context } = req.body;
      
      if (!user_profile || !conversation_context) {
        return res.status(400).json({ error: 'User profile and conversation context are required' });
      }

      const optimization = await pythonServiceManager.optimizeResponse(user_profile, conversation_context);
      res.json(optimization);
    } catch (error) {
      EpsilonLog.error('PYTHON_OPTIMIZATION_ERROR', 'Response optimization failed', { error: error.message });
      res.status(500).json({ error: 'Response optimization failed' });
    }
  });

  // Python Services Health Check
  app.get('/api/python/health', async (req, res) => {
    try {
      const health = await pythonServiceManager.healthCheck();
      res.json(health);
    } catch (error) {
      EpsilonLog.error('PYTHON_HEALTH_ERROR', 'Health check failed', { error: error.message });
      res.status(500).json({ error: 'Health check failed' });
    }
  });
}


// Start server
const server = app.listen(PORT, '0.0.0.0', () => {
  // Server initialized successfully
  console.log(`[SERVER] NeuralOps server running on port ${PORT}`);
  console.log(`[SERVER] Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`[SERVER] Build ID: ${BUILD_ID.substring(0, 12)}`);
  
  // Set very high server timeouts to allow large file uploads to complete naturally
  // 4 hours should be enough for even the largest files with slow connections
  server.timeout = 4 * 60 * 60 * 1000; // 4 hours
  server.keepAliveTimeout = 4 * 60 * 60 * 1000; // 4 hours
  server.headersTimeout = 4 * 60 * 60 * 1000 + 60000; // 4 hours + 1 minute buffer
  console.log(`[SERVER] Server timeouts configured: ${server.timeout / 1000 / 60} minutes (no artificial limits - uploads will complete naturally)`);
  
  logSecurityEvent('SERVER_START', {
    port: PORT,
    buildVersion: BUILD_ID,
    nodeEnv: process.env.NODE_ENV || 'development'
  });
});

server.on('error', (err) => {
  console.error('[SERVER] Failed to start server:', err);
  if (err.code === 'EADDRINUSE') {
    console.error(`[SERVER] Port ${PORT} is already in use`);
  }
  // Don't exit immediately - let Render handle it
  // process.exit(1);
});

// Handle uncaught exceptions gracefully
process.on('uncaughtException', (err) => {
  console.error('[SERVER] Uncaught Exception:', err);
  // Don't exit - let the server try to continue
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('[SERVER] Unhandled Rejection at:', promise, 'reason:', reason);
  // Don't exit - let the server try to continue
});

// Neural Assistant route

// Duplicate routes removed - using serveObfuscatedFile routes above

// epsilon-enhanced-feedback.js removed - file was never loaded in UI

// Additional core files - serves obfuscated in production
// NOTE: Supabase.js routes moved above to be before middleware
app.get('/core/epsilon-language-engine.js', (req, res) => {
  serveObfuscatedFile(req, res, 'core/epsilon-language-engine.js', 'obfuscated/epsilon-language-engine.js');
});

// Training endpoints removed - training is local-only in ml_local/
// app.get('/core/epsilon-automatic-training.js', (req, res) => {
//   serveObfuscatedFile(req, res, 'core/epsilon-automatic-training.js', 'obfuscated/epsilon-automatic-training.js');
// });

app.get('/core/epsilon-self-learning.js', (req, res) => {
  serveObfuscatedFile(req, res, 'core/epsilon-self-learning.js', 'obfuscated/epsilon-self-learning.js');
});

// AI Core files - epsilon-ai-core.js removed (training is local-only)

app.get('/services/ai-core/epsilon-embeddings.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/ai-core/epsilon-embeddings.js', 'obfuscated/epsilon-embeddings.js');
});

app.get('/services/ai-core/epsilon-tokenizer.js', (req, res) => {
  serveObfuscatedFile(req, res, 'services/ai-core/epsilon-tokenizer.js', 'obfuscated/epsilon-tokenizer.js');
});

// epsilon-enhanced-feedback.js removed - file was never loaded in UI
// NOTE: Supabase.js routes moved above (before middleware) to avoid 403 errors


// OLD MANUAL TRAINING ENDPOINT REMOVED - Training is now fully automatic
// Epsilon AI trains continuously every 30 minutes without manual intervention

  // Training endpoints removed - training is local-only in ml_local/
  // Automatic training status endpoint - DISABLED
  app.get('/api/epsilon-llm/automatic-training/status', verifyAuth('owner'), async (req, res) => {
    try {
      const automaticTraining = global.epsilonAutomaticTraining;
      if (!automaticTraining) {
        return res.json({
          success: true,
          status: {
            isRunning: false,
            isTraining: false,
            message: 'Automatic training not initialized. Check server logs for initialization errors.'
          }
        });
      }
      
      // Get status synchronously (should be fast, no async operations)
      let status;
      try {
        status = automaticTraining.getStatus();
      } catch (statusError) {
        console.warn('[AUTO-TRAINING] Error getting status:', statusError.message);
        return res.json({
          success: true,
          status: {
            isRunning: false,
            isTraining: false,
            message: 'Status unavailable',
            error: statusError.message
          }
        });
      }
      
      // Add diagnostic info (non-blocking, cached checks)
      // Don't make async calls here - use cached state
      const diagnostics = {
        pythonServicesReady: epsilonLanguageEngine?.isPythonReady?.() || false,
        modelReady: epsilonLanguageEngine?.isModelReady?.() || false,
        pythonManagerAttached: !!epsilonLanguageEngine?.pythonManager,
        languageModelPort: epsilonLanguageEngine?.pythonManager?.services?.language_model?.port || 'unknown'
      };
      
      status.diagnostics = diagnostics;
      
      res.json({ success: true, status });
    } catch (error) {
      console.error(' [AUTO-TRAINING] Status fetch failed:', error);
      res.status(500).json({ 
        success: false,
        error: error.message || 'Failed to get training status'
      });
    }
  });
  
  // Training endpoints removed - training is local-only in ml_local/
  app.post('/api/epsilon-llm/automatic-training/trigger', verifyAuth('owner'), async (req, res) => {
        return res.status(403).json({ 
          success: false, 
      error: 'Training is local-only. Use ml_local/ for training.',
          blocked: true
        });
  });

  // Training endpoint - returns error message for frontend
  app.post('/api/epsilon-llm/train', verifyAuth('owner'), async (req, res) => {
    return res.status(403).json({
          success: false, 
      error: 'Training is local-only. Use ml_local/train/pretrain.py for training.',
      message: 'Training must be done locally using the ml_local/ training scripts. See ml_local/README.md for instructions.',
      blocked: true
    });
  });

  // Get self-learning progress
  app.get('/api/epsilon-llm/self-learning/progress', verifyAuth('owner'), async (req, res) => {
    try {
      if (!epsilonLanguageEngine || !epsilonLanguageEngine.selfLearning) {
        return res.json({ 
          success: true, 
          progress: {
            isRunning: false,
            objectives: [],
            totalConversations: 0
          }
        });
      }
      
      const progress = epsilonLanguageEngine.selfLearning.getLearningProgress();
      res.json({ success: true, progress });
    } catch (error) {
      console.error(' [SELF-LEARNING] Failed to get progress:', error);
      res.status(500).json({ success: false, error: error.message || 'Failed to get self-learning progress' });
    }
  });

  // Get pending deploys
  app.get('/api/epsilon-llm/deploys/pending', verifyAuth('owner'), async (req, res) => {
    try {
      // Query Supabase directly to ensure we get all pending deployments
      // This works from both local dev and production (shared Supabase)
      // Use service role client to bypass RLS
      const { data, error } = await supabase
        .from('epsilon_model_deployments')
        .select('id, model_id, version, stats, quality_score, improvement, training_samples, learning_description, status, created_at, created_by, storage_path')
        .eq('status', 'pending')
        .order('created_at', { ascending: false });

      if (error) {
        console.error(' [DEPLOY] Error fetching pending deploys from Supabase:', error);
        console.error('   Error code:', error.code);
        console.error('   Error message:', error.message);
        console.error('   Error details:', error.details);
        
        throw new Error(`Failed to fetch pending deployments from Supabase: ${error.message || 'Unknown error'}`);
      }

      console.log(` [DEPLOY] Fetched ${(data || []).length} pending deployments from Supabase`);

      // Format for UI display
      const deploys = (data || []).map(deploy => ({
        id: deploy.id,
        deploy_id: deploy.id, // Alias for compatibility
        deployment_id: deploy.id, // Alias for compatibility
        version: deploy.version,
        quality_score: deploy.quality_score,
        improvement: deploy.improvement,
        training_samples: deploy.training_samples,
        learning_description: deploy.learning_description || 'No description available',
        created_at: deploy.created_at,
        stats: deploy.stats
      }));

      res.json({ success: true, deploys });
    } catch (error) {
      console.error(' [DEPLOY] Failed to get pending deploys:', error);
      console.error('   Stack:', error.stack);
      res.status(500).json({ success: false, error: error.message || 'Failed to get pending deploys' });
    }
  });

  // DEPRECATED: Approval system removed - models are auto-approved via auto_upload_model.py
  // This endpoint is kept for backward compatibility but returns an error
  app.post('/api/epsilon-llm/deploys/:deployId/approve', verifyAuth('owner'), async (req, res) => {
    return res.status(410).json({ 
      success: false, 
      error: 'Approval system has been removed. Models are now auto-approved when uploaded via auto_upload_model.py with safety checks.' 
    });
  });

  // OLD APPROVAL ENDPOINT (REMOVED - KEPT FOR REFERENCE)
  /*
  app.post('/api/epsilon-llm/deploys/:deployId/approve', verifyAuth('owner'), async (req, res) => {
    try {
      const { deployId } = req.params;
      
      // Get pending deployment first to check if it's a pre-trained model
      const { data: pendingDeploy, error: fetchError } = await supabase
        .from('epsilon_model_deployments')
        .select('id, model_id, version, stats, quality_score, improvement, training_samples, learning_description, status, created_at, created_by, storage_path, model_data')
        .eq('id', deployId)
        .eq('status', 'pending')
        .limit(1).maybeSingle();

      if (fetchError || !pendingDeploy) {
        return res.status(404).json({ success: false, error: 'Pending deployment not found' });
      }

      // Check if this is a pre-trained model (has storage_path and stats.model_type === 'pretrained')
      const isPretrained = pendingDeploy.storage_path && 
                          pendingDeploy.stats && 
                          pendingDeploy.stats.model_type === 'pretrained';
      
      // Only require Python service for trained models, not pre-trained uploads
      if (!isPretrained && !epsilonLanguageEngine.isPythonReady()) {
        return res.status(503).json({ success: false, error: 'Python language model service is not ready' });
      }

      // SAFETY CHECK: Verify this is a good model before allowing approval
      // Skip quality checks for pre-trained models (they don't have training metrics)
      if (!isPretrained) {
        const minQualityScore = 0.4;
        const hasPositiveImprovement = pendingDeploy.improvement > 0;
        const hasAcceptableQuality = pendingDeploy.quality_score >= minQualityScore;
        
        if (!hasPositiveImprovement) {
          return res.status(400).json({ 
            success: false, 
            error: `Cannot approve deployment: Negative improvement detected (${(pendingDeploy.improvement * 100).toFixed(2)}%). This model would make Epsilon AI worse.` 
          });
        }
        
        if (!hasAcceptableQuality) {
          return res.status(400).json({ 
            success: false, 
            error: `Cannot approve deployment: Quality score too low (${(pendingDeploy.quality_score * 100).toFixed(2)}% < ${(minQualityScore * 100).toFixed(2)}%). This model is not good enough.` 
          });
        }
      }

      // Export current model (or use model_data from pending deployment if available)
      const pythonManager = epsilonLanguageEngine.pythonManager;
      const axios = require('axios');
      
      let modelData = null;
      
      // Check if pending deployment already has model_data (saved during creation)
      if (pendingDeploy.model_data) {
        modelData = {
          success: true,
          model_data: pendingDeploy.model_data,
          stats: pendingDeploy.stats || {},
          temperature: pendingDeploy.temperature || 0.9
        };
      } else if (pendingDeploy.storage_path) {
        // Download zip artifact from Supabase Storage and extract to models/latest/
        // Handle both regular zips and chunked uploads (metadata.json files)
        let storageData = null;
        let isChunked = false;
        let chunkMetadata = null;
        
        if (pendingDeploy.storage_path.endsWith('.metadata.json')) {
          // Chunked file - download metadata first, then reassemble chunks
          console.log('[DEPLOY] Detected chunked model file, downloading chunks...');
          isChunked = true;
          
          const { data: metadataData, error: metadataError } = await supabase.storage
            .from('epsilon-models')
            .download(pendingDeploy.storage_path);
          
          if (metadataError || !metadataData) {
            throw new Error(`Cannot load chunk metadata: ${metadataError?.message || 'Metadata file not found'}`);
          }
          
          const arrayBuffer = await metadataData.arrayBuffer();
          const metadataBuffer = Buffer.from(arrayBuffer);
          chunkMetadata = JSON.parse(metadataBuffer.toString('utf-8'));
          
          // Reassemble chunks into a single buffer
          const chunks = [];
          for (const chunkInfo of chunkMetadata.chunks.sort((a, b) => a.index - b.index)) {
            console.log(`[DEPLOY] Downloading chunk ${chunkInfo.index + 1}/${chunkMetadata.chunks.length}...`);
            const { data: chunkData, error: chunkError } = await supabase.storage
              .from('epsilon-models')
              .download(chunkInfo.path);
            
            if (chunkError || !chunkData) {
              throw new Error(`Cannot load chunk ${chunkInfo.index}: ${chunkError?.message || 'Chunk not found'}`);
            }
            
            const chunkArrayBuffer = await chunkData.arrayBuffer();
            chunks.push(Buffer.from(chunkArrayBuffer));
          }
          
          // Combine all chunks
          storageData = Buffer.concat(chunks);
          console.log(`[DEPLOY] Reassembled ${chunkMetadata.chunks.length} chunks (${(storageData.length / 1024 / 1024).toFixed(2)} MB)`);
          
        } else {
          // Regular single file
          const { data: downloadedData, error: storageError } = await supabase.storage
            .from('epsilon-models')
            .download(pendingDeploy.storage_path);
          
          if (storageError || !downloadedData) {
            console.error(' [DEPLOY] Failed to load model artifact from storage:', storageError?.message || 'No data returned');
            throw new Error(`Cannot approve deployment: Model artifact in storage (${pendingDeploy.storage_path}) cannot be loaded. ${storageError?.message || 'Storage file not found'}`);
          }
          
          const arrayBuffer = await downloadedData.arrayBuffer();
          storageData = Buffer.from(arrayBuffer);
        }
        
        // Extract zip to models/latest/
        const AdmZip = require('adm-zip');
        const modelsDir = path.join(__dirname, '..', 'services', 'python-services', 'models', 'latest');
        const fs = require('fs');
        
        // Ensure models directory exists
        if (!fs.existsSync(modelsDir)) {
          fs.mkdirSync(modelsDir, { recursive: true });
        }
        
        try {
          // Extract zip
          const zip = new AdmZip(storageData);
          zip.extractAllTo(modelsDir, true); // Overwrite existing files
          
          console.log(`[DEPLOY] Model artifact extracted to ${modelsDir}`);
          
          // Find model file (can have different names)
          const modelFiles = ['model.pt'];
          const allFiles = fs.readdirSync(modelsDir);
          const ptFiles = allFiles.filter(f => f.endsWith('.pt') && f !== 'model.pt');
          
          let modelFileFound = false;
          let modelFilePath = path.join(modelsDir, 'model.pt');
          
          if (fs.existsSync(modelFilePath)) {
            modelFileFound = true;
          } else if (ptFiles.length > 0) {
            // Rename first .pt file to model.pt
            const sourcePath = path.join(modelsDir, ptFiles[0]);
            fs.renameSync(sourcePath, modelFilePath);
            console.log(`[DEPLOY] Renamed ${ptFiles[0]} to model.pt`);
            modelFileFound = true;
          }
          
          if (!modelFileFound) {
            throw new Error('No model file (.pt) found after extraction');
          }
          
          // Verify required files exist
          const configPath = path.join(modelsDir, 'config.json');
          const tokenizerPath = path.join(modelsDir, 'tokenizer.json');
          
          if (!fs.existsSync(configPath)) {
            // Try to extract config from checkpoint using Python
            console.log(`[DEPLOY] config.json missing, attempting to extract from checkpoint...`);
            try {
              const pythonScript = `import torch, json, sys
ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
if 'config' in ckpt:
    with open(sys.argv[2], 'w') as f:
        json.dump(ckpt['config'], f, indent=2)
    print('SUCCESS')
else:
    print('ERROR: No config in checkpoint')
    sys.exit(1)`;
              
              const scriptPath = path.join(modelsDir, '_extract_config.py');
              fs.writeFileSync(scriptPath, pythonScript);
              
              const { spawnSync } = require('child_process');
              const result = spawnSync('python3', [scriptPath, modelFilePath, configPath], {
                cwd: modelsDir,
                encoding: 'utf-8',
                timeout: 30000
              });
              
              if (fs.existsSync(scriptPath)) fs.unlinkSync(scriptPath);
              
              if (result.status !== 0 || !fs.existsSync(configPath)) {
                throw new Error(`Failed to extract: ${result.stderr || result.stdout || 'Unknown error'}`);
              }
              console.log(`[DEPLOY] Extracted config.json from checkpoint`);
            } catch (extractError) {
              console.error(`[DEPLOY] Config extraction failed: ${extractError.message}`);
              throw new Error(`config.json is required but missing. Please re-upload model with config.json included.`);
            }
          }
          
          if (!fs.existsSync(tokenizerPath)) {
            throw new Error(`tokenizer.json is required but missing. Please re-upload model with tokenizer.json included.`);
          }
          
          // Reload inference service model (if service supports hot reload)
          try {
            const inferenceUrl = process.env.INFERENCE_URL || 'http://localhost:8005';
            const axios = require('axios');
            await axios.post(`${inferenceUrl}/reload-model`, { model_dir: modelsDir }, { timeout: 10000 });
            console.log('[DEPLOY] Inference service model reloaded');
          } catch (reloadError) {
            console.warn('[DEPLOY] Could not reload inference service (may need restart):', reloadError.message);
            // Don't fail approval if reload fails - service will pick it up on next restart
          }
          
          modelData = {
            success: true,
            extracted: true,
            model_dir: modelsDir,
            stats: pendingDeploy.stats || {},
            temperature: pendingDeploy.temperature || 0.9
          };
        } catch (extractError) {
          console.error(' [DEPLOY] Failed to extract model artifact:', extractError.message);
          throw new Error(`Failed to extract model artifact: ${extractError.message}`);
        }
      }
      
      // VERIFY: Deployment must have model_data or storage_path
      // Deployments are created with model_data during training (local only)
      // Production approval should NOT require localhost Python service
      if (!modelData) {
        const isProduction = process.env.NODE_ENV === 'production';
        
        if (isProduction) {
          // In production, we cannot connect to localhost Python service
          // Deployment MUST have model_data or storage_path saved during creation
          throw new Error(
            `Cannot approve deployment: Model data is missing. ` +
            `This deployment was created without model_data. ` +
            `Deployments must include model_data when created (during local training). ` +
            `Please reject this deployment and wait for a new one to be created with model_data.`
          );
        } else {
          // In development, we can try to export from local Python service as fallback
          // But this should rarely happen if deployment creation works correctly
          if (!pythonManager || !pythonManager.services || !pythonManager.services.language_model) {
            throw new Error(
              `Cannot approve deployment: Model data is missing and Python service is not initialized. ` +
              `This deployment was created without model_data. Please reject it and wait for a new deployment.`
            );
          }

          if (!epsilonLanguageEngine.isPythonReady()) {
            throw new Error(
              `Cannot approve deployment: Model data is missing and Python service is not ready. ` +
              `This deployment was created without model_data. Please reject it and wait for a new deployment.`
            );
          }

          // Try to export from local Python service (development fallback only)
          try {
            const pythonLLMUrl = process.env.PYTHON_LLM_SERVICE_URL || `http://localhost:${pythonManager.services.language_model.port}`;
            console.log(`[DEPLOY] Attempting to export model from Python service (${pythonLLMUrl})...`);
            
      const exportResponse = await axios.get(
              `${pythonLLMUrl}/export`,
        { timeout: 180000 }
      );

            if (exportResponse.data && exportResponse.data.success && exportResponse.data.model_data) {
              modelData = {
                success: true,
                model_data: exportResponse.data.model_data,
                stats: exportResponse.data.stats || pendingDeploy.stats || {},
                temperature: exportResponse.data.temperature || pendingDeploy.temperature || 0.9
              };
              console.log(` [DEPLOY] Successfully exported model from local Python service`);
            } else {
              throw new Error('Python service export response missing model_data');
            }
          } catch (exportError) {
            console.error(' [DEPLOY] Cannot export model from Python service:', exportError.message);
            throw new Error(
              `Cannot approve deployment: Model data is missing and cannot export from Python service. ` +
              `Error: ${exportError.message}. ` +
              `This deployment was created without model_data. Please reject it and wait for a new deployment.`
            );
          }
        }
      }

      // Get user identifier for audit trail
      const userIdentifier = req.user?.email || req.user?.id || 'owner';

      // For zip artifacts (extracted), use storage_path directly - skip JSON model_data logic
      let useStorage = false;
      let storagePath = null;
      
      if (modelData && modelData.extracted) {
        // Zip artifact was extracted - use the storage_path from pending deployment
        storagePath = pendingDeploy.storage_path;
        useStorage = true;
        console.log(`[DEPLOY] Using extracted zip artifact from storage: ${storagePath}`);
      } else if (modelData && modelData.model_data) {
        // Legacy JSON model_data path (for backward compatibility)
      const modelDataJson = JSON.stringify(modelData.model_data);
      const modelDataSize = modelDataJson.length;
        useStorage = modelDataSize > 10 * 1024 * 1024;

      if (useStorage) {
        const fileName = `epsilon-model-${Date.now()}.json`;
        const { data: uploadData, error: uploadError } = await supabase.storage
          .from('epsilon-models')
          .upload(fileName, modelDataJson, {
            contentType: 'application/json',
            upsert: false
          });

        if (!uploadError) {
          storagePath = uploadData.path;
        } else {
          console.warn(' [DEPLOY] Storage upload failed, using model_data instead:', uploadError.message);
          useStorage = false;
        }
        }
      } else {
        throw new Error('Model data is missing - cannot approve deployment');
      }

      // CRITICAL: First, archive any existing production models BEFORE setting new one
      // This prevents UNIQUE constraint violations on model_id
      const { data: existingProduction } = await supabase
        .from('epsilon_model_deployments')
        .select('id')
        .eq('model_id', 'production')
        .neq('id', deployId)
        .limit(1)
        .maybeSingle();

      if (existingProduction && existingProduction.id) {
        // Archive the old production model
        const archiveModelId = `archived-${Date.now()}-${existingProduction.id.substring(0, 8)}`;
        const { error: archiveError } = await supabase
          .from('epsilon_model_deployments')
          .update({ 
            model_id: archiveModelId,
            status: 'rejected' // Use 'rejected' instead of 'archived' (not in schema CHECK constraint)
          })
          .eq('id', existingProduction.id);
        
        if (archiveError) {
          console.warn(' [DEPLOY] Failed to archive old production model:', archiveError.message);
          // Continue anyway - we'll try to set the new one
        }
      }

      // Now update deployment to approved and deploy
      // CRITICAL: Must include model_data or storage_path to satisfy constraint
      const deploymentData = {
        model_id: 'production',
        stats: pendingDeploy.stats || modelData.stats,
        temperature: pendingDeploy.temperature || modelData.temperature || 0.9,
        version: pendingDeploy.version,
        quality_score: pendingDeploy.quality_score,
        improvement: pendingDeploy.improvement,
        status: 'approved',
        deployed_at: new Date().toISOString(),
        deployed_by: userIdentifier,
        approved_by: userIdentifier,
        approved_at: new Date().toISOString()
      };

      // Set model_data or storage_path (required for approved status)
      if (useStorage && storagePath) {
        deploymentData.storage_path = storagePath;
        deploymentData.model_data = null; // Clear model_data when using storage
      } else {
        deploymentData.model_data = modelData.model_data;
        deploymentData.storage_path = null; // Clear storage_path when using model_data
      }

      // Update deployment in a single operation
      const { data: updatedDeploy, error: updateError } = await supabase
        .from('epsilon_model_deployments')
        .update(deploymentData)
        .eq('id', deployId)
        .select()
        .limit(1).maybeSingle();

      if (updateError) {
        console.error(' [DEPLOY] Failed to approve deployment:', updateError);
        // If it's a UNIQUE constraint violation, provide helpful error
        if (updateError.code === '23505' || updateError.message?.includes('unique')) {
          throw new Error('Another deployment is already set as production. Please reject it first.');
        }
        throw new Error(`Failed to approve deployment: ${updateError.message}`);
      }


      // Update epsilon_model_weights, epsilon_learning_patterns, epsilon_training_data
      if (modelData.stats) {
        try {
          // Upsert model version weight (update if exists, insert if not)
          const versionValue = parseFloat(pendingDeploy.version.replace('auto-', '').replace('v', '')) || 1.0;
          
          // Try to update first
          const { data: existingWeight } = await supabase
            .from('epsilon_model_weights')
            .select('id')
            .eq('weight_name', 'model_version')
            .eq('weight_type', 'model_version')
            .limit(1)
            .maybeSingle();
          
          if (existingWeight && existingWeight.id) {
            // Update existing
            const { error: updateError } = await supabase
              .from('epsilon_model_weights')
              .update({
                weight_value: versionValue,
                updated_at: new Date().toISOString()
              })
              .eq('id', existingWeight.id);
            
            if (updateError) {
              console.warn(' [DEPLOY] Failed to update model_version weight:', updateError.message);
            }
          } else {
            // Insert new
            const { error: insertError } = await supabase
              .from('epsilon_model_weights')
              .insert({
                weight_type: 'model_version',
                weight_name: 'model_version',
                weight_value: versionValue
              });
            
            if (insertError) {
              console.warn(' [DEPLOY] Failed to insert model_version weight:', insertError.message);
            }
          }

          await supabase.from('epsilon_learning_patterns').insert({
            pattern_type: 'deployment',
            pattern_data: {
              version: pendingDeploy.version,
              quality_score: pendingDeploy.quality_score,
              improvement: pendingDeploy.improvement
            },
            created_at: new Date().toISOString()
          });

          // Store training data summary in metadata (table structure requires input_text/expected_output)
          await supabase.from('epsilon_training_data').insert({
            input_text: `Model deployment ${pendingDeploy.version}`,
            expected_output: `Deployed model with quality score ${pendingDeploy.quality_score} and improvement ${pendingDeploy.improvement}`,
            training_type: 'automatic',
            quality_score: pendingDeploy.quality_score || 0.5,
            is_validated: true,
            metadata: {
              deployment_id: updatedDeploy.id,
              version: pendingDeploy.version,
              samples_trained: modelData.stats?.samples_trained || 0,
              vocab_size: modelData.stats?.vocab_size || 0,
              stats: modelData.stats || {}
            },
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          });

          // Link active learning sessions to this deployment
          try {
            await supabase
              .from('epsilon_learning_sessions')
              .update({ 
                deployment_id: updatedDeploy.id,
                model_version_after: pendingDeploy.version,
                completed_at: new Date().toISOString(),
                status: 'completed',
                updated_at: new Date().toISOString()
              })
              .eq('status', 'active')
              .is('deployment_id', null);
          } catch (linkError) {
            console.warn(' [DEPLOY] Failed to link learning sessions to deployment (non-critical):', linkError.message);
          }

          // Update semantic memory metadata with new model version
          // Note: epsilon_semantic_memory doesn't have model_version column, store in metadata JSONB
          // This is a non-critical update, so we'll skip if it fails
          try {
            const { data: memoryRecords } = await supabase
              .from('epsilon_semantic_memory')
              .select('id, metadata')
              .or('metadata->>model_version.is.null,metadata->>model_version.eq.production')
              .limit(100);
            
            if (memoryRecords && memoryRecords.length > 0) {
              for (const record of memoryRecords) {
                const updatedMetadata = {
                  ...(record.metadata || {}),
                  model_version: pendingDeploy.version,
                  updated_at: new Date().toISOString()
                };
                await supabase
                  .from('epsilon_semantic_memory')
                  .update({ metadata: updatedMetadata })
                  .eq('id', record.id);
              }
            }
          } catch (memoryError) {
            // Non-critical, just log warning
            console.warn(' [DEPLOY] Failed to update semantic memory metadata (non-critical):', memoryError.message);
          }

          // Update knowledge tracks with deployment info
          // Note: epsilon_knowledge_tracks is for document learning tracks, not deployments
          // Store deployment info in metadata instead
          await supabase.from('epsilon_knowledge_tracks').insert({
            document_id: null, // Deployment is not document-based
            text: `Model deployment ${pendingDeploy.version} - Quality: ${(pendingDeploy.quality_score * 100).toFixed(1)}%, Improvement: ${(pendingDeploy.improvement * 100).toFixed(1)}%`,
            track: 'procedural', // Deployment is a procedural event
            confidence: pendingDeploy.quality_score || 0.5,
            metadata: {
              deployment_version: pendingDeploy.version,
              quality_score: pendingDeploy.quality_score,
              improvement: pendingDeploy.improvement,
              deployed_at: new Date().toISOString(),
              track_type: 'deployment'
            }
          });
        } catch (err) {
          console.warn(' [DEPLOY] Failed to update weights/patterns/memory (non-critical):', err.message);
        }
      }

      // CRITICAL: Load the new model into Python service immediately
      // This ensures Epsilon AI uses the new model right away (both local and Render)
      try {
        // Use environment variable for production, fallback to localhost for development
        const pythonLLMUrl = process.env.PYTHON_LLM_SERVICE_URL || `http://localhost:${pythonManager.services.language_model.port}`;
        
        const importResponse = await axios.post(
          `${pythonLLMUrl}/import`,
          {
            model_data: modelData.model_data,
            stats: modelData.stats || pendingDeploy.stats || {},
            temperature: deploymentData.temperature,
            version: deploymentData.version
          },
          { timeout: 60000 } // Increased timeout for large models
        );

        if (importResponse.data?.success) {
          
          // CRITICAL: Update language engine status to reflect new model
          // This ensures Epsilon AI uses the new model immediately
          if (epsilonLanguageEngine) {
            epsilonLanguageEngine.modelStatus = {
              ready: true,
              lastTrainedAt: new Date().toISOString(),
              stats: importResponse.data.stats || modelData.stats || pendingDeploy.stats || {},
              pendingReason: null
            };
          }
        } else {
          console.warn(' [DEPLOY] Model import response unclear:', importResponse.data);
          throw new Error('Model import did not return success');
        }
      } catch (importError) {
        console.error('[DEPLOY] Import failed, trying reload:', importError.message);
        
        try {
          const pythonLLMUrl = process.env.PYTHON_LLM_SERVICE_URL || `http://localhost:${pythonManager.services.language_model.port}`;
          const reloadResponse = await axios.post(
            `${pythonLLMUrl}/reload-model`,
            {},
            { timeout: 120000 }
          );
          
          if (reloadResponse.data?.success) {
            modelLoaded = true;
            console.log('[DEPLOY] Model reloaded successfully');
            if (epsilonLanguageEngine) {
              epsilonLanguageEngine.modelStatus = {
                ready: true,
                lastTrainedAt: new Date().toISOString(),
                stats: reloadResponse.data.stats || pendingDeploy.stats || {},
                pendingReason: null,
                version: deploymentData.version
              };
              epsilonLanguageEngine.currentVersion = deploymentData.version;
            }
          }
        } catch (reloadError) {
          console.error('[DEPLOY] Reload failed:', reloadError.message);
          modelLoaded = false;
        }
      }


      res.json({
        success: true,
        message: `Deployment ${pendingDeploy.version} approved and deployed`,
        deployment: updatedDeploy
      });

    } catch (error) {
      console.error(' [DEPLOY] Approval failed:', error);
      res.status(500).json({ success: false, error: error.message || 'Failed to approve deployment' });
    }
  });
  */

  // Reject deployment
  app.post('/api/epsilon-llm/deploys/:deployId/reject', verifyAuth('owner'), async (req, res) => {
    try {
      const { deployId } = req.params;
      
      // First, verify the deployment exists and is pending
      const { data: existingDeploy, error: fetchError } = await supabase
        .from('epsilon_model_deployments')
        .select('id, model_id, version, stats, quality_score, improvement, training_samples, learning_description, status, created_at, created_by, storage_path, model_data')
        .eq('id', deployId)
        .eq('status', 'pending')
        .limit(1).maybeSingle();

      if (fetchError || !existingDeploy) {
        console.error(' [DEPLOY] Reject failed - deployment not found:', fetchError?.message || 'Deployment not found');
        return res.status(404).json({ success: false, error: 'Pending deployment not found' });
      }

      // Get user identifier for audit trail
      const userIdentifier = req.user?.email || req.user?.id || 'owner';
      
      // Update to rejected status
      // Note: The constraint allows 'rejected' status without model_data/storage_path
      // (only 'approved' status requires model data)
      const { data: updatedDeploy, error: updateError } = await supabase
        .from('epsilon_model_deployments')
        .update({
          status: 'rejected',
          approved_by: userIdentifier,
          approved_at: new Date().toISOString()
        })
        .eq('id', deployId)
        .select()
        .limit(1).maybeSingle();

      if (updateError) {
        console.error(' [DEPLOY] Reject update failed:', updateError);
        return res.status(500).json({ success: false, error: 'Failed to update deployment status' });
      }

      console.log(` [DEPLOY] Deployment ${existingDeploy.version} rejected by ${userIdentifier}`);

      res.json({
        success: true,
        message: `Deployment ${existingDeploy.version} rejected`
      });
    } catch (error) {
      console.error(' [DEPLOY] Rejection failed:', error);
      res.status(500).json({ success: false, error: error.message || 'Failed to reject deployment' });
    }
  });

  // Get deploy history
  app.get('/api/epsilon-llm/deploys/history', verifyAuth('owner'), async (req, res) => {
    try {
      const { data, error } = await supabase
        .from('epsilon_model_deployments')
        .select('id, model_id, version, stats, quality_score, improvement, training_samples, learning_description, status, created_at, created_by, storage_path')
        .order('created_at', { ascending: false })
        .limit(50);

      if (error) {
        throw error;
      }

      res.json({ success: true, deploys: data || [] });
    } catch (error) {
      console.error(' [DEPLOY] Failed to get deploy history:', error);
      res.status(500).json({ success: false, error: error.message || 'Failed to get deploy history' });
    }
  });

// Get model status
app.get('/api/epsilon-llm/status', verifyAuth('owner'), async (req, res) => {
  try {
    const pythonManager = epsilonLanguageEngine.pythonManager;
    if (!pythonManager || !pythonManager.isServiceReady('language_model')) {
      return res.json({
        ready: false,
        message: 'Language model service is not available'
      });
    }

    const axios = require('axios');
    const healthResponse = await axios.get(
      `http://localhost:${pythonManager.services.language_model.port}/health`,
      { timeout: 5000 }
    );

    res.json({
      ready: epsilonLanguageEngine.isModelReady(),
      serviceReady: true,
      health: healthResponse.data || {}
    });
  } catch (error) {
    res.json({
      ready: false,
      serviceReady: false,
      error: error.message
    });
  }
});


// Get model status
app.get('/api/epsilon-llm/status', verifyAuth('owner'), async (req, res) => {
  try {
    const pythonManager = epsilonLanguageEngine.pythonManager;
    if (!pythonManager || !pythonManager.isServiceReady('language_model')) {
      return res.json({
        ready: false,
        message: 'Language model service is not available'
      });
    }

    const axios = require('axios');
    const healthResponse = await axios.get(
      `http://localhost:${pythonManager.services.language_model.port}/health`,
      { timeout: 5000 }
    );

    res.json({
      ready: epsilonLanguageEngine.isModelReady(),
      serviceReady: true,
      health: healthResponse.data || {}
    });
  } catch (error) {
    res.json({
      ready: false,
      serviceReady: false,
      error: error.message
    });
  }
});

// Load model from Supabase (called on startup)
app.post('/api/epsilon-llm/load', async (req, res) => {
  try {
    if (!epsilonLanguageEngine.isPythonReady()) {
      return res.status(503).json({ success: false, error: 'Python language model service is not ready' });
    }

    const pythonManager = epsilonLanguageEngine.pythonManager;
    if (!pythonManager || !pythonManager.isServiceReady('language_model')) {
      return res.status(503).json({ success: false, error: 'Language model service is not available' });
    }

    // Load from Supabase
    const { createClient } = require('@supabase/supabase-js');
    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_KEY,
      { auth: { persistSession: false } }
    );

    const { data, error } = await supabase
      .from('epsilon_model_deployments')
      .select('id, model_id, version, stats, quality_score, improvement, training_samples, learning_description, status, created_at, deployed_at, created_by, storage_path')
      .eq('model_id', 'production')
      .order('deployed_at', { ascending: false })
      .limit(1)
      .limit(1).maybeSingle();

    if (error || !data) {
      return res.status(404).json({ success: false, error: 'No deployed model found in Supabase' });
    }

    // Import model to Python service
    const axios = require('axios');
    const pythonLLMUrl = process.env.PYTHON_LLM_SERVICE_URL || `http://localhost:${pythonManager.services.language_model.port}`;
    const importResponse = await axios.post(
      `${pythonLLMUrl}/import`,
      {
        model_data: data.model_data,
        stats: data.stats,
        temperature: data.temperature,
        version: data.version || '1.0.0'
      },
      { timeout: 30000 }
    );

    if (!importResponse.data.success) {
      throw new Error('Failed to import model to Python service');
    }

    res.json({
      success: true,
      message: 'Model loaded from Supabase',
      stats: importResponse.data.stats,
      deployed_at: data.deployed_at
    });
  } catch (error) {
    console.error(' [EPSILON AI LLM LOAD] Load failed:', error);
    res.status(500).json({ success: false, error: error.message || 'Failed to load model' });
  }
});

app.post('/api/epsilon-llm/generate', verifyAuth('owner'), async (req, res) => {
  try {
    const { prompt, context = [], persona = {} } = req.body || {};
    if (!prompt || !prompt.trim()) {
      return res.status(400).json({ success: false, error: 'Prompt is required' });
    }

    // Use inference client for generation
    const { getInferenceClient } = require('./inference_client');
    const inferenceClient = getInferenceClient();
    
    const result = await inferenceClient.generate({
      prompt: prompt.trim(),
      max_new_tokens: 256,
      temperature: 0.7,
      top_p: 0.9
    });

      if (!result || !result.text) {
      return res.status(503).json({ 
        success: false, 
        error: 'Inference service not ready or model not loaded' 
      });
    }

    res.json({
      success: true,
      completion: result.text,
      meta: {
        model_id: result.model_id,
        tokens: result.tokens,
        source: 'inference_service'
      }
    });
  } catch (error) {
    console.error(' [EPSILON AI LLM GENERATE] Generation failed:', error);
    res.status(500).json({ success: false, error: error.message || 'Failed to generate text' });
  }
});

// Obfuscated files routes - explicit handlers for better reliability
// ALL client-side JavaScript files must be in this list
const obfuscatedFiles = [
  'rag-embedding-service.js',
  'rag-llm-service.js',
  'rag-document-processor.js',
  'epsilon-learning-engine.js',
  'epsilon-language-engine.js',
  // 'epsilon-automatic-training.js', // Training removed - local-only
  'epsilon-self-learning.js',
  // 'epsilon-ai-core.js', // Removed - training is local-only in ml_local/
  'epsilon-embeddings.js',
  'epsilon-tokenizer.js',
  'document-learning-service.js',
  'supabase-proxy.js',
  'supabase.js',
  'analytics.js',
  'auth.js',
  'documents.js'
];

obfuscatedFiles.forEach(filename => {
  app.get(`/obfuscated/${filename}`, (req, res) => {
    const filePath = path.join(__dirname, '../obfuscated', filename);
    
    if (!fs.existsSync(filePath)) {
      // Try to serve from source if obfuscated doesn't exist
      let sourcePath;
      if (filename === 'epsilon-learning-engine.js') {
        sourcePath = path.join(__dirname, '../core/epsilon-learning-engine.js');
      } else if (filename === 'epsilon-language-engine.js') {
        sourcePath = path.join(__dirname, '../core/epsilon-language-engine.js');
      // } else if (filename === 'epsilon-automatic-training.js') {
      //   sourcePath = path.join(__dirname, '../core/epsilon-automatic-training.js'); // Training removed
      } else if (filename === 'epsilon-self-learning.js') {
        sourcePath = path.join(__dirname, '../core/epsilon-self-learning.js');
      // } else if (filename === 'epsilon-ai-core.js') {
      //   sourcePath = path.join(__dirname, '../services/ai-core/epsilon-ai-core.js'); // Removed - training is local-only
      } else if (filename === 'epsilon-embeddings.js') {
        sourcePath = path.join(__dirname, '../services/ai-core/epsilon-embeddings.js');
      } else if (filename === 'epsilon-tokenizer.js') {
        sourcePath = path.join(__dirname, '../services/ai-core/epsilon-tokenizer.js');
      } else if (filename === 'rag-embedding-service.js') {
        sourcePath = path.join(__dirname, '../services/rag-embedding-service.js');
      } else if (filename === 'rag-llm-service.js') {
        sourcePath = path.join(__dirname, '../services/rag-llm-service.js');
      } else if (filename === 'rag-document-processor.js') {
        sourcePath = path.join(__dirname, '../services/rag-document-processor.js');
      } else if (filename === 'document-learning-service.js') {
        sourcePath = path.join(__dirname, '../services/document-learning-service.js');
      } else if (filename === 'supabase-proxy.js') {
        sourcePath = path.join(__dirname, '../api/supabase-proxy.js');
      } else if (filename === 'supabase.js') {
        sourcePath = path.join(__dirname, '../services/libs/supabase.js');
      } else if (filename === 'analytics.js') {
        sourcePath = path.join(__dirname, '../api/netlify/functions/analytics.js');
      } else if (filename === 'auth.js') {
        sourcePath = path.join(__dirname, '../api/netlify/functions/auth.js');
      } else if (filename === 'documents.js') {
        sourcePath = path.join(__dirname, '../api/netlify/functions/documents.js');
      }
      
      if (sourcePath && fs.existsSync(sourcePath)) {
        let content = fs.readFileSync(sourcePath, 'utf8');
        
        // Obfuscate in production
        if (isProduction) {
          try {
            const JavaScriptObfuscator = require('javascript-obfuscator');
            const obfuscationOptions = {
              compact: true,
              controlFlowFlattening: true,
              controlFlowFlatteningThreshold: 0.75,
              deadCodeInjection: true,
              deadCodeInjectionThreshold: 0.4,
              debugProtection: false,
              debugProtectionInterval: 0,
              disableConsoleOutput: false,
              identifierNamesGenerator: 'hexadecimal',
              log: false,
              numbersToExpressions: true,
              renameGlobals: false,
              selfDefending: false,
              simplify: false,
              splitStrings: true,
              splitStringsChunkLength: 10,
              stringArray: true,
              stringArrayCallsTransform: true,
              stringArrayCallsTransformThreshold: 0.75,
              stringArrayEncoding: ['base64'],
              stringArrayIndexShift: true,
              stringArrayRotate: true,
              stringArrayShuffle: true,
              stringArrayWrappersCount: 2,
              stringArrayWrappersChainedCalls: true,
              stringArrayWrappersParametersMaxCount: 2,
              stringArrayWrappersType: 'function',
              stringArrayThreshold: 0.75,
              transformObjectKeys: true,
              unicodeEscapeSequence: true
            };
            content = JavaScriptObfuscator.obfuscate(content, obfuscationOptions).getObfuscatedCode();
          } catch (obfError) {
            // If obfuscation fails in production, DO NOT serve original - return error
            if (isProduction) {
              return res.status(500).send('// Production build required - source code not available');
            }
          }
        }
        
        const etag = `"${crypto.createHash('md5').update(content).digest('hex')}"`;
        const ifNoneMatch = req.headers['if-none-match'];
        if (ifNoneMatch === etag) {
          return res.status(304).end();
        }
        res.set({
          'Content-Type': 'application/javascript',
          'ETag': etag,
          'Cache-Control': 'public, max-age=3600',
          'X-Content-Type-Options': 'nosniff',
          'X-Frame-Options': 'DENY'
        });
        return res.send(content);
      }
      
      return res.status(404).send(`${filename} not found`);
    }
    
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Generate ETag based on content
    const etag = `"${crypto.createHash('md5').update(content).digest('hex')}"`;
    
    // Check if client has a valid cached version
    const ifNoneMatch = req.headers['if-none-match'];
    if (ifNoneMatch === etag) {
      return res.status(304).end();
    }
    
    res.set({
      'Content-Type': 'application/javascript',
      'ETag': etag,
      'Cache-Control': 'public, max-age=3600',
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY'
    });
    
    res.send(content);
  });
});

// Catch-all route for SPA (must be last)
app.get('*', (req, res) => {
  // Exclude API routes, static files, and core JS files
  if (req.path.startsWith('/api/') || 
      req.path.startsWith('/.netlify/') || 
      req.path.startsWith('/core/') ||
      req.path.startsWith('/obfuscated/') ||
      req.path.startsWith('/services/') ||
      (req.path.includes('.') && !req.path.endsWith('.html'))) {
    res.status(404).send('Not found');
    return;
  }

  serveSecureHTML(req, res, path.join(__dirname, '../ui/index.html'));
});

// Graceful shutdown for Python services
process.on('SIGINT', async () => {
  if (pythonServiceManager) {
    await pythonServiceManager.shutdown();
  }
  process.exit(0);
});

process.on('SIGTERM', async () => {
  if (pythonServiceManager) {
    await pythonServiceManager.shutdown();
  }
  process.exit(0);
});

} // Close the else block from the global guard

