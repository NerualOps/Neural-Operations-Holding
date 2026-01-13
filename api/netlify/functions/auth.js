// Auth handler for Epsilon AI
const jwt = require('jsonwebtoken');
const { createClient } = require('@supabase/supabase-js');
const { logSecurityEvent } = require('../../../runtime/logging');
const { checkLoginAttempts, recordLoginAttempt } = require('../../../runtime/account-security');
const { sanitizeText } = require('../../../runtime/sanitize');
const { associateIPWithAccount, getClientIP } = require('../../../runtime/ip-tracking');

// Initialize Supabase client with no cookie forwarding
// Using service key means we don't need cookies - this prevents "cookie too large" errors
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY,
  {
    auth: {
      persistSession: false,
      autoRefreshToken: false,
      detectSessionInUrl: false
    },
    global: {
      headers: {
        'X-Client-Info': 'neuralops-auth'
      }
    }
  }
);

// Use environment variable for JWT secret
const JWT_SECRET = process.env.JWT_SECRET;
if (!JWT_SECRET) {
  console.error('[SECURITY] JWT_SECRET environment variable is required');
  throw new Error('JWT_SECRET environment variable is required');
}

exports.handler = async (event, context) => {
  console.log('[AUTH SERVER] ========================================');
  console.log('[AUTH SERVER] Auth handler called');
  console.log('[AUTH SERVER] Method:', event.httpMethod);
  console.log('[AUTH SERVER] Path:', event.path);
  console.log('[AUTH SERVER] Has body:', !!event.body);
  console.log('[AUTH SERVER] ========================================');
  
  // Set secure CORS headers
  const headers = {
    'Access-Control-Allow-Origin': process.env.FRONTEND_URL || 'https://neuralops.biz',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, Cookie, X-CSRF-Token',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Credentials': 'true',
    'Content-Type': 'application/json'
  };

  // Handle preflight requests
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers
    };
  }

  // Only allow POST requests
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  try {
    // Parse request body
    const body = JSON.parse(event.body);
    const action = body.action;
    
    console.log('[AUTH SERVER] Action:', action);
    
    // Get client IP for security logging and tracking
    const ip = event.headers['client-ip'] || 
               (event.headers['x-forwarded-for'] ? event.headers['x-forwarded-for'].split(',')[0].trim() : null) ||
               'unknown';
    
    // Log auth attempt
    logSecurityEvent('AUTH_ATTEMPT', {
      action,
      ip: ip
    });
    
    console.log('[AUTH SERVER] Processing action:', action, 'from IP:', ip);

    switch (action) {
      case 'login':
        return await handleLogin(body, headers, ip);
      case 'register':
        return await handleRegister(body, headers, ip);
      case 'logout':
        return await handleLogout(headers);
      case 'verify':
        return await handleVerify(event.headers.cookie, headers);
      case 'refresh_token':
        return await handleRefreshToken(body, headers, ip);
      case 'google_oauth_url':
        return await handleGoogleOAuthUrl(body, headers, ip);
      case 'google_oauth_callback':
        return await handleGoogleOAuthCallback(body, headers, ip);
      case 'apple_oauth_url':
        return await handleAppleOAuthUrl(body, headers, ip);
      case 'apple_oauth_callback':
        return await handleAppleOAuthCallback(body, headers, ip);
      default:
        return {
          statusCode: 400,
          headers,
          body: JSON.stringify({ error: 'Invalid action' })
        };
    }
  } catch (error) {
    console.error('[AUTH SERVER] ========================================');
    console.error('[AUTH SERVER] Auth handler error:', error.message);
    console.error('[AUTH SERVER] Error stack:', error.stack);
    console.error('[AUTH SERVER] ========================================');
    
    logSecurityEvent('AUTH_ERROR', {
      error: error.message,
      ip: event.headers['client-ip'] || event.headers['x-forwarded-for'] || 'unknown'
    }, 'error');
    
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Internal server error' })
    };
  }
};

// Handle login requests
async function handleLogin(body, headers, ip) {
  const { email, password } = body;
  
  console.log('[AUTH SERVER] Login attempt:', { email: email?.substring(0, 10) + '...', ip });
  
  // Validate inputs
  if (!email || !password) {
    console.log('[AUTH SERVER] Login failed: Missing email or password');
    return {
      statusCode: 400,
      headers,
      body: JSON.stringify({ error: 'Email and password are required' })
    };
  }
  
  const sanitizedEmail = sanitizeText(email.toLowerCase());
  
  // Check for too many login attempts
  const lockStatus = checkLoginAttempts(sanitizedEmail);
  if (lockStatus.locked) {
    logSecurityEvent('LOGIN_ATTEMPT_BLOCKED', {
      email: sanitizedEmail,
      ip,
      reason: 'Account locked',
      remainingTime: lockStatus.remainingTime
    }, 'warn');
    
    return {
      statusCode: 429,
      headers,
      body: JSON.stringify({
        error: 'Too many failed login attempts',
        message: `Account is temporarily locked. Try again in ${Math.ceil(lockStatus.remainingTime / 60)} minutes.`,
        remainingTime: lockStatus.remainingTime
      })
    };
  }
  
  try {
    // Try to sign in with Supabase Auth
    console.log('[AUTH SERVER] ========================================');
    console.log('[AUTH SERVER] Attempting Supabase authentication...');
    console.log('[AUTH SERVER] Email:', sanitizedEmail);
    console.log('[AUTH SERVER] Password length:', password.length);
    console.log('[AUTH SERVER] ========================================');
    
    const authStartTime = Date.now();
    const { data: authData, error: authError } = await supabase.auth.signInWithPassword({
      email: sanitizedEmail,
      password
    });
    const authDuration = Date.now() - authStartTime;
    
    console.log('[AUTH SERVER] Supabase auth completed:', {
      duration: authDuration + 'ms',
      hasData: !!authData,
      hasError: !!authError,
      hasUser: !!(authData && authData.user)
    });

    if (authError) {
      console.error('[AUTH SERVER] Auth error:', authError.message);
      console.log('[AUTH SERVER] Login failed for:', sanitizedEmail);
      
      // Record failed login attempt
      const attemptStatus = recordLoginAttempt(sanitizedEmail, false, ip);
      
      if (attemptStatus.locked) {
        logSecurityEvent('ACCOUNT_LOCKED', {
          email: sanitizedEmail,
          ip,
          reason: 'Too many failed attempts'
        }, 'warn');
        
        return {
          statusCode: 429,
          headers,
          body: JSON.stringify({
            error: 'Too many failed login attempts',
            message: `Account is temporarily locked. Try again in ${Math.ceil(attemptStatus.remainingTime / 60)} minutes.`,
            remainingTime: attemptStatus.remainingTime
          })
        };
      }
      
      return {
        statusCode: 401,
        headers,
        body: JSON.stringify({ 
          error: 'Invalid credentials',
          attempts: attemptStatus.attempts,
          remaining: 5 - attemptStatus.attempts
        })
      };
    }

    if (!authData || !authData.user) {
      console.error('[AUTH SERVER] No auth data returned from Supabase');
      console.log('[AUTH SERVER] Login failed: Missing auth data');
      
      // Record failed login attempt
      recordLoginAttempt(sanitizedEmail, false, ip);
      
      return {
        statusCode: 401,
        headers,
        body: JSON.stringify({ error: 'Authentication failed' })
      };
    }

    // Record successful login
    recordLoginAttempt(sanitizedEmail, true, ip);
    logSecurityEvent('LOGIN_SUCCESS', {
      email: sanitizedEmail,
      userId: authData.user.id,
      ip
    });
    
    // Associate IP with user account
    if (ip && ip !== 'unknown') {
      console.log('[AUTH SERVER] Associating IP with account on login:', { ip, userId: authData.user.id, email: sanitizedEmail });
      associateIPWithAccount(ip, authData.user.id, sanitizedEmail).catch(err => {
        console.error('[AUTH SERVER] Error associating IP with account:', err.message);
      });
    }

    console.log('[AUTH SERVER] Login successful, fetching role...', { 
      userId: authData.user.id, 
      email: sanitizedEmail
    });
    
    // Get role and avatar_url from profile or default to client
    let role = 'client';
    let avatarUrl = null;
    console.log('[AUTH SERVER] Fetching user role and avatar from profiles table...');
    console.log('[AUTH SERVER] User ID:', authData.user.id);
    
    // Check user role and avatar from database - no hardcoded emails
    // Role will be determined by database lookup
    try {
      const roleStartTime = Date.now();
      const { data: profileData, error: profileError } = await supabase
        .from('profiles')
        .select('role, avatar_url')
        .eq('id', authData.user.id)
        .limit(1)
        .maybeSingle();
      const roleDuration = Date.now() - roleStartTime;
      
      console.log('[AUTH SERVER] Role and avatar query completed:', {
        duration: roleDuration + 'ms',
        hasData: !!profileData,
        hasError: !!profileError,
        role: profileData?.role,
        avatarUrl: profileData?.avatar_url ? 'exists' : 'missing'
      });
        
      if (!profileError && profileData) {
        if (profileData.role) {
          role = profileData.role;
          console.log('[AUTH SERVER] Role from database:', role);
        }
        if (profileData.avatar_url) {
          avatarUrl = profileData.avatar_url;
          console.log('[AUTH SERVER] Avatar URL from database: exists');
        }
      } else if (profileError) {
        console.log('[AUTH SERVER] Profile query error (using default role):', profileError.message);
        console.log('[AUTH SERVER] Using default role: client');
      } else {
        console.log('[AUTH SERVER] No role in profile, using default: client');
      }
    } catch (profileError) {
      console.error('[AUTH SERVER] Profile query exception:', profileError.message);
      console.log('[AUTH SERVER] Using default role: client');
    }
    
    console.log('[AUTH SERVER] Final role determined:', role);
    console.log('[AUTH SERVER] Final avatar URL:', avatarUrl ? 'exists' : 'missing');
    
    // Silent - no console.log
    
    // Create JWT with appropriate role
    console.log('[AUTH SERVER] Creating JWT token...');
    const tokenPayload = { 
      userId: authData.user.id, 
      email: authData.user.email, 
      role: role,
      iat: Math.floor(Date.now() / 1000)
    };
    console.log('[AUTH SERVER] Token payload:', {
      userId: tokenPayload.userId,
      email: tokenPayload.email,
      role: tokenPayload.role
    });
    
    const token = jwt.sign(
      tokenPayload,
      JWT_SECRET,
      { expiresIn: '7d' }
    );
    console.log('[AUTH SERVER] JWT token created, length:', token.length);

    // Create secure HTTP-only cookie with proper security flags
    const cookie = `authToken=${token}; HttpOnly; Secure; SameSite=Strict; Max-Age=${7 * 24 * 60 * 60}; Path=/`;
    console.log('[AUTH SERVER] Cookie string created, length:', cookie.length);
    
    // Try to get user profile data
    console.log('[AUTH SERVER] Building user data object...');
    let userData = {
      id: authData.user.id,
      email: authData.user.email,
      name: sanitizedEmail.split('@')[0], // Default name from email
      role: role,
      company: role === 'owner' ? 'Neural Ops' : ''
    };
    console.log('[AUTH SERVER] Initial user data:', userData);
    
    try {
      console.log('[AUTH SERVER] Fetching full profile data...');
      const profileStartTime = Date.now();
      const { data: profileData, error: profileError } = await supabase
        .from('profiles')
        .select('name, company, industry, role, avatar_url')
        .eq('id', authData.user.id)
        .limit(1)
        .maybeSingle();
      const profileDuration = Date.now() - profileStartTime;
      
      console.log('[AUTH SERVER] Profile query completed:', {
        duration: profileDuration + 'ms',
        hasData: !!profileData,
        hasError: !!profileError
      });
        
      if (!profileError && profileData) {
        console.log('[AUTH SERVER] Profile data found:', {
          name: profileData.name,
          company: profileData.company,
          industry: profileData.industry
        });
        userData = {
          ...userData,
          name: profileData.name || userData.name,
          company: profileData.company || userData.company,
          industry: profileData.industry || '',
          role: profileData.role || role || 'client', // Use profile role, fallback to query role, then default
          avatar_url: profileData.avatar_url || avatarUrl || null, // Use profile avatar, fallback to query avatar
          avatar: profileData.avatar_url || avatarUrl || null
        };
        console.log('[AUTH SERVER] Updated user data:', userData);
      } else if (profileError) {
        console.log('[AUTH SERVER] Profile query error (using defaults):', profileError.message);
        console.log('[AUTH SERVER] Using default user data');
      } else {
        console.log('[AUTH SERVER] No profile data found, using defaults');
      }
    } catch (profileError) {
      console.error('[AUTH SERVER] Profile data query exception:', profileError.message);
      console.error('[AUTH SERVER] Stack:', profileError.stack);
    }
    
    console.log('[AUTH SERVER] Final user data to return:', userData);
    
    // Update last login timestamp
    try {
      const { error: updateError } = await supabase
        .from('profiles')
        .update({ last_login: new Date().toISOString() })
        .eq('id', authData.user.id);
        
      if (updateError) {
        // Silent - no console.log
        // Silent - no console.log
      }
    } catch (updateError) {
      console.error('Last login update exception:', updateError.message);
      // Silent - no console.log
    }
    
    console.log('[AUTH SERVER] Returning login response:', { 
      userId: userData.id, 
      email: userData.email,
      role: userData.role,
      hasCookie: !!cookie 
    });
    
    return {
      statusCode: 200,
      headers: {
        ...headers,
        'Set-Cookie': cookie
      },
      body: JSON.stringify({
        success: true,
        token: token,
        user: userData
      })
    };
  } catch (error) {
    console.error('[AUTH SERVER] Login error:', error.message);
    console.error('[AUTH SERVER] Login error stack:', error.stack);
    
    logSecurityEvent('LOGIN_ERROR', {
      email: sanitizedEmail,
      error: error.message,
      ip
    }, 'error');
    
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Login failed. Please try again.' })
    };
  }
}

// Handle registration requests
async function handleRegister(body, headers, ip) {
  const { email, password, name, company, industry } = body;
  
  // Validate inputs
  if (!email || !password || !name) {
    return {
      statusCode: 400,
      headers,
      body: JSON.stringify({ error: 'Email, password, and name are required' })
    };
  }
  
  const sanitizedEmail = sanitizeText(email.toLowerCase());
  const sanitizedName = sanitizeText(name);
  
  // Validate email format
  if (!isValidEmail(sanitizedEmail)) {
    return {
      statusCode: 400,
      headers,
      body: JSON.stringify({ error: 'Invalid email format' })
    };
  }
  
  // Validate password strength
  if (!isStrongPassword(password)) {
    return {
      statusCode: 400,
      headers,
      body: JSON.stringify({ 
        error: 'Password must be at least 8 characters and include uppercase, lowercase, and numbers' 
      })
    };
  }
  
  try {
    // Get role from profile or default to client
    let role = 'client';
    
    // Role is determined by database lookup - no hardcoded emails
    // All new registrations default to 'client' role
    
    // Create user with Supabase Auth
    const { data: authData, error: authError } = await supabase.auth.admin.createUser({
      email: sanitizedEmail,
      password,
      email_confirm: true
    });

    if (authError) {
      console.error('Auth error:', authError);
      logSecurityEvent('REGISTRATION_FAILED', { email: sanitizedEmail, ip, error: authError.message }, 'warn');
      // Don't leak detailed error messages to users
      const userMessage = authError.message.includes('already registered') || authError.message.includes('already exists')
        ? 'An account with this email already exists'
        : 'Failed to create user account';
      return {
        statusCode: 500,
        headers,
        body: JSON.stringify({ error: userMessage })
      };
    }

    if (!authData || !authData.user) {
      console.error('No user data returned from auth');
      logSecurityEvent('REGISTRATION_FAILED', { email: sanitizedEmail, ip, error: 'No user data returned' }, 'warn');
      return {
        statusCode: 500,
        headers,
        body: JSON.stringify({ error: 'Failed to create user account' })
      };
    }

    // Silent - no console.log
    // Silent - no console.log

    // Try to create profile
    try {
      const { data: profileData, error: profileInsertError } = await supabase
        .from('profiles')
        .insert([{
          id: authData.user.id,
          email: sanitizedEmail.trim(),
          name: sanitizedName,
          company: sanitizeText(company),
          industry: sanitizeText(industry),
          role: role,
          created_at: new Date().toISOString()
        }]);

      if (profileInsertError) {
        console.error('Profile creation error:', profileInsertError);
      } else {
        // Silent - no console.log
      }
    } catch (profileError) {
      console.error('Error creating profile:', profileError);
    }

    // Associate IP with user account on registration
    if (ip && ip !== 'unknown') {
      console.log('[AUTH SERVER] Associating IP with account on registration:', { ip, userId: authData.user.id, email: sanitizedEmail });
      associateIPWithAccount(ip, authData.user.id, sanitizedEmail).catch(err => {
        console.error('[AUTH SERVER] Error associating IP with account:', err.message);
      });
    }

    logSecurityEvent('REGISTRATION_SUCCESS', { email: sanitizedEmail, userId: authData.user.id, ip });
    
    // Create JWT token
    const token = jwt.sign(
      { 
        userId: authData.user.id, 
        email: authData.user.email,
        role: role,
        name: sanitizedName
      }, 
      JWT_SECRET,
      { expiresIn: '7d' }
    );
    
    // Set auth cookie
    const cookie = `authToken=${token}; HttpOnly; Secure; SameSite=Strict; Max-Age=${7 * 24 * 60 * 60}; Path=/`;
    
    return {
      statusCode: 201,
      headers: {
        ...headers,
        'Set-Cookie': cookie
      },
      body: JSON.stringify({
        success: true,
        user: {
          id: authData.user.id,
          email: authData.user.email,
          name: sanitizedName,
          role: role,
          created_at: new Date().toISOString()
        },
        token
      })
    };
  } catch (error) {
    console.error('Registration error:', error);
    logSecurityEvent('REGISTRATION_ERROR', { email: sanitizedEmail, ip, error: error.message }, 'error');
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Registration failed. Please try again.' })
    };
  }
}

// Handle logout requests
async function handleLogout(headers) {
  // Clear auth cookie
  const cookie = `authToken=; HttpOnly; Secure; SameSite=Strict; Max-Age=0; Path=/`;
  
  return {
    statusCode: 200,
    headers: {
      ...headers,
      'Set-Cookie': cookie
    },
    body: JSON.stringify({ success: true })
  };
}

// Handle token verification
async function handleVerify(cookieHeader, headers) {
  try {
    console.log('[AUTH SERVER] Verify request received');
    
    // Get token from cookie
    const cookies = parseCookies(cookieHeader || '');
    const token = cookies.authToken;
    
    if (!token) {
      console.log('[AUTH SERVER] Verify failed: No token in cookie');
      return {
        statusCode: 401,
        headers,
        body: JSON.stringify({ error: 'No authentication token found' })
      };
    }
    
    console.log('[AUTH SERVER] Token found, verifying...');
    
    // Verify token
    const decoded = jwt.verify(token, JWT_SECRET);
    console.log('[AUTH SERVER] Token verified:', { userId: decoded.userId, email: decoded.email, role: decoded.role });
    
    // Check if user still exists
    const { data: user, error } = await supabase
      .from('profiles')
      .select('id, email, name, role')
      .eq('id', decoded.userId)
      .limit(1)
      .maybeSingle();
    
    if (error || !user) {
      // Profile doesn't exist - try to create it
      console.log('[AUTH SERVER] Profile not found for user:', decoded.userId, 'Attempting to create profile');
      
      try {
        const { data: newProfile, error: insertError } = await supabase
          .from('profiles')
          .insert([{
            id: decoded.userId,
            email: decoded.email || '',
            name: decoded.name || decoded.email?.split('@')[0] || 'User',
            role: decoded.role || 'client',
            created_at: new Date().toISOString()
          }])
          .select()
          .limit(1)
          .maybeSingle();
          
        if (insertError) {
          console.error('[AUTH SERVER] Profile creation failed:', insertError);
          throw new Error(`Failed to create profile: ${insertError.message}`);
        }
        
        if (!newProfile) {
          throw new Error('Profile creation returned no data');
        }
        
        console.log('[AUTH SERVER] Profile created successfully');
        return {
          statusCode: 200,
          headers,
          body: JSON.stringify({
            valid: true,
            user: newProfile,
            token
          })
        };
      } catch (insertError) {
        console.error('[AUTH SERVER] Failed to create profile:', insertError);
        return {
          statusCode: 500,
          headers,
          body: JSON.stringify({
            error: 'Profile verification failed',
            message: insertError.message || 'Failed to create or retrieve user profile'
          })
        };
      }
    }
    
    console.log('[AUTH SERVER] Profile found:', { userId: user.id, email: user.email, role: user.role });
    
    // Check if role has changed
    if (user.role !== decoded.role) {
      // Create new token with updated role
      const newToken = jwt.sign(
        { 
          userId: user.id, 
          email: user.email,
          role: user.role,
          name: user.name
        }, 
        JWT_SECRET,
        { expiresIn: '7d' }
      );
      
      // Set new auth cookie
      const cookie = `authToken=${newToken}; HttpOnly; Secure; SameSite=Strict; Max-Age=${7 * 24 * 60 * 60}; Path=/`;
      
      return {
        statusCode: 200,
        headers: {
          ...headers,
          'Set-Cookie': cookie
        },
        body: JSON.stringify({
          valid: true,
          user,
          token: newToken,
          roleUpdated: true
        })
      };
    }
    
    console.log('[AUTH SERVER] Verification successful, returning user data');
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        valid: true,
        user,
        token
      })
    };
  } catch (error) {
    console.error('[AUTH SERVER] Token verification error:', error.message, error.name);
    
    // If token is expired or invalid
    if (error.name === 'JsonWebTokenError' || error.name === 'TokenExpiredError') {
      return {
        statusCode: 401,
        headers,
        body: JSON.stringify({ 
          error: 'Invalid or expired token',
          expired: error.name === 'TokenExpiredError'
        })
      };
    }
    
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Verification failed. Please try again.' })
    };
  }
}

// Handle refresh token requests
async function handleRefreshToken(body, headers, ip) {
  try {
    const { user_id, email, role } = body;
    
    if (!user_id || !email) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'User ID and email are required' })
      };
    }
    
    // Validate role if provided
    const validRole = role || 'client';
    if (!['client', 'owner'].includes(validRole)) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'Invalid role' })
      };
    }
    
    // Silent - no console.log
    
    // Create new JWT token with updated role
    const token = jwt.sign(
      { 
        userId: user_id,
        email: email,
        role: validRole
      }, 
      JWT_SECRET,
      { expiresIn: '7d' }
    );
    
    // Set auth cookie
    const cookie = `authToken=${token}; HttpOnly; Secure; SameSite=Strict; Max-Age=${7 * 24 * 60 * 60}; Path=/`;
    
    logSecurityEvent('TOKEN_REFRESHED', { 
      userId: user_id, 
      email: email,
      role: validRole,
      ip 
    });
    
    return {
      statusCode: 200,
      headers: {
        ...headers,
        'Set-Cookie': cookie
      },
      body: JSON.stringify({
        success: true,
        token: token,
        role: validRole
      })
    };
  } catch (error) {
    console.error('Token refresh error:', error);
    
    logSecurityEvent('TOKEN_REFRESH_ERROR', {
      error: error.message,
      ip
    }, 'error');
    
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to refresh token' })
    };
  }
}

// Helper functions
function isValidEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

function isStrongPassword(password) {
  // At least 8 characters, 1 uppercase, 1 lowercase, 1 number
  return password.length >= 8 && 
         /[A-Z]/.test(password) && 
         /[a-z]/.test(password) && 
         /[0-9]/.test(password);
}

function parseCookies(cookieHeader) {
  const cookies = {};
  if (!cookieHeader) return cookies;
  
  cookieHeader.split(';').forEach(cookie => {
    const parts = cookie.split('=');
    const name = parts[0].trim();
    const value = parts.slice(1).join('=').trim();
    if (name) cookies[name] = value;
  });
  
  return cookies;
}

// Handle Google OAuth URL generation
async function handleGoogleOAuthUrl(body, headers, ip) {
  try {
    const { redirectTo } = body;
    const frontendUrl = process.env.FRONTEND_URL || 'https://neuralops.biz';
    const callbackUrl = redirectTo || `${frontendUrl}/auth/callback`;
    
    console.log('[AUTH SERVER] Generating Google OAuth URL');
    
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: callbackUrl,
        queryParams: {
          access_type: 'offline',
          prompt: 'consent'
        }
      }
    });
    
    if (error) {
      console.error('[AUTH SERVER] Google OAuth URL error:', error.message);
      logSecurityEvent('GOOGLE_OAUTH_ERROR', { error: error.message, ip }, 'error');
      return {
        statusCode: 500,
        headers,
        body: JSON.stringify({ error: 'Failed to generate OAuth URL', details: error.message })
      };
    }
    
    if (!data || !data.url) {
      console.error('[AUTH SERVER] No OAuth URL returned');
      return {
        statusCode: 500,
        headers,
        body: JSON.stringify({ error: 'Failed to generate OAuth URL' })
      };
    }
    
    logSecurityEvent('GOOGLE_OAUTH_INITIATED', { ip });
    
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ 
        success: true,
        url: data.url 
      })
    };
  } catch (error) {
    console.error('[AUTH SERVER] Google OAuth URL handler error:', error.message);
    logSecurityEvent('GOOGLE_OAUTH_ERROR', { error: error.message, ip }, 'error');
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Internal server error' })
    };
  }
}

// Handle Google OAuth callback (code exchange)
async function handleGoogleOAuthCallback(body, headers, ip) {
  try {
    const { code } = body;
    
    if (!code) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'Authorization code is required' })
      };
    }
    
    console.log('[AUTH SERVER] Exchanging Google OAuth code for session');
    
    // Exchange code for session
    const { data, error } = await supabase.auth.exchangeCodeForSession(code);
    
    if (error) {
      console.error('[AUTH SERVER] Code exchange error:', error.message);
      logSecurityEvent('GOOGLE_OAUTH_ERROR', { error: error.message, ip }, 'error');
      return {
        statusCode: 401,
        headers,
        body: JSON.stringify({ error: 'Failed to exchange code for session', details: error.message })
      };
    }
    
    if (!data || !data.session || !data.user) {
      console.error('[AUTH SERVER] No session or user returned from code exchange');
      return {
        statusCode: 401,
        headers,
        body: JSON.stringify({ error: 'Failed to create session' })
      };
    }
    
    const { user, session } = data;
    
    // Get or create user profile
    let profile = null;
    try {
      const { data: profileData, error: profileError } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', user.id)
        .maybeSingle();
      
      if (profileError && profileError.code !== 'PGRST116') {
        console.error('[AUTH SERVER] Profile lookup error:', profileError.message);
      }
      
      profile = profileData;
      
      // If no profile exists, create one
      if (!profile) {
        const { data: newProfile, error: createError } = await supabase
          .from('profiles')
          .insert([{
            id: user.id,
            email: user.email,
            full_name: user.user_metadata?.full_name || user.user_metadata?.name || user.email?.split('@')[0] || 'User',
            role: 'client',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          }])
          .select()
          .single();
        
        if (createError) {
          console.error('[AUTH SERVER] Profile creation error:', createError.message);
        } else {
          profile = newProfile;
        }
      }
    } catch (profileErr) {
      console.error('[AUTH SERVER] Profile handling error:', profileErr.message);
    }
    
    // Generate JWT token for the client
    const token = jwt.sign(
      {
        userId: user.id,
        email: user.email,
        role: profile?.role || 'client'
      },
      JWT_SECRET,
      { expiresIn: '7d' }
    );
    
    logSecurityEvent('GOOGLE_OAUTH_SUCCESS', { 
      userId: user.id, 
      email: user.email, 
      ip 
    });
    
    return {
      statusCode: 200,
      headers: {
        ...headers,
        'Set-Cookie': `neuralops_token=${token}; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=604800`
      },
      body: JSON.stringify({
        success: true,
        user: {
          id: user.id,
          email: user.email,
          name: profile?.full_name || user.user_metadata?.full_name || user.email?.split('@')[0],
          role: profile?.role || 'client'
        },
        session: {
          access_token: session.access_token,
          refresh_token: session.refresh_token,
          expires_at: session.expires_at
        }
      })
    };
  } catch (error) {
    console.error('[AUTH SERVER] Google OAuth callback error:', error.message);
    logSecurityEvent('GOOGLE_OAUTH_ERROR', { error: error.message, ip }, 'error');
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Internal server error' })
    };
  }
}
// Handle Apple OAuth URL generation (ready for future implementation)
async function handleAppleOAuthUrl(body, headers, ip) {
  try {
    const { redirectTo } = body;
    const frontendUrl = process.env.FRONTEND_URL || 'https://neuralops.biz';
    const callbackUrl = redirectTo || `${frontendUrl}/auth/callback`;
    
    console.log('[AUTH SERVER] Generating Apple OAuth URL');
    
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: 'apple',
      options: {
        redirectTo: callbackUrl
      }
    });
    
    if (error) {
      console.error('[AUTH SERVER] Apple OAuth URL error:', error.message);
      logSecurityEvent('APPLE_OAUTH_ERROR', { error: error.message, ip }, 'error');
      return {
        statusCode: 500,
        headers,
        body: JSON.stringify({ error: 'Failed to generate OAuth URL', details: error.message })
      };
    }
    
    if (!data || !data.url) {
      console.error('[AUTH SERVER] No OAuth URL returned');
      return {
        statusCode: 500,
        headers,
        body: JSON.stringify({ error: 'Failed to generate OAuth URL' })
      };
    }
    
    logSecurityEvent('APPLE_OAUTH_INITIATED', { ip });
    
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ 
        success: true,
        url: data.url 
      })
    };
  } catch (error) {
    console.error('[AUTH SERVER] Apple OAuth URL handler error:', error.message);
    logSecurityEvent('APPLE_OAUTH_ERROR', { error: error.message, ip }, 'error');
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Internal server error' })
    };
  }
}

// Handle Apple OAuth callback (code exchange) - ready for future implementation
async function handleAppleOAuthCallback(body, headers, ip) {
  try {
    const { code } = body;
    
    if (!code) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'Authorization code is required' })
      };
    }
    
    console.log('[AUTH SERVER] Exchanging Apple OAuth code for session');
    
    // Exchange code for session
    const { data, error } = await supabase.auth.exchangeCodeForSession(code);
    
    if (error) {
      console.error('[AUTH SERVER] Code exchange error:', error.message);
      logSecurityEvent('APPLE_OAUTH_ERROR', { error: error.message, ip }, 'error');
      return {
        statusCode: 401,
        headers,
        body: JSON.stringify({ error: 'Failed to exchange code for session', details: error.message })
      };
    }
    
    if (!data || !data.session || !data.user) {
      console.error('[AUTH SERVER] No session or user returned from code exchange');
      return {
        statusCode: 401,
        headers,
        body: JSON.stringify({ error: 'Failed to create session' })
      };
    }
    
    const { user, session } = data;
    
    // Get or create user profile
    let profile = null;
    try {
      const { data: profileData, error: profileError } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', user.id)
        .maybeSingle();
      
      if (profileError && profileError.code !== 'PGRST116') {
        console.error('[AUTH SERVER] Profile lookup error:', profileError.message);
      }
      
      profile = profileData;
      
      // If no profile exists, create one
      if (!profile) {
        const { data: newProfile, error: createError } = await supabase
          .from('profiles')
          .insert([{
            id: user.id,
            email: user.email,
            full_name: user.user_metadata?.full_name || user.user_metadata?.name || user.email?.split('@')[0] || 'User',
            role: 'client',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          }])
          .select()
          .single();
        
        if (createError) {
          console.error('[AUTH SERVER] Profile creation error:', createError.message);
        } else {
          profile = newProfile;
        }
      }
    } catch (profileErr) {
      console.error('[AUTH SERVER] Profile handling error:', profileErr.message);
    }
    
    // Generate JWT token for the client
    const token = jwt.sign(
      {
        userId: user.id,
        email: user.email,
        role: profile?.role || 'client'
      },
      JWT_SECRET,
      { expiresIn: '7d' }
    );
    
    logSecurityEvent('APPLE_OAUTH_SUCCESS', { 
      userId: user.id, 
      email: user.email, 
      ip 
    });
    
    return {
      statusCode: 200,
      headers: {
        ...headers,
        'Set-Cookie': `neuralops_token=${token}; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=604800`
      },
      body: JSON.stringify({
        success: true,
        user: {
          id: user.id,
          email: user.email,
          name: profile?.full_name || user.user_metadata?.full_name || user.email?.split('@')[0],
          role: profile?.role || 'client'
        },
        session: {
          access_token: session.access_token,
          refresh_token: session.refresh_token,
          expires_at: session.expires_at
        }
      })
    };
  } catch (error) {
    console.error('[AUTH SERVER] Apple OAuth callback error:', error.message);
    logSecurityEvent('APPLE_OAUTH_ERROR', { error: error.message, ip }, 'error');
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Internal server error' })
    };
  }
}

