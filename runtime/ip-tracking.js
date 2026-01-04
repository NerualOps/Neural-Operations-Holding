/**
 * IP Tracking and Guest Usage Management
 * Tracks all visitors by IP and enforces usage limits
 * © 2025 Neural Operation's & Holding's LLC. All rights reserved.
 */

const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY,
  { auth: { persistSession: false } }
);

// Guest usage limits
const GUEST_MESSAGE_LIMIT = 20; // Allow 20 messages before requiring account
const GUEST_COOLDOWN_HOURS = 24; // 24-hour cooldown after limit reached

/**
 * Extract IP address from request
 */
function getClientIP(req) {
  // Check various headers for IP (handles proxies/load balancers)
  const forwarded = req.headers['x-forwarded-for'];
  if (forwarded) {
    // x-forwarded-for can contain multiple IPs, take the first one
    return forwarded.split(',')[0].trim();
  }
  
  const realIP = req.headers['x-real-ip'];
  if (realIP) {
    return realIP;
  }
  
  // Fallback to connection IP
  return req.ip || 
         req.connection?.remoteAddress || 
         req.socket?.remoteAddress ||
         '127.0.0.1';
}

/**
 * Track visitor IP
 */
async function trackVisitorIP(req) {
  try {
    const ip = getClientIP(req);
    const userAgent = req.headers['user-agent'] || 'Unknown';
    const timestamp = new Date().toISOString();
    
    // Silent tracking - only log errors, not every visit
    // First try to get existing record
    const { data: existing, error: selectError } = await supabase
      .from('visitor_ips')
      .select('visit_count')
      .eq('ip_address', ip)
      .limit(1).maybeSingle();
    
    // Handle Supabase connection errors (Cloudflare 522, etc.)
    if (selectError && selectError.message && selectError.message.includes('<!DOCTYPE html>')) {
      // Silently fail on connection issues - don't spam logs
      return ip;
    }
    
    const visitCount = existing?.visit_count ? existing.visit_count + 1 : 1;
    
    // Upsert visitor record
    const { data, error } = await supabase
      .from('visitor_ips')
      .upsert({
        ip_address: ip,
        user_agent: userAgent,
        last_seen: timestamp,
        visit_count: visitCount,
        updated_at: timestamp
      }, {
        onConflict: 'ip_address'
      })
      .select()
      .limit(1).maybeSingle();
    
    if (error) {
      // Handle HTML error responses (Cloudflare issues)
      const errorStr = error?.message || '';
      if (!errorStr.includes('<!DOCTYPE html>')) {
        console.error('[IP TRACKING] Error tracking visitor:', error.message);
      }
      // Don't throw - continue even if tracking fails
    }
    // Silent on success - don't spam logs
    
    return ip;
  } catch (error) {
    // Silent - don't spam logs with tracking exceptions
    return getClientIP(req); // Return IP even if tracking fails
  }
}

/**
 * Check guest usage limits for IP
 */
async function checkGuestUsage(ip) {
  try {
    console.log('[IP TRACKING] Checking guest usage for IP:', ip);
    
    // Check guest usage tracking
    const { data: usageData, error: usageError } = await supabase
      .from('guest_usage')
      .select('id, ip_address, messages_used, last_used, created_at')
      .eq('ip_address', ip)
      .limit(1).maybeSingle();
    
    if (usageError && usageError.code !== 'PGRST116') {
      console.error('[IP TRACKING] Error checking guest usage:', usageError.message);
      // On error, allow access (fail open)
      return { allowed: true, reason: 'error', messagesUsed: 0, messagesRemaining: GUEST_MESSAGE_LIMIT };
    }
    
    // If no record exists, create one
    if (!usageData) {
      console.log('[IP TRACKING] New guest user, creating usage record');
      const { data: newData, error: createError } = await supabase
        .from('guest_usage')
        .insert({
          ip_address: ip,
          messages_used: 0,
          first_used: new Date().toISOString(),
          last_used: new Date().toISOString(),
          cooldown_until: null
        })
        .select()
        .limit(1).maybeSingle();
      
      if (createError) {
        console.error('[IP TRACKING] Error creating guest usage record:', createError.message);
        return { allowed: true, reason: 'error', messagesUsed: 0, messagesRemaining: GUEST_MESSAGE_LIMIT };
      }
      
      return {
        allowed: true,
        reason: 'new_user',
        messagesUsed: 0,
        messagesRemaining: GUEST_MESSAGE_LIMIT,
        cooldownUntil: null
      };
    }
    
    // Check if in cooldown period
    if (usageData.cooldown_until) {
      const cooldownUntil = new Date(usageData.cooldown_until);
      const now = new Date();
      
      if (now < cooldownUntil) {
        const hoursRemaining = Math.ceil((cooldownUntil - now) / (1000 * 60 * 60));
        console.log('[IP TRACKING] Guest in cooldown period:', { hoursRemaining, cooldownUntil });
        
        return {
          allowed: false,
          reason: 'cooldown',
          messagesUsed: usageData.messages_used,
          messagesRemaining: 0,
          cooldownUntil: usageData.cooldown_until,
          hoursRemaining: hoursRemaining
        };
      } else {
        // Cooldown expired, reset usage
        console.log('[IP TRACKING] Cooldown expired, resetting usage');
        await supabase
          .from('guest_usage')
          .update({
            messages_used: 0,
            cooldown_until: null,
            last_used: new Date().toISOString()
          })
          .eq('ip_address', ip);
        
        return {
          allowed: true,
          reason: 'cooldown_expired',
          messagesUsed: 0,
          messagesRemaining: GUEST_MESSAGE_LIMIT,
          cooldownUntil: null
        };
      }
    }
    
    // Check if limit reached
    if (usageData.messages_used >= GUEST_MESSAGE_LIMIT) {
      // Set cooldown period
      const cooldownUntil = new Date();
      cooldownUntil.setHours(cooldownUntil.getHours() + GUEST_COOLDOWN_HOURS);
      
      console.log('[IP TRACKING] Guest limit reached, setting cooldown:', { cooldownUntil });
      
      await supabase
        .from('guest_usage')
        .update({
          cooldown_until: cooldownUntil.toISOString(),
          last_used: new Date().toISOString()
        })
        .eq('ip_address', ip);
      
      return {
        allowed: false,
        reason: 'limit_reached',
        messagesUsed: usageData.messages_used,
        messagesRemaining: 0,
        cooldownUntil: cooldownUntil.toISOString(),
        hoursRemaining: GUEST_COOLDOWN_HOURS
      };
    }
    
    // Usage is within limits
    const remaining = GUEST_MESSAGE_LIMIT - usageData.messages_used;
    console.log('[IP TRACKING] Guest usage OK:', { messagesUsed: usageData.messages_used, remaining });
    
    return {
      allowed: true,
      reason: 'within_limit',
      messagesUsed: usageData.messages_used,
      messagesRemaining: remaining,
      cooldownUntil: null
    };
    
  } catch (error) {
    console.error('[IP TRACKING] Exception checking guest usage:', error.message);
    // Fail open - allow access on error
    return { allowed: true, reason: 'error', messagesUsed: 0, messagesRemaining: GUEST_MESSAGE_LIMIT };
  }
}

/**
 * Increment guest usage for IP
 */
async function incrementGuestUsage(ip) {
  try {
    console.log('[IP TRACKING] Incrementing guest usage for IP:', ip);
    
    // Get current usage count with proper null handling
    const { data: current, error: selectError } = await supabase
      .from('guest_usage')
      .select('messages_used')
      .eq('ip_address', ip)
      .limit(1).maybeSingle();
    
    // Handle Supabase connection errors (Cloudflare 522, etc.)
    if (selectError && selectError.message && selectError.message.includes('<!DOCTYPE html>')) {
      console.warn('[IP TRACKING] Supabase connection issue detected, skipping usage increment');
      return; // Fail silently on connection issues
    }
    
    // Calculate new count - handle null/undefined safely
    const currentCount = current && typeof current.messages_used === 'number' ? current.messages_used : 0;
    const newCount = currentCount + 1;
    
    const { data, error } = await supabase
      .from('guest_usage')
      .update({
        messages_used: newCount,
        last_used: new Date().toISOString()
      })
      .eq('ip_address', ip)
      .select()
      .limit(1).maybeSingle();
    
    if (error) {
      console.error('[IP TRACKING] Error incrementing guest usage:', error.message);
      // Try to create record if it doesn't exist
      const { data: newData, error: createError } = await supabase
        .from('guest_usage')
        .insert({
          ip_address: ip,
          messages_used: 1,
          first_used: new Date().toISOString(),
          last_used: new Date().toISOString()
        })
        .select()
        .limit(1).maybeSingle();
      
      if (createError) {
        console.error('[IP TRACKING] Error creating guest usage record:', createError.message);
      } else {
        console.log('[IP TRACKING] Created new guest usage record');
      }
    } else if (data && data.messages_used) {
      console.log('[IP TRACKING] Guest usage incremented:', { messagesUsed: data.messages_used });
    }
  } catch (error) {
    console.error('[IP TRACKING] Exception incrementing guest usage:', error.message);
    // Fail silently - don't break the request
  }
}

/**
 * Associate IP with user account on signup/login
 */
async function associateIPWithAccount(ip, userId, email) {
  try {
    console.log('[IP TRACKING] Associating IP with account:', { ip, userId, email });
    
    // Update visitor_ips table
    const { error: visitorError } = await supabase
      .from('visitor_ips')
      .update({
        user_id: userId,
        user_email: email,
        associated_at: new Date().toISOString()
      })
      .eq('ip_address', ip);
    
    if (visitorError) {
      console.error('[IP TRACKING] Error associating IP with account (visitor_ips):', visitorError.message);
    } else {
      console.log('[IP TRACKING] IP associated with account in visitor_ips');
    }
    
    // Update guest_usage table
    const { error: guestError } = await supabase
      .from('guest_usage')
      .update({
        user_id: userId,
        user_email: email,
        associated_at: new Date().toISOString()
      })
      .eq('ip_address', ip);
    
    if (guestError && guestError.code !== 'PGRST116') {
      console.error('[IP TRACKING] Error associating IP with account (guest_usage):', guestError.message);
    } else {
      console.log('[IP TRACKING] IP associated with account in guest_usage');
    }
    
    // Also check if user has other IPs and associate them
    const { data: userIPs, error: userIPsError } = await supabase
      .from('visitor_ips')
      .select('ip_address')
      .eq('user_id', userId);
    
    if (!userIPsError && userIPsData) {
      console.log('[IP TRACKING] User has', userIPs.length, 'associated IPs');
    }
    
  } catch (error) {
    console.error('[IP TRACKING] Exception associating IP with account:', error.message);
  }
}

/**
 * Check if IP is associated with any account
 */
async function isIPAssociated(ip) {
  try {
    const { data, error } = await supabase
      .from('visitor_ips')
      .select('user_id, user_email')
      .eq('ip_address', ip)
      .limit(1).maybeSingle();
    
    if (error && error.code !== 'PGRST116') {
      console.error('[IP TRACKING] Error checking IP association:', error.message);
      return null;
    }
    
    if (data && data.user_id) {
      return {
        userId: data.user_id,
        email: data.user_email
      };
    }
    
    return null;
  } catch (error) {
    console.error('[IP TRACKING] Exception checking IP association:', error.message);
    return null;
  }
}

module.exports = {
  getClientIP,
  trackVisitorIP,
  checkGuestUsage,
  incrementGuestUsage,
  associateIPWithAccount,
  isIPAssociated
};

