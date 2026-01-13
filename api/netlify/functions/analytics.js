const jwt = require('jsonwebtoken');
const { createClient } = require('@supabase/supabase-js');

// Initialize Supabase client - no cookies to prevent "header too large" errors
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
        'X-Client-Info': 'neuralops-analytics'
      }
    }
  }
);

// Use environment variable for JWT secret with fallback for development
// Security: Never use fallback secret
if (!process.env.JWT_SECRET) {
  console.error('[SECURITY] JWT_SECRET environment variable is required');
  throw new Error('JWT_SECRET environment variable is required');
}
const JWT_SECRET = process.env.JWT_SECRET;

exports.handler = async (event, context) => {
  // Set secure CORS headers - NO WILDCARDS
  const headers = {
    'Access-Control-Allow-Origin': process.env.FRONTEND_URL || 'https://neuralops.biz',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, Cookie, X-CSRF-Token',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Credentials': 'true',
    'Content-Type': 'application/json'
  };

  // Handle preflight requests
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers };
  }

    // Silent - no console.log

  try {
    const userAgent = event.headers['user-agent'] || '';
    const isBot = /bot|crawler|spider|scraper|curl|wget|python|java|googlebot|bingbot|slurp|duckduckbot|baiduspider|yandexbot|sogou|exabot|facebot|ia_archiver|archive\.org_bot/i.test(userAgent);
    const path = event.path || '';
    const isTrackEndpoint = path.includes('/track') || event.httpMethod === 'POST';
    
    if (isTrackEndpoint && isBot) {
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({ success: true })
      };
    }
    
    if (event.httpMethod !== 'GET' && !isTrackEndpoint) {
      if (!event.headers['x-csrf-token']) {
        return {
          statusCode: 403,
          headers,
          body: JSON.stringify({ error: 'CSRF token missing' })
        };
      }
    }
    
    const token = extractToken(event);
    
    if (!token && !isTrackEndpoint) {
      return {
        statusCode: 401,
        headers,
        body: JSON.stringify({ error: 'Authentication required' })
      };
    }

    // Verify JWT token
    let decoded;
    try {
      // Silent - no console.log
      decoded = jwt.verify(token, JWT_SECRET);
      // Silent - no console.log
      // Silent - no console.log
    } catch (jwtError) {
      console.error('JWT verification failed:', jwtError.message);
      return {
        statusCode: 401,
        headers,
        body: JSON.stringify({ error: 'Invalid authentication token' })
      };
    }

    // Check if user is owner directly from the token
    if (decoded.role !== 'owner') {
      // Silent - no console.log
      return {
        statusCode: 403,
        headers,
        body: JSON.stringify({ error: 'Owner access required' })
      };
    }

    // Silent - no console.log

    // Parse request body
    let body = {};
    try {
      body = JSON.parse(event.body || '{}');
    } catch (parseError) {
      console.error('Error parsing request body:', parseError);
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: 'Invalid JSON in request body' })
      };
    }

    const { action } = body;
    // Silent - no console.log

    // Process different actions
    switch (action) {
      case 'get-analytics':
        return await getAnalytics(headers, decoded);
      case 'get-users':
        return await getUsers(headers, decoded);
      case 'get-conversations':
        return await getConversations(headers, decoded);
      case 'get-page-visits':
        return await getPageVisits(headers, decoded);
      case 'get-ai-interactions':
        return await getAIInteractions(headers, decoded);
      case 'get-estimates':
        return await getEstimates(headers, decoded);
      case 'get-conversions':
        return await getConversions(headers, decoded);
      default:
        // Silent - no console.log
        return {
          statusCode: 400,
          headers,
          body: JSON.stringify({ error: 'Invalid action' })
        };
    }
  } catch (error) {
    console.error('Analytics error:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Server error: ' + error.message })
    };
  }
};

// Helper function to extract token from request
function extractToken(event) {
  // Try to get token from cookie
  if (event.headers.cookie) {
    const cookies = parseCookies(event.headers.cookie);
    if (cookies.authToken) {
      return cookies.authToken;
    }
  }

  // Try to get token from Authorization header
  const authHeader = event.headers.authorization || event.headers.Authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    return authHeader.substring(7);
  }

  return null;
}

// Helper function to detect HTML errors (Supabase downtime)
function isHtmlError(error) {
  if (!error) return false;
  let errorStr = '';
  if (typeof error === 'string') {
    errorStr = error;
  } else {
    errorStr = error?.message || error?.toString() || JSON.stringify(error) || '';
  }
  return errorStr.includes('<!DOCTYPE html>') || 
         errorStr.includes('Cloudflare') || 
         errorStr.includes('Error code 522') || 
         errorStr.includes('Error code 521') ||
         errorStr.includes('Connection timed out') ||
         errorStr.includes('Web server is down');
}

// Helper function to safely log errors (detects HTML and logs warnings instead)
function safeLogError(context, error) {
  if (isHtmlError(error)) {
    console.warn(`[WARN] ${context}: Supabase connection issue (likely downtime)`);
  } else {
    const errorMsg = typeof error === 'string' ? error.substring(0, 200) : (error?.message || 'Unknown error');
    console.error(`${context}:`, errorMsg);
  }
}

// Helper function to parse cookies
function parseCookies(cookieHeader) {
  const cookies = {};
  if (cookieHeader) {
    cookieHeader.split(';').forEach(cookie => {
      const [name, value] = cookie.trim().split('=');
      if (name && value) {
        cookies[name] = decodeURIComponent(value);
      }
    });
  }
  return cookies;
}

// Get analytics data
async function getAnalytics(headers, user) {
  try {
    // Silent - no console.log
    
    // Get total users count from profiles table
    const { count: totalUsers, error: usersError } = await supabase
      .from('profiles')
      .select('*', { count: 'exact', head: true });
    
    if (usersError) {
      safeLogError('Error fetching total users', usersError);
    }

    // Get active users (last 24 hours) - using last_login from profiles
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    
    const { count: activeUsers, error: activeError } = await supabase
      .from('profiles')
      .select('*', { count: 'exact', head: true })
      .gte('last_login', yesterday.toISOString());
    
    if (activeError) {
      safeLogError('Error fetching active users', activeError);
    }

    // Get total conversations and calculate average response time
    const { count: conversations, error: convError } = await supabase
      .from('epsilon_conversations')
      .select('*', { count: 'exact', head: true });
    
    if (convError) {
      safeLogError('Error fetching conversations', convError);
    }

    // Calculate average response time from actual data
    const { data: responseTimeData, error: responseTimeError } = await supabase
      .from('epsilon_conversations')
      .select('response_time_ms')
      .not('response_time_ms', 'is', null)
      .gt('response_time_ms', 0)
      .limit(1000); // Sample last 1000 for performance
    
    let averageResponseTime = 0;
    if (!responseTimeError && responseTimeData && responseTimeData.length > 0) {
      const totalTime = responseTimeData.reduce((sum, conv) => sum + (conv.response_time_ms || 0), 0);
      averageResponseTime = (totalTime / responseTimeData.length / 1000).toFixed(2); // Convert ms to seconds
    }

    // Get user satisfaction from analytics_events table
    const { data: satisfactionEvents, error: satisfactionError } = await supabase
      .from('analytics_events')
      .select('event_data')
      .eq('event_type', 'satisfaction_rating')
      .not('event_data->rating', 'is', null);
    
    let userSatisfaction = 0;
    if (!satisfactionError && satisfactionEvents && satisfactionEvents.length > 0) {
      const validRatings = satisfactionEvents
        .map(e => e.event_data?.rating)
        .filter(r => r && r > 0);
      if (validRatings.length > 0) {
        userSatisfaction = (validRatings.reduce((sum, r) => sum + r, 0) / validRatings.length).toFixed(1);
      }
    }

    // Get top intents from analytics_events
    const { data: intentEvents, error: intentError } = await supabase
      .from('analytics_events')
      .select('event_data')
      .eq('event_type', 'conversation_intent')
      .not('event_data->intent', 'is', null);
    
    let topIntents = [];
    if (!intentError && intentEvents) {
      const intentCounts = {};
      intentEvents.forEach(event => {
        const intent = event.event_data?.intent;
        if (intent) {
          intentCounts[intent] = (intentCounts[intent] || 0) + 1;
        }
      });
      
      topIntents = Object.entries(intentCounts)
        .map(([intent, count]) => ({ intent, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 5);
    }

    // Get daily activity for last 7 days
    const dailyActivity = [];
    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const startOfDay = new Date(date);
      startOfDay.setHours(0, 0, 0, 0);
      const endOfDay = new Date(date);
      endOfDay.setHours(23, 59, 59, 999);
      
      const { count: dayConversations, error: dayConvError } = await supabase
        .from('epsilon_conversations')
        .select('*', { count: 'exact', head: true })
        .gte('created_at', startOfDay.toISOString())
        .lte('created_at', endOfDay.toISOString());
      
      const { count: dayUsers, error: dayUsersError } = await supabase
        .from('profiles')
        .select('*', { count: 'exact', head: true })
        .gte('last_login', startOfDay.toISOString())
        .lte('last_login', endOfDay.toISOString());
      
      dailyActivity.push({
        date: date.toISOString().split('T')[0],
        conversations: dayConvError ? 0 : (dayConversations || 0),
        users: dayUsersError ? 0 : (dayUsers || 0)
      });
    }

    const analyticsData = {
      totalUsers: totalUsers || 0,
      activeUsers: activeUsers || 0,
      conversations: conversations || 0,
      averageResponseTime: parseFloat(averageResponseTime) || 0,
      userSatisfaction: parseFloat(userSatisfaction) || 0,
      topIntents: topIntents,
      dailyActivity: dailyActivity
    };

    // Silent - no console.log

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(analyticsData)
    };
  } catch (error) {
    console.error('Error getting analytics:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to get analytics data' })
    };
  }
}

// Get users data
async function getUsers(headers, user) {
  try {
    // Silent - no console.log
    
    // Get all users from profiles table
    const { data: users, error: usersError } = await supabase
      .from('profiles')
      .select(`
        id,
        name,
        email,
        role,
        last_login,
        created_at
      `)
      .order('last_login', { ascending: false });

    if (usersError) {
      if (isHtmlError(usersError)) {
        safeLogError('Error fetching users', usersError);
        // Return empty users list when Supabase is down
        return {
          statusCode: 200,
          headers,
          body: JSON.stringify({ users: [], total: 0 })
        };
      } else {
        safeLogError('Error fetching users', usersError);
        return {
          statusCode: 500,
          headers,
          body: JSON.stringify({ error: 'Failed to fetch users' })
        };
      }
    }

    // Get conversation counts for each user from analytics_events
    const usersWithConversations = await Promise.all(
      (users || []).map(async (user) => {
        const { count: conversationCount, error: convError } = await supabase
          .from('analytics_events')
          .select('*', { count: 'exact', head: true })
          .eq('user_id', user.id)
          .eq('event_type', 'conversation_start');
        
        return {
          id: user.id,
          name: user.name || 'Unknown',
          email: user.email || 'No email',
          role: user.role || 'client',
          lastActive: user.last_login || user.created_at,
          conversations: convError ? 0 : (conversationCount || 0)
        };
      })
    );

    const usersData = {
      users: usersWithConversations,
      total: usersWithConversations.length
    };

    // Silent - no console.log

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(usersData)
    };
  } catch (error) {
    console.error('Error getting users:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to get users data' })
    };
  }
}

// Get conversations data
async function getConversations(headers, user) {
  try {
    // Silent - no console.log
    
    // Get conversations from the conversations table
    const { data: conversations, error: convError } = await supabase
      .from('epsilon_conversations')
      .select(`
        id,
        session_id,
        user_message,
        epsilon_response,
        context_data,
        learning_metadata,
        created_at
      `)
      .order('created_at', { ascending: false })
      .limit(50);

    if (convError) {
      if (isHtmlError(convError)) {
        safeLogError('Error fetching conversations', convError);
        // Return empty conversations when Supabase is down
        return {
          statusCode: 200,
          headers,
          body: JSON.stringify({ conversations: [], total: 0 })
        };
      } else {
        safeLogError('Error fetching conversations', convError);
        return {
          statusCode: 500,
          headers,
          body: JSON.stringify({ error: 'Failed to fetch conversations' })
        };
      }
    }

    // Group conversations by session_id to create conversation objects
    const sessionGroups = {};
    conversations?.forEach(conv => {
      if (!sessionGroups[conv.session_id]) {
        sessionGroups[conv.session_id] = {
          id: conv.session_id,
          messages: [],
          created_at: conv.created_at,
          last_activity: conv.created_at,
          intent: conv.context_data?.intent || 'unknown' // Store intent from first conversation
        };
      }
      // Convert epsilon_conversations format to message format
      sessionGroups[conv.session_id].messages.push({
        id: conv.id,
        role: 'user',
        text: conv.user_message,
        created_at: conv.created_at
      });
      sessionGroups[conv.session_id].messages.push({
        id: conv.id + '_response',
        role: 'epsilon',
        text: conv.epsilon_response,
        created_at: conv.created_at
      });
      if (conv.created_at > sessionGroups[conv.session_id].last_activity) {
        sessionGroups[conv.session_id].last_activity = conv.created_at;
      }
      // Update intent if this conversation has one and we don't have one yet
      if (conv.context_data?.intent && sessionGroups[conv.session_id].intent === 'unknown') {
        sessionGroups[conv.session_id].intent = conv.context_data.intent;
      }
    });

    // Convert to array and get user info from analytics_events
    const conversationSessions = Object.values(sessionGroups);
    
    const conversationsWithUsers = await Promise.all(
      conversationSessions.map(async (session) => {
        // Try to get user info from analytics_events
        const { data: userEvent, error: userError } = await supabase
          .from('analytics_events')
          .select('user_id, event_data')
          .eq('session_id', session.id)
          .eq('event_type', 'conversation_start')
          .limit(1)
          .limit(1).maybeSingle();
        
        let userInfo = { name: 'Unknown', email: 'No email' };
        if (!userError && userEvent?.user_id) {
          const { data: profile } = await supabase
            .from('profiles')
            .select('name, email')
            .eq('id', userEvent.user_id)
            .limit(1).maybeSingle();
          if (profile) {
            userInfo = profile;
          }
        }
        
        return {
          id: session.id,
          user: userInfo.name,
          email: userInfo.email,
          date: session.created_at,
          messages: session.messages.length,
          intent: session.intent || 'unknown',
          satisfaction: null // Would need to be tracked separately
        };
      })
    );

    const conversationsData = {
      conversations: conversationsWithUsers,
      total: conversationsWithUsers.length
    };

    // Silent - no console.log

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(conversationsData)
    };
  } catch (error) {
    console.error('Error getting conversations:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to get conversations data' })
    };
  }
}

// Get page visits analytics
async function getPageVisits(headers, user) {
  try {
    // Silent - no console.log
    
    // Get page visit events from analytics_events
    const { data: visitEvents, error: visitError } = await supabase
      .from('analytics_events')
      .select(`
        id,
        user_id,
        session_id,
        event_data,
        created_at
      `)
      .eq('event_type', 'page_visit')
      .order('created_at', { ascending: false })
      .limit(1000);

    if (visitError) {
      safeLogError('Error fetching page visits', visitError);
      // Return empty data structure instead of mock data
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({
          totalVisits: 0,
          uniqueVisitors: 0,
          topPages: [],
          dailyVisits: []
        })
      };
    }

    // Calculate metrics from events
    const totalVisits = visitEvents?.length || 0;
    
    // Get unique visitors (unique user_ids)
    const uniqueUserIds = new Set();
    visitEvents?.forEach(event => {
      if (event.user_id) {
        uniqueUserIds.add(event.user_id);
      }
    });
    const uniqueVisitors = uniqueUserIds.size;

    // Group by page
    const pageCounts = {};
    visitEvents?.forEach(event => {
      const page = event.event_data?.page || event.event_data?.path || '/';
      if (!pageCounts[page]) {
        pageCounts[page] = { visits: 0, uniqueVisitors: new Set() };
      }
      pageCounts[page].visits++;
      if (event.user_id) {
        pageCounts[page].uniqueVisitors.add(event.user_id);
      }
    });

    const topPages = Object.entries(pageCounts)
      .map(([page, data]) => ({
        page,
        visits: data.visits,
        uniqueVisitors: data.uniqueVisitors.size
      }))
      .sort((a, b) => b.visits - a.visits)
      .slice(0, 10);

    // Get daily visits for last 7 days
    const dailyVisits = [];
    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const startOfDay = new Date(date);
      startOfDay.setHours(0, 0, 0, 0);
      const endOfDay = new Date(date);
      endOfDay.setHours(23, 59, 59, 999);
      
      const dayEvents = visitEvents?.filter(event => {
        const eventDate = new Date(event.created_at);
        return eventDate >= startOfDay && eventDate <= endOfDay;
      }) || [];
      
      const dayUniqueVisitors = new Set();
      dayEvents.forEach(event => {
        if (event.user_id) {
          dayUniqueVisitors.add(event.user_id);
        }
      });
      
      dailyVisits.push({
        date: date.toISOString().split('T')[0],
        visits: dayEvents.length,
        uniqueVisitors: dayUniqueVisitors.size
      });
    }

    const pageVisitsData = {
      totalVisits,
      uniqueVisitors,
      topPages,
      dailyVisits
    };

    // Silent - no console.log

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(pageVisitsData)
    };
  } catch (error) {
    console.error('Error getting page visits:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to get page visits data' })
    };
  }
}

// Get AI interactions analytics
async function getAIInteractions(headers, user) {
  try {
    // Silent - no console.log
    
    // Get AI interaction data from analytics_events
    const { data: interactions, error: intError } = await supabase
      .from('analytics_events')
      .select(`
        id,
        user_id,
        session_id,
        event_type,
        event_data,
        created_at
      `)
      .in('event_type', ['conversation_start', 'ai_response', 'satisfaction_rating'])
      .order('created_at', { ascending: false })
      .limit(100);

    if (intError) {
      if (isHtmlError(intError)) {
        console.warn('[WARN] Error fetching AI interactions: Supabase connection issue (likely downtime)');
        // Return empty data instead of error when Supabase is down
        return {
          statusCode: 200,
          headers,
          body: JSON.stringify({
            totalInteractions: 0,
            interactions: [],
            interactionsByType: {},
            recentInteractions: []
          })
        };
      } else {
        safeLogError('Error fetching AI interactions', intError);
        return {
          statusCode: 500,
          headers,
          body: JSON.stringify({ error: 'Failed to fetch AI interactions' })
        };
      }
    }

    // Calculate interaction metrics
    const totalInteractions = interactions?.length || 0;
    
    // Get satisfaction ratings
    const satisfactionEvents = interactions?.filter(int => int.event_type === 'satisfaction_rating') || [];
    const avgSatisfaction = satisfactionEvents.length > 0 
      ? (satisfactionEvents.reduce((sum, int) => sum + (int.event_data?.rating || 0), 0) / satisfactionEvents.length).toFixed(1)
      : 0;

    // Group by intent from event_data
    const intentCounts = {};
    interactions?.forEach(int => {
      const intent = int.event_data?.intent || 'unknown';
      intentCounts[intent] = (intentCounts[intent] || 0) + 1;
    });

    const topIntents = Object.entries(intentCounts)
      .map(([intent, count]) => ({ intent, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    // Get user info for recent interactions
    const recentInteractions = await Promise.all(
      (interactions?.slice(0, 10) || []).map(async (int) => {
        let userName = 'Unknown';
        if (int.user_id) {
          const { data: profile } = await supabase
            .from('profiles')
            .select('name')
            .eq('id', int.user_id)
            .limit(1).maybeSingle();
          if (profile) {
            userName = profile.name;
          }
        }
        
        return {
          id: int.id,
          user: userName,
          intent: int.event_data?.intent || 'unknown',
          date: int.created_at,
          satisfaction: int.event_type === 'satisfaction_rating' ? int.event_data?.rating : null
        };
      })
    );

    const aiInteractionsData = {
      totalInteractions,
      averageSatisfaction: parseFloat(avgSatisfaction),
      topIntents,
      recentInteractions
    };

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(aiInteractionsData)
    };
  } catch (error) {
    console.error('Error getting AI interactions:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to get AI interactions data' })
    };
  }
}

// Get cost estimates analytics
async function getEstimates(headers, user) {
  try {
    // Query estimates table
    const { data: estimates, error } = await supabase
      .from('estimates')
      .select('id, estimate_value, status, description, created_at')
      .order('created_at', { ascending: false })
      .limit(100);

    if (error) {
      if (isHtmlError(error)) {
        safeLogError('Error fetching estimates', error);
        // Return empty estimates when Supabase is down
        return {
          statusCode: 200,
          headers,
          body: JSON.stringify({
            totalEstimates: 0,
            totalValue: 0,
            averageEstimate: 0,
            estimateBreakdown: { pending: 0, accepted: 0, rejected: 0 },
            estimates: []
          })
        };
      } else {
        safeLogError('Error fetching estimates', error);
        throw error;
      }
    }

    const estimatesList = estimates || [];
    const totalEstimates = estimatesList.length;
    const totalValue = estimatesList.reduce((sum, e) => sum + parseFloat(e.estimate_value || 0), 0);
    const averageEstimate = totalEstimates > 0 ? totalValue / totalEstimates : 0;

    const estimateBreakdown = {
      pending: estimatesList.filter(e => e.status === 'pending').length,
      approved: estimatesList.filter(e => e.status === 'approved').length,
      rejected: estimatesList.filter(e => e.status === 'rejected').length
    };

    const recentEstimates = estimatesList.slice(0, 10).map(e => ({
      id: e.id,
      value: parseFloat(e.estimate_value || 0),
      status: e.status,
      description: e.description,
      created_at: e.created_at
    }));

    const estimatesData = {
      totalEstimates,
      averageEstimate,
      totalValue,
      recentEstimates,
      estimateBreakdown
    };

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(estimatesData)
    };
  } catch (error) {
    console.error('Error getting estimates:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to get estimates data' })
    };
  }
}

// Get conversion tracking analytics
async function getConversions(headers, user) {
  try {
    // Query conversions table
    const { data: conversions, error } = await supabase
      .from('conversions')
      .select('id, conversion_type, revenue, created_at')
      .order('created_at', { ascending: false })
      .limit(1000);

    if (error) {
      if (isHtmlError(error)) {
        safeLogError('Error fetching conversions', error);
        // Return empty conversions when Supabase is down
        return {
          statusCode: 200,
          headers,
          body: JSON.stringify({
            totalConversions: 0,
            totalRevenue: 0,
            averageRevenue: 0,
            conversions: []
          })
        };
      } else {
        safeLogError('Error fetching conversions', error);
        throw error;
      }
    }

    const conversionsList = conversions || [];
    const totalConversions = conversionsList.filter(c => c.conversion_type === 'conversion').length;
    const totalRevenue = conversionsList
      .filter(c => c.conversion_type === 'conversion')
      .reduce((sum, c) => sum + parseFloat(c.revenue || 0), 0);

    // Get visitor count from page_visits for conversion rate calculation
    const { count: visitorCount } = await supabase
      .from('page_visits')
      .select('id', { count: 'exact', head: true });

    const visitors = visitorCount || 0;
    const conversionRate = visitors > 0 ? (totalConversions / visitors) * 100 : 0;

    const conversionFunnel = {
      visitors: visitors,
      leads: conversionsList.filter(c => c.conversion_type === 'lead').length,
      qualifiedLeads: conversionsList.filter(c => c.conversion_type === 'qualified_lead').length,
      conversions: totalConversions
    };

    const recentConversions = conversionsList
      .filter(c => c.conversion_type === 'conversion')
      .slice(0, 10)
      .map(c => ({
        id: c.id,
        revenue: parseFloat(c.revenue || 0),
        created_at: c.created_at
      }));

    // Calculate monthly trend (last 12 months)
    const monthlyTrend = [];
    const now = new Date();
    for (let i = 11; i >= 0; i--) {
      const monthStart = new Date(now.getFullYear(), now.getMonth() - i, 1);
      const monthEnd = new Date(now.getFullYear(), now.getMonth() - i + 1, 0);
      const monthConversions = conversionsList.filter(c => {
        const convDate = new Date(c.created_at);
        return convDate >= monthStart && convDate <= monthEnd && c.conversion_type === 'conversion';
      });
      monthlyTrend.push({
        month: monthStart.toISOString().substring(0, 7),
        count: monthConversions.length,
        revenue: monthConversions.reduce((sum, c) => sum + parseFloat(c.revenue || 0), 0)
      });
    }

    const conversionsData = {
      totalConversions,
      conversionRate,
      totalRevenue,
      conversionFunnel,
      recentConversions,
      monthlyTrend
    };

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(conversionsData)
    };
  } catch (error) {
    console.error('Error getting conversions:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Failed to get conversions data' })
    };
  }
}
