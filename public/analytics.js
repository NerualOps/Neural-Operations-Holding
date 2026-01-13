// Google Analytics 4 (GA4) with Consent Mode
// This file handles all analytics tracking with proper consent management

(function() {
    'use strict';
    
    // Get GA4 Measurement ID from environment or use placeholder
    // Replace 'G-XXXXXXXXXX' with your actual GA4 Measurement ID
    const GA4_MEASUREMENT_ID = 'G-XXXXXXXXXX'; // TODO: Replace with actual GA4 ID
    
    // Initialize Google Analytics with consent mode
    function initGA4() {
        // Only initialize if a valid GA4 ID is provided (not placeholder)
        if (!GA4_MEASUREMENT_ID || GA4_MEASUREMENT_ID === 'G-XXXXXXXXXX' || GA4_MEASUREMENT_ID.includes('XXXXXXXXXX')) {
            console.debug('[ANALYTICS] Google Analytics disabled - placeholder ID detected');
            return;
        }
        
        // Set default consent mode to 'denied' until user accepts
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', GA4_MEASUREMENT_ID, {
            'anonymize_ip': true,
            'cookie_flags': 'SameSite=None;Secure',
            'consent_mode': {
                'analytics_storage': 'denied',
                'ad_storage': 'denied',
                'wait_for_update': 500
            }
        });
        window.gtag = gtag;
        
        // Load GA4 script
        const script = document.createElement('script');
        script.async = true;
        script.src = 'https://www.googletagmanager.com/gtag/js?id=' + GA4_MEASUREMENT_ID;
        document.head.appendChild(script);
    }
    
    // Track page view
    async function trackPageView(pagePath, pageTitle) {
        if (typeof gtag !== 'undefined') {
            gtag('event', 'page_view', {
                'page_path': pagePath,
                'page_title': pageTitle,
                'page_location': window.location.href
            });
        }
        
        // Also send to custom analytics endpoint
        await sendCustomEvent('page_view', {
            path: pagePath,
            title: pageTitle,
            url: window.location.href,
            referrer: document.referrer,
            timestamp: new Date().toISOString()
        });
    }
    
    // Track custom events
    async function trackEvent(eventName, eventParams) {
        if (typeof gtag !== 'undefined') {
            gtag('event', eventName, eventParams);
        }
        
        await sendCustomEvent(eventName, eventParams);
    }
    
    // Send custom analytics to your backend
    async function sendCustomEvent(eventName, eventData) {
        // Only send if analytics cookies are enabled
        if (!isAnalyticsEnabled()) return;
        
        // Validate event name
        if (!eventName || typeof eventName !== 'string' || eventName.trim() === '') {
            console.debug('[ANALYTICS] Invalid event name, skipping');
            return;
        }
        
        // Generate session ID if not exists
        let sessionId = sessionStorage.getItem('analytics_session_id');
        if (!sessionId) {
            sessionId = 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            sessionStorage.setItem('analytics_session_id', sessionId);
        }
        
        try {
            // Get CSRF token if available
            let csrfToken = null;
            try {
                if (typeof window.getCsrfToken === 'function') {
                    csrfToken = await window.getCsrfToken();
                } else {
                    // Fallback: try to get from cookie
                    const cookies = document.cookie.split(';');
                    for (let i = 0; i < cookies.length; i++) {
                        const cookie = cookies[i].trim();
                        if (cookie.startsWith('csrfToken=')) {
                            csrfToken = cookie.substring('csrf_token='.length);
                            break;
                        }
                    }
                }
            } catch (e) {
                // CSRF token not available - continue anyway
            }
            
            const headers = {
                'Content-Type': 'application/json',
                'X-Session-ID': sessionId
            };
            
            if (csrfToken) {
                headers['X-CSRF-Token'] = csrfToken;
            }
            
            fetch('/api/analytics/track', {
                method: 'POST',
                headers: headers,
                credentials: 'include',
                body: JSON.stringify({
                    event: eventName.trim(),
                    data: eventData || {},
                    timestamp: new Date().toISOString(),
                    userAgent: navigator.userAgent,
                    language: navigator.language,
                    screen: {
                        width: window.screen.width,
                        height: window.screen.height
                    },
                    viewport: {
                        width: window.innerWidth,
                        height: window.innerHeight
                    }
                }),
                keepalive: true
            }).catch(err => {
                // Silently fail - analytics should never break the site
                console.debug('Analytics tracking failed:', err);
            });
        } catch (e) {
            // Silently fail
            console.debug('Analytics error:', e);
        }
    }
    
    // Check if analytics is enabled
    function isAnalyticsEnabled() {
        const consent = getCookie('cookie_consent');
        const analytics = getCookie('cookie_analytics');
        return consent && (analytics === 'true' || consent === 'accepted');
    }
    
    // Check if marketing is enabled
    function isMarketingEnabled() {
        const consent = getCookie('cookie_consent');
        const marketing = getCookie('cookie_marketing');
        return consent && (marketing === 'true' || consent === 'accepted');
    }
    
    // Get cookie helper
    function getCookie(name) {
        const nameEQ = name + '=';
        const ca = document.cookie.split(';');
        for (let i = 0; i < ca.length; i++) {
            let c = ca[i];
            while (c.charAt(0) === ' ') c = c.substring(1, c.length);
            if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
        }
        return null;
    }
    
    // Track user interactions
    function setupEventTracking() {
        // Track button clicks
        document.addEventListener('click', function(e) {
            const target = e.target;
            if (target.tagName === 'BUTTON' || target.tagName === 'A') {
                const buttonText = target.textContent.trim();
                const buttonId = target.id || target.className;
                trackEvent('button_click', {
                    button_text: buttonText,
                    button_id: buttonId,
                    button_url: target.href || null
                });
            }
        });
        
        // Track form submissions
        document.addEventListener('submit', function(e) {
            const form = e.target;
            if (form.tagName === 'FORM') {
                trackEvent('form_submit', {
                    form_id: form.id || form.className,
                    form_action: form.action || null
                });
            }
        });
        
        // Track scroll depth
        let maxScroll = 0;
        window.addEventListener('scroll', function() {
            const scrollPercent = Math.round(
                (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100
            );
            if (scrollPercent > maxScroll) {
                maxScroll = scrollPercent;
                if (scrollPercent === 25 || scrollPercent === 50 || scrollPercent === 75 || scrollPercent === 100) {
                    trackEvent('scroll_depth', {
                        scroll_percent: scrollPercent
                    });
                }
            }
        });
        
        // Track time on page
        let startTime = Date.now();
        window.addEventListener('beforeunload', function() {
            const timeOnPage = Math.round((Date.now() - startTime) / 1000);
            trackEvent('time_on_page', {
                seconds: timeOnPage
            });
        });
    }
    
    // Initialize when DOM is ready
    function init() {
        // Always initialize GA4 (it will respect consent mode)
        initGA4();
        
        // Track initial page view after a short delay (to ensure consent is checked)
        setTimeout(function() {
            if (isAnalyticsEnabled()) {
                trackPageView(window.location.pathname, document.title);
            }
        }, 1000);
        
        // Setup event tracking
        setupEventTracking();
    }
    
    // Expose tracking functions globally
    window.trackEvent = trackEvent;
    window.trackPageView = trackPageView;
    window.isAnalyticsEnabled = isAnalyticsEnabled;
    window.isMarketingEnabled = isMarketingEnabled;
    
    // Initialize when ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

