// Cookie Consent System - Shared across all pages
(function() {
    'use strict';
    
    // Cookie management functions
    function setCookie(name, value, days) {
        const expires = new Date();
        expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
        document.cookie = name + '=' + value + ';expires=' + expires.toUTCString() + ';path=/;SameSite=Lax';
    }
    
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
    
    function acceptAllCookies() {
        setCookie('cookie_consent', 'accepted', 365);
        setCookie('cookie_analytics', 'true', 365);
        setCookie('cookie_marketing', 'true', 365);
        const banner = document.getElementById('cookie-consent-banner');
        if (banner) {
            // Smooth fade out animation
            banner.style.opacity = '0';
            banner.style.transform = 'translateY(100%)';
            setTimeout(function() {
                banner.style.display = 'none';
            }, 300);
        }
        
        // Initialize analytics if enabled
        if (typeof gtag !== 'undefined') {
            gtag('consent', 'update', {
                'analytics_storage': 'granted',
                'ad_storage': 'granted'
            });
        }
        
        // Track consent acceptance
        if (typeof trackEvent !== 'undefined') {
            trackEvent('cookie_consent_accepted', { type: 'all' });
        }
        
        // Reload analytics if it exists
        if (typeof window.initAnalytics === 'function') {
            window.initAnalytics();
        }
    }
    
    function savePreferences() {
        const analytics = document.getElementById('cookie-analytics');
        const marketing = document.getElementById('cookie-marketing');
        if (!analytics || !marketing) return;
        
        const analyticsChecked = analytics.checked;
        const marketingChecked = marketing.checked;
        
        setCookie('cookie_consent', 'custom', 365);
        setCookie('cookie_analytics', analyticsChecked ? 'true' : 'false', 365);
        setCookie('cookie_marketing', marketingChecked ? 'true' : 'false', 365);
        
        const modal = document.getElementById('cookie-preferences-modal');
        const banner = document.getElementById('cookie-consent-banner');
        if (modal) {
            modal.style.opacity = '0';
            setTimeout(function() {
                modal.style.display = 'none';
            }, 200);
        }
        if (banner) {
            banner.style.opacity = '0';
            banner.style.transform = 'translateY(100%)';
            setTimeout(function() {
                banner.style.display = 'none';
            }, 300);
        }
        
        // Update Google Analytics consent
        if (typeof gtag !== 'undefined') {
            gtag('consent', 'update', {
                'analytics_storage': analyticsChecked ? 'granted' : 'denied',
                'ad_storage': marketingChecked ? 'granted' : 'denied'
            });
        }
        
        // Track preference save
        if (typeof trackEvent !== 'undefined') {
            trackEvent('cookie_preferences_saved', {
                analytics: analyticsChecked,
                marketing: marketingChecked
            });
        }
        
        // Reload analytics if enabled
        if (analyticsChecked && typeof window.initAnalytics === 'function') {
            window.initAnalytics();
        }
    }
    
    // Initialize on page load
    function initCookieConsent() {
        // Check if consent already given
        const consent = getCookie('cookie_consent');
        const banner = document.getElementById('cookie-consent-banner');
        
        if (!consent && banner) {
            // Show banner immediately with smooth animation
            // Use requestAnimationFrame for smooth rendering
            requestAnimationFrame(function() {
                banner.style.display = 'block';
                // Trigger CSS transition
                requestAnimationFrame(function() {
                    banner.style.opacity = '0';
                    banner.style.transform = 'translateY(100%)';
                    banner.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                    requestAnimationFrame(function() {
                        banner.style.opacity = '1';
                        banner.style.transform = 'translateY(0)';
                    });
                });
            });
        } else if (consent) {
            // Load saved preferences
            const analytics = getCookie('cookie_analytics') === 'true';
            const marketing = getCookie('cookie_marketing') === 'true';
            
            const analyticsEl = document.getElementById('cookie-analytics');
            const marketingEl = document.getElementById('cookie-marketing');
            if (analyticsEl) analyticsEl.checked = analytics;
            if (marketingEl) marketingEl.checked = marketing;
        }
        
        // Event listeners
        const acceptBtn = document.getElementById('cookie-consent-accept-all');
        const preferencesBtn = document.getElementById('cookie-consent-preferences');
        const closeBtn = document.getElementById('cookie-preferences-close');
        const saveBtn = document.getElementById('cookie-preferences-save');
        
        if (acceptBtn) {
            acceptBtn.addEventListener('click', acceptAllCookies);
        }
        
        if (preferencesBtn) {
            preferencesBtn.addEventListener('click', function() {
                const modal = document.getElementById('cookie-preferences-modal');
                if (modal) {
                    modal.style.display = 'block';
                    requestAnimationFrame(function() {
                        modal.style.opacity = '0';
                        modal.style.transition = 'opacity 0.2s ease';
                        requestAnimationFrame(function() {
                            modal.style.opacity = '1';
                        });
                    });
                }
            });
        }
        
        if (closeBtn) {
            closeBtn.addEventListener('click', function() {
                const modal = document.getElementById('cookie-preferences-modal');
                if (modal) {
                    modal.style.opacity = '0';
                    setTimeout(function() {
                        modal.style.display = 'none';
                    }, 200);
                }
            });
        }
        
        if (saveBtn) {
            saveBtn.addEventListener('click', savePreferences);
        }
        
        // Close modal on background click
        const modal = document.getElementById('cookie-preferences-modal');
        if (modal) {
            modal.addEventListener('click', function(e) {
                if (e.target === modal) {
                    modal.style.opacity = '0';
                    setTimeout(function() {
                        modal.style.display = 'none';
                    }, 200);
                }
            });
        }
        
        // Toggle switch styling
        const toggles = document.querySelectorAll('input[type="checkbox"][id^="cookie-"]');
        toggles.forEach(function(toggle) {
            toggle.addEventListener('change', function() {
                const span = this.nextElementSibling;
                if (span) {
                    if (this.checked) {
                        span.style.backgroundColor = '#0066ff';
                        const innerSpan = span.querySelector('span');
                        if (innerSpan) innerSpan.style.transform = 'translateX(24px)';
                    } else {
                        span.style.backgroundColor = '#333';
                        const innerSpan = span.querySelector('span');
                        if (innerSpan) innerSpan.style.transform = 'translateX(0)';
                    }
                }
            });
            
            // Initialize toggle state
            if (toggle.checked) {
                const span = toggle.nextElementSibling;
                if (span) {
                    span.style.backgroundColor = '#0066ff';
                    const innerSpan = span.querySelector('span');
                    if (innerSpan) innerSpan.style.transform = 'translateX(24px)';
                }
            }
        });
    }
    
    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initCookieConsent);
    } else {
        initCookieConsent();
    }
})();

