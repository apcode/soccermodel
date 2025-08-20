import puppeteer from 'puppeteer';

class FastFotmobCrawler {
    constructor() {
        this.browser = null;
        this.page = null;
        this.baseUrl = 'https://www.fotmob.com/en-GB';
        // No global duplicate tracking - allow leagues to appear in appropriate countries
    }

    async init() {
        console.log('Initializing fast crawler...');
        
        this.browser = await puppeteer.launch({
            headless: false,
            defaultViewport: {
                width: 1920,
                height: 1080
            },
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]
        });

        this.page = await this.browser.newPage();
        
        await this.page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
        
        await this.page.setViewport({
            width: 1920,
            height: 1080
        });

        console.log('Fast crawler initialized successfully');
    }

    async loadHomepage() {
        console.log('Loading fotmob homepage...');
        
        await this.page.goto(this.baseUrl, {
            waitUntil: 'networkidle2',
            timeout: 30000
        });

        await new Promise(resolve => setTimeout(resolve, 1000));
        
        console.log('Homepage loaded successfully');
    }

    async extractAllLeagues() {
        try {
            console.log('Looking for and expanding leagues menu...');
            
            // Find and click the "All leagues" dropdown
            const found = await this.page.evaluate(() => {
                const allElements = document.querySelectorAll('*');
                for (let el of allElements) {
                    const text = el.textContent?.trim();
                    if (text === 'All leagues' || text === 'All Leagues') {
                        let parent = el.parentElement;
                        for (let i = 0; i < 3 && parent; i++) {
                            if (parent.tagName === 'BUTTON' || parent.tagName === 'A' || parent.onclick || parent.getAttribute('role') === 'button') {
                                parent.click();
                                return true;
                            }
                            parent = parent.parentElement;
                        }
                    }
                }
                return false;
            });
            
            if (found) {
                console.log('Clicked on leagues menu, waiting for content to load...');
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
            
        } catch (error) {
            console.log('Error finding leagues menu:', error.message);
        }
        
        // Now expand each country and collect all leagues
        console.log('Expanding all countries rapidly...');
        
        const countries = await this.page.evaluate(() => {
            const countryNames = ['International', 'Albania', 'Algeria', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Chile', 'China', 'Colombia', 'Costa Rica', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Ecuador', 'Egypt', 'El Salvador', 'England', 'Estonia', 'Faroe Islands', 'Finland', 'France', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Guatemala', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Latvia', 'Lebanon', 'Lithuania', 'Luxembourg', 'Malaysia', 'Malta', 'Mexico', 'Moldova', 'Montenegro', 'Morocco', 'Myanmar', 'Netherlands', 'New Zealand', 'Nicaragua', 'North Macedonia', 'Northern Ireland', 'Norway', 'Oman', 'Pakistan', 'Palestine', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'San Marino', 'Saudi Arabia', 'Scotland', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sweden', 'Switzerland', 'Thailand', 'Tunisia', 'Turkey', 'Turkiye', 'UAE', 'Ukraine', 'United States', 'Uruguay', 'USA', 'Uzbekistan', 'Venezuela', 'Vietnam', 'Wales', 'Zimbabwe'];
            
            const foundCountries = [];
            const allElements = document.querySelectorAll('*');
            
            for (let el of allElements) {
                const text = el.textContent?.trim();
                if (text && countryNames.includes(text)) {
                    const rect = el.getBoundingClientRect();
                    if (rect.height > 0 && rect.width > 0) {
                        foundCountries.push(text);
                    }
                }
            }
            
            return [...new Set(foundCountries)];
        });
        
        console.log(`Found ${countries.length} countries: ${countries.slice(0, 5).join(', ')}...`);
        
        const result = {
            sections: [],
            debug: [`Found ${countries.length} countries: ${countries.join(', ')}`]
        };
        
        // Process all countries super fast
        for (let i = 0; i < countries.length; i++) {
            const countryName = countries[i];
            try {
                console.log(`[${i + 1}/${countries.length}] Processing ${countryName}...`);
                
                // Record leagues before clicking
                const leaguesBefore = await this.page.evaluate(() => {
                    return Array.from(document.querySelectorAll('a[href*="/leagues/"]')).map(link => link.href);
                });
                
                // Click the country arrow (improved logic)
                const expanded = await this.page.evaluate((country) => {
                    const allElements = document.querySelectorAll('*');
                    
                    for (let el of allElements) {
                        if (el.textContent?.trim() === country) {
                            // Look for arrow in parent element or nearby elements
                            let arrowElement = null;
                            
                            // Check parent element for arrow
                            const parent = el.parentElement;
                            if (parent && parent.innerHTML.includes('►')) {
                                arrowElement = parent;
                            }
                            
                            // Check siblings for arrow
                            let sibling = el.nextElementSibling;
                            while (sibling && !arrowElement) {
                                if (sibling.innerHTML?.includes('►') || sibling.textContent?.includes('►')) {
                                    arrowElement = sibling;
                                    break;
                                }
                                sibling = sibling.nextElementSibling;
                            }
                            
                            // Check if the element itself or parent is clickable
                            if (!arrowElement) {
                                if (parent && (parent.tagName === 'BUTTON' || parent.onclick || parent.style.cursor === 'pointer')) {
                                    arrowElement = parent;
                                } else if (el.tagName === 'BUTTON' || el.onclick || el.style.cursor === 'pointer') {
                                    arrowElement = el;
                                }
                            }
                            
                            if (arrowElement) {
                                arrowElement.click();
                                return true;
                            }
                        }
                    }
                    return false;
                }, countryName);
                
                if (expanded) {
                    // Very short wait for DOM update
                    await new Promise(resolve => setTimeout(resolve, 200));
                    
                    // Get leagues that belong to this specific country by looking at DOM position
                    const countryLeagues = await this.page.evaluate((country) => {
                        const leagues = [];
                        
                        // Find the country heading element
                        let countryElement = null;
                        const allElements = document.querySelectorAll('*');
                        
                        for (let el of allElements) {
                            if (el.textContent?.trim() === country) {
                                const rect = el.getBoundingClientRect();
                                if (rect.height > 0 && rect.width > 0) {
                                    countryElement = el;
                                    break;
                                }
                            }
                        }
                        
                        if (countryElement) {
                            const countryRect = countryElement.getBoundingClientRect();
                            const leagueLinks = document.querySelectorAll('a[href*="/leagues/"]');
                            
                            leagueLinks.forEach(link => {
                                const rect = link.getBoundingClientRect();
                                const text = link.textContent?.trim();
                                const href = link.getAttribute('href');
                                
                                // Only include leagues that appear below this country and are visible
                                if (text && href && rect.height > 0 && rect.width > 0) {
                                    const isBelow = rect.top > countryRect.top;
                                    const isReasonablyClose = rect.top - countryRect.top < 500; // Within 500px
                                    
                                    if (isBelow && isReasonablyClose) {
                                        leagues.push({
                                            name: text,
                                            url: href
                                        });
                                    }
                                }
                            });
                        }
                        
                        return leagues.slice(0, 20); // Limit to first 20 leagues per country to avoid cross-contamination
                    }, countryName);
                    
                    if (countryLeagues.length > 0) {
                        // Add to results without global duplicate checking
                        result.sections.push({
                            section: countryName,
                            leagues: countryLeagues
                        });
                        
                        console.log(`Found ${countryLeagues.length} leagues for ${countryName}`);
                    }
                } else {
                    console.log(`Could not expand ${countryName}`);
                }
                
            } catch (error) {
                console.log(`Error processing ${countryName}:`, error.message);
            }
        }
        
        const totalLeagues = result.sections.reduce((sum, section) => sum + section.leagues.length, 0);
        
        console.log(`\\nExtracted Data Summary:`);
        result.sections.forEach(section => {
            console.log(`${section.section}: ${section.leagues.length} leagues`);
        });
        
        console.log(`\\nTotal: ${totalLeagues} leagues across ${result.sections.length} sections`);
        
        return result;
    }

    async close() {
        console.log('Closing browser...');
        if (this.browser) {
            await this.browser.close();
        }
        console.log('Browser closed');
    }

    async run() {
        try {
            await this.init();
            await this.loadHomepage();
            const data = await this.extractAllLeagues();
            
            console.log('\\nSaving data to file...');
            const fs = await import('fs');
            fs.writeFileSync('fotmob_data_fast.json', JSON.stringify(data, null, 2));
            console.log('Data saved to fotmob_data_fast.json');
            
            return data;
        } catch (error) {
            console.error('Crawler error:', error);
            throw error;
        } finally {
            await this.close();
        }
    }
}

// Run the crawler
const crawler = new FastFotmobCrawler();
crawler.run().catch(console.error);