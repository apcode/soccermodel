import puppeteer from 'puppeteer';

class FotmobCrawler {
    constructor() {
        this.browser = null;
        this.page = null;
        this.baseUrl = 'https://www.fotmob.com/en-GB';
    }

    async init() {
        console.log('Initializing crawler...');
        
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

        console.log('Crawler initialized successfully');
    }

    async loadHomepage() {
        console.log('Loading fotmob homepage...');
        
        await this.page.goto(this.baseUrl, {
            waitUntil: 'networkidle2',
            timeout: 30000
        });

        await new Promise(resolve => setTimeout(resolve, 500));
        
        console.log('Homepage loaded successfully');
        console.log('Page title:', await this.page.title());
    }

    async loadLeaguesDirectory() {
        console.log('Loading leagues directory...');
        
        // Try direct league directory URLs
        const possibleLeagueUrls = [
            `${this.baseUrl}/all-leagues`
        ];
        
        for (const url of possibleLeagueUrls) {
            try {
                console.log(`Trying: ${url}`);
                await this.page.goto(url, {
                    waitUntil: 'networkidle2',
                    timeout: 20000
                });
                
                const title = await this.page.title();
                console.log(`Loaded: ${title}`);
                
                // Check if this page has more leagues
                const leagueCount = await this.page.evaluate(() => {
                    return document.querySelectorAll('a[href*="/leagues/"]').length;
                });
                
                console.log(`Found ${leagueCount} league links on this page`);
                
                if (leagueCount > 50) { // If we found more leagues, use this page
                    console.log('This page has more leagues, using it for extraction');
                    return;
                }
                
            } catch (error) {
                console.log(`Failed to load ${url}: ${error.message}`);
            }
        }
        
        console.log('Could not find a better leagues directory, using current page');
    }

    async extractCountriesAndLeagues() {
        console.log('Extracting countries and leagues from All Leagues section...');
        
        // First, try to find and navigate to the "All Leagues" section
        try {
            console.log('Looking for leagues menu...');
            
            // Wait for the page to be fully loaded
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Take a screenshot for debugging
            await this.page.screenshot({ path: 'homepage.png' });
            console.log('Screenshot saved as homepage.png');
            
            // Look specifically for the "All leagues" dropdown
            const found = await this.page.evaluate(() => {
                // Look for the "All leagues" dropdown specifically
                const allElements = document.querySelectorAll('*');
                for (let el of allElements) {
                    const text = el.textContent?.trim();
                    // Look for exact "All leagues" text
                    if (text === 'All leagues' || text === 'All Leagues') {
                        console.log('Found "All leagues" dropdown:', text);
                        // Check if it's clickable (has click handler or is a button/link)
                        if (el.tagName === 'BUTTON' || el.tagName === 'A' || el.onclick || el.getAttribute('role') === 'button') {
                            el.click();
                            return true;
                        }
                        // If not directly clickable, try parent elements
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
                
                // Also try looking for elements with class or data attributes related to leagues
                const leagueDropdowns = document.querySelectorAll('[class*="league"], [data-*="league"], [aria-label*="league" i]');
                for (let el of leagueDropdowns) {
                    if (el.textContent?.toLowerCase().includes('all')) {
                        console.log('Found potential dropdown by attribute:', el.textContent.trim());
                        el.click();
                        return true;
                    }
                }
                
                return false;
            });
            
            if (found) {
                console.log('Clicked on leagues menu, waiting for content to load...');
                await new Promise(resolve => setTimeout(resolve, 800)); // Wait for dynamic content
                
                // Take another screenshot to see the expanded menu
                await this.page.screenshot({ path: 'leagues_expanded.png' });
                console.log('Screenshot of expanded leagues saved as leagues_expanded.png');
                
                // Check how many league links are now available
                const leagueCount = await this.page.evaluate(() => {
                    return document.querySelectorAll('a[href*="/leagues/"]').length;
                });
                console.log(`Found ${leagueCount} league links after expansion`);
                
            } else {
                console.log('Could not find leagues menu, will extract from current page');
            }
            
        } catch (error) {
            console.log('Error finding leagues menu:', error.message);
        }
        
        // Now expand each country section and collect leagues
        console.log('Finding and expanding country sections...');
        
        // Get list of visible countries first
        const countries = await this.page.evaluate(() => {
            const countryNames = ['International', 'Albania', 'Algeria', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Chile', 'China', 'Colombia', 'Costa Rica', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Ecuador', 'Egypt', 'El Salvador', 'England', 'Estonia', 'Faroe Islands', 'Finland', 'France', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Guatemala', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Latvia', 'Lebanon', 'Lithuania', 'Luxembourg', 'Malaysia', 'Malta', 'Mexico', 'Moldova', 'Montenegro', 'Morocco', 'Myanmar', 'Netherlands', 'New Zealand', 'Nicaragua', 'North Macedonia', 'Northern Ireland', 'Norway', 'Oman', 'Pakistan', 'Palestine', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'San Marino', 'Saudi Arabia', 'Scotland', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sweden', 'Switzerland', 'Thailand', 'Tunisia', 'Turkey', 'Turkiye', 'UAE', 'Ukraine', 'United States', 'Uruguay', 'USA', 'Uzbekistan', 'Venezuela', 'Vietnam', 'Wales', 'Zimbabwe'];
            
            const foundCountries = [];
            const allElements = document.querySelectorAll('*');
            
            for (let el of allElements) {
                const text = el.textContent?.trim();
                if (text && countryNames.includes(text)) {
                    // Check if this looks like a clickable country section
                    const rect = el.getBoundingClientRect();
                    if (rect.height > 0 && rect.width > 0) { // Visible element
                        foundCountries.push(text);
                    }
                }
            }
            
            // Remove duplicates
            return [...new Set(foundCountries)];
        });
        
        console.log(`Found ${countries.length} countries: ${countries.slice(0, 5).join(', ')}${countries.length > 5 ? '...' : ''}`);
        
        const result = {
            sections: [],
            debug: [`Found ${countries.length} countries: ${countries.join(', ')}`]
        };
        
        // Now properly expand each country by clicking their arrows
        console.log('Expanding country sections by clicking their arrows...');
        
        // Process all countries for complete dataset
        const countriesToTest = countries;
        
        for (let i = 0; i < countriesToTest.length; i++) {
            const countryName = countriesToTest[i];
            try {
                console.log(`[${i + 1}/${countriesToTest.length}] Expanding ${countryName} by clicking arrow...`);
                
                // Find and click the arrow next to this country
                const expanded = await this.page.evaluate((country) => {
                    // Look for the country text and find the arrow nearby
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
                                // Sometimes the whole row is clickable
                                if (parent && (parent.tagName === 'BUTTON' || parent.onclick || parent.style.cursor === 'pointer')) {
                                    arrowElement = parent;
                                } else if (el.tagName === 'BUTTON' || el.onclick || el.style.cursor === 'pointer') {
                                    arrowElement = el;
                                }
                            }
                            
                            if (arrowElement) {
                                console.log(`Found clickable element for ${country}`);
                                arrowElement.click();
                                return true;
                            }
                        }
                    }
                    
                    return false;
                }, countryName);
                
                if (expanded) {
                    // Wait for country leagues to load (super fast)
                    await new Promise(resolve => setTimeout(resolve, 100));
                    
                    // Extract leagues that appeared for this specific country by looking for newly visible leagues
                    const countryLeagues = await this.page.evaluate((country) => {
                        const leagues = [];
                        const leagueLinks = document.querySelectorAll('a[href*="/leagues/"]');
                        
                        // Find the country element to establish context
                        let countryElement = null;
                        const allElements = document.querySelectorAll('*');
                        for (let el of allElements) {
                            if (el.textContent?.trim() === country) {
                                countryElement = el;
                                break;
                            }
                        }
                        
                        if (countryElement) {
                            const countryRect = countryElement.getBoundingClientRect();
                            
                            leagueLinks.forEach(link => {
                                const text = link.textContent?.trim();
                                const href = link.getAttribute('href');
                                const rect = link.getBoundingClientRect();
                                
                                // Only include visible leagues that appear after this country in the DOM
                                if (text && href && rect.height > 0 && rect.width > 0) {
                                    // Check if this league appears below the country heading and within reasonable distance
                                    const isAfterCountry = rect.top > countryRect.top;
                                    const isCloseToCountry = rect.top - countryRect.top < 800; // Within 800px below country
                                    
                                    // Also check if there's another country between this league and our target country
                                    let hasInterveningCountry = false;
                                    const allCountries = ['International', 'Albania', 'Algeria', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Chile', 'China', 'Colombia', 'Costa Rica', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Ecuador', 'Egypt', 'El Salvador', 'England', 'Estonia', 'Faroe Islands', 'Finland', 'France', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Guatemala', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Latvia', 'Lebanon', 'Lithuania', 'Luxembourg', 'Malaysia', 'Malta', 'Mexico', 'Moldova', 'Montenegro', 'Morocco', 'Myanmar', 'Netherlands', 'New Zealand', 'Nicaragua', 'North Macedonia', 'Northern Ireland', 'Norway', 'Oman', 'Pakistan', 'Palestine', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'San Marino', 'Saudi Arabia', 'Scotland', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sweden', 'Switzerland', 'Thailand', 'Tunisia', 'Turkey', 'Turkiye', 'UAE', 'Ukraine', 'United States', 'Uruguay', 'USA', 'Uzbekistan', 'Venezuela', 'Vietnam', 'Wales', 'Zimbabwe'];
                                    
                                    for (let otherCountry of allCountries) {
                                        if (otherCountry !== country) {
                                            const otherCountryElements = Array.from(allElements).filter(el => el.textContent?.trim() === otherCountry);
                                            for (let otherEl of otherCountryElements) {
                                                const otherRect = otherEl.getBoundingClientRect();
                                                if (otherRect.top > countryRect.top && otherRect.top < rect.top) {
                                                    hasInterveningCountry = true;
                                                    break;
                                                }
                                            }
                                            if (hasInterveningCountry) break;
                                        }
                                    }
                                    
                                    if (isAfterCountry && isCloseToCountry && !hasInterveningCountry) {
                                        leagues.push({
                                            name: text,
                                            url: href
                                        });
                                    }
                                }
                            });
                        }
                        
                        return leagues;
                    }, countryName);
                    
                    // Also remove duplicates from previous countries
                    const newLeagues = countryLeagues.filter(league => 
                        !result.sections.some(section => 
                            section.leagues.some(existingLeague => existingLeague.url === league.url)
                        )
                    );
                    
                    if (newLeagues.length > 0) {
                        result.sections.push({
                            section: countryName,
                            leagues: newLeagues
                        });
                        console.log(`Found ${newLeagues.length} new leagues for ${countryName}`);
                    } else {
                        console.log(`No new leagues found for ${countryName} (${countryLeagues.length} total, all duplicates)`);
                    }
                    
                    // Screenshots disabled for maximum speed
                    
                } else {
                    console.log(`Could not find/click arrow for ${countryName}`);
                }
                
            } catch (error) {
                console.log(`Error expanding ${countryName}: ${error.message}`);
            }
        }
        
        // If we didn't get good results, extract all visible leagues
        if (result.sections.length === 0) {
            console.log('Extracting all visible leagues as fallback...');
            const allLeagues = await this.page.evaluate(() => {
                const leagues = [];
                const leagueLinks = document.querySelectorAll('a[href*="/leagues/"]');
                
                leagueLinks.forEach(link => {
                    const text = link.textContent?.trim();
                    const href = link.getAttribute('href');
                    if (text && href) {
                        leagues.push({
                            name: text,
                            url: href
                        });
                    }
                });
                
                return leagues;
            });
            
            result.sections = [{
                section: 'All Leagues',
                leagues: allLeagues
            }];
        }
        
        const data = result;

        let totalLeagues = 0;
        data.sections.forEach(section => totalLeagues += section.leagues.length);
        
        console.log(`Found ${data.sections.length} sections with ${totalLeagues} total leagues`);
        return data;
    }

    async close() {
        if (this.browser) {
            await this.browser.close();
            console.log('Browser closed');
        }
    }
}

async function main() {
    const crawler = new FotmobCrawler();
    
    try {
        await crawler.init();
        await crawler.loadHomepage();
        await crawler.loadLeaguesDirectory();
        
        const data = await crawler.extractCountriesAndLeagues();
        
        console.log('\nExtracted Data Summary:');
        if (data.sections && data.sections.length > 0) {
            data.sections.forEach(section => {
                console.log(`${section.section}: ${section.leagues.length} leagues`);
            });
            
            const totalLeagues = data.sections.reduce((sum, section) => sum + section.leagues.length, 0);
            console.log(`\nTotal: ${totalLeagues} leagues across ${data.sections.length} sections`);
        }
        
        if (data.debug) {
            console.log('\nDebug info:', data.debug);
        }
        
        console.log('\nSaving data to file...');
        const fs = await import('fs');
        await fs.promises.writeFile('fotmob_data.json', JSON.stringify(data, null, 2));
        console.log('Data saved to fotmob_data.json');
        
    } catch (error) {
        console.error('Error:', error);
    } finally {
        await crawler.close();
    }
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}

export default FotmobCrawler;
