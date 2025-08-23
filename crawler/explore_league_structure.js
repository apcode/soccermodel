import puppeteer from 'puppeteer';

class LeagueStructureExplorer {
    constructor() {
        this.browser = null;
        this.page = null;
        this.baseUrl = 'https://www.fotmob.com';
    }

    async init() {
        console.log('Initializing league structure explorer...');
        
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

        // Add random delays to mimic human behavior
        this.page.on('response', async () => {
            await new Promise(resolve => setTimeout(resolve, Math.random() * 100 + 50));
        });

        console.log('Explorer initialized successfully');
    }

    async humanDelay(min = 500, max = 2000) {
        const delay = Math.random() * (max - min) + min;
        await new Promise(resolve => setTimeout(resolve, delay));
    }

    async navigateToLeague(leagueUrl) {
        console.log(`Navigating to league: ${leagueUrl}`);
        
        const fullUrl = this.baseUrl + leagueUrl;
        await this.page.goto(fullUrl, {
            waitUntil: 'networkidle2',
            timeout: 30000
        });

        await this.humanDelay();
        console.log('League page loaded');
    }

    async exploreSeasonStructure() {
        console.log('Looking for season selector...');
        
        const seasonInfo = await this.page.evaluate(() => {
            const result = {
                seasonSelector: null,
                availableSeasons: [],
                currentSeason: null
            };

            // Look for season dropdown/selector
            const selectors = [
                'select',
                '[role="combobox"]',
                '.season-selector',
                '[data-testid="season-selector"]'
            ];

            for (const selector of selectors) {
                const element = document.querySelector(selector);
                if (element) {
                    result.seasonSelector = selector;
                    
                    // Get options if it's a select
                    if (element.tagName === 'SELECT') {
                        const options = element.querySelectorAll('option');
                        options.forEach(option => {
                            result.availableSeasons.push({
                                value: option.value,
                                text: option.textContent.trim()
                            });
                        });
                        result.currentSeason = element.value;
                    }
                    break;
                }
            }

            // Look for any text that might indicate seasons
            const allText = document.body.textContent;
            const seasonMatches = allText.match(/20\d{2}\/\d{2}/g) || allText.match(/20\d{2}-\d{2}/g);
            if (seasonMatches) {
                result.foundSeasonText = [...new Set(seasonMatches)];
            }

            return result;
        });

        console.log('Season structure:', JSON.stringify(seasonInfo, null, 2));
        return seasonInfo;
    }

    async navigateToMatches() {
        console.log('Looking for matches/fixtures tab...');
        
        const matchesFound = await this.page.evaluate(() => {
            const links = document.querySelectorAll('a');
            for (const link of links) {
                const text = link.textContent.trim().toLowerCase();
                if (text.includes('matches') || text.includes('fixtures')) {
                    return {
                        found: true,
                        url: link.href,
                        text: link.textContent.trim()
                    };
                }
            }
            return { found: false };
        });

        if (matchesFound.found) {
            console.log(`Found matches link: ${matchesFound.text} -> ${matchesFound.url}`);
            
            // Click the matches link
            await this.page.click('a[href*="matches"]');
            await this.humanDelay();
            
            console.log('Navigated to matches page');
            return true;
        }
        
        console.log('No matches link found');
        return false;
    }

    async exploreMatchesStructure() {
        console.log('Analyzing matches page structure...');
        
        const matchesStructure = await this.page.evaluate(() => {
            const result = {
                matchLinks: [],
                dateNavigation: null,
                pagination: null,
                filters: []
            };

            // Look for individual match links
            const matchLinks = document.querySelectorAll('a[href*="/match/"]');
            matchLinks.forEach(link => {
                result.matchLinks.push({
                    url: link.href,
                    text: link.textContent.trim()
                });
            });

            // Look for date navigation
            const dateElements = document.querySelectorAll('[data-date], .date-picker, .calendar, [class*="date"]');
            if (dateElements.length > 0) {
                result.dateNavigation = `Found ${dateElements.length} date-related elements`;
            }

            // Look for pagination
            const paginationElements = document.querySelectorAll('[class*="pagination"], .page-nav, [aria-label*="page"]');
            if (paginationElements.length > 0) {
                result.pagination = `Found ${paginationElements.length} pagination elements`;
            }

            // Look for any filters or controls
            const filterElements = document.querySelectorAll('select, [role="combobox"], .filter, [class*="filter"]');
            filterElements.forEach(el => {
                result.filters.push({
                    tag: el.tagName,
                    class: el.className,
                    text: el.textContent.trim().substring(0, 50)
                });
            });

            return result;
        });

        console.log('Matches structure:', JSON.stringify(matchesStructure, null, 2));
        return matchesStructure;
    }

    async exploreIndividualMatch() {
        console.log('Exploring individual match page...');
        
        // Get the first match link
        const firstMatchUrl = await this.page.evaluate(() => {
            const link = document.querySelector('a[href*="/match/"]');
            return link ? link.href : null;
        });

        if (firstMatchUrl) {
            console.log(`Navigating to match: ${firstMatchUrl}`);
            
            await this.page.goto(firstMatchUrl, {
                waitUntil: 'networkidle2',
                timeout: 30000
            });

            await this.humanDelay();

            const matchStructure = await this.page.evaluate(() => {
                const result = {
                    matchInfo: {},
                    availableStats: [],
                    playerTables: [],
                    events: []
                };

                // Look for basic match info
                const scoreElements = document.querySelectorAll('[class*="score"], .match-score');
                if (scoreElements.length > 0) {
                    result.matchInfo.score = scoreElements[0].textContent.trim();
                }

                // Look for team names
                const teamElements = document.querySelectorAll('[class*="team"], .team-name');
                teamElements.forEach(el => {
                    const text = el.textContent.trim();
                    if (text && !result.matchInfo.teams) {
                        result.matchInfo.teams = [];
                    }
                    if (text && result.matchInfo.teams && result.matchInfo.teams.length < 2) {
                        result.matchInfo.teams.push(text);
                    }
                });

                // Look for stats sections
                const statSections = document.querySelectorAll('[class*="stat"], .statistics, [data-testid*="stat"]');
                statSections.forEach(section => {
                    const heading = section.querySelector('h2, h3, .heading, [class*="title"]');
                    if (heading) {
                        result.availableStats.push(heading.textContent.trim());
                    }
                });

                // Look for player tables
                const tables = document.querySelectorAll('table, [role="table"]');
                tables.forEach(table => {
                    const headers = table.querySelectorAll('th, [role="columnheader"]');
                    if (headers.length > 0) {
                        const tableHeaders = Array.from(headers).map(h => h.textContent.trim());
                        result.playerTables.push(tableHeaders);
                    }
                });

                // Look for match events (goals, cards, etc.)
                const eventElements = document.querySelectorAll('[class*="event"], .timeline, [class*="incident"]');
                eventElements.forEach(event => {
                    const eventText = event.textContent.trim();
                    if (eventText.length > 0 && eventText.length < 200) {
                        result.events.push(eventText);
                    }
                });

                return result;
            });

            console.log('Match page structure:', JSON.stringify(matchStructure, null, 2));
            return matchStructure;
        }

        return null;
    }

    async close() {
        console.log('Closing browser...');
        if (this.browser) {
            await this.browser.close();
        }
    }

    async exploreLeague(leagueUrl) {
        try {
            await this.navigateToLeague(leagueUrl);
            
            const exploration = {
                leagueUrl,
                seasonStructure: await this.exploreSeasonStructure(),
                matchesNavigation: null,
                matchesStructure: null,
                sampleMatch: null
            };

            // Try to navigate to matches
            const matchesFound = await this.navigateToMatches();
            exploration.matchesNavigation = matchesFound;

            if (matchesFound) {
                exploration.matchesStructure = await this.exploreMatchesStructure();
                exploration.sampleMatch = await this.exploreIndividualMatch();
            }

            return exploration;

        } catch (error) {
            console.error('Error exploring league:', error);
            return { error: error.message };
        }
    }

    async run() {
        try {
            await this.init();
            
            // Test with Premier League first
            const premierLeagueUrl = '/en-GB/leagues/47/overview/premier-league';
            console.log('Starting exploration with Premier League...');
            
            const results = await this.exploreLeague(premierLeagueUrl);
            
            console.log('\n=== EXPLORATION COMPLETE ===');
            console.log(JSON.stringify(results, null, 2));
            
            // Save results
            const fs = await import('fs');
            fs.writeFileSync('league_structure_exploration.json', JSON.stringify(results, null, 2));
            console.log('\nResults saved to league_structure_exploration.json');
            
            return results;
            
        } catch (error) {
            console.error('Explorer error:', error);
            throw error;
        } finally {
            await this.close();
        }
    }
}

// Run the explorer
const explorer = new LeagueStructureExplorer();
explorer.run().catch(console.error);