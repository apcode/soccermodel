import puppeteer from 'puppeteer';
import fs from 'fs';

/**
 * Modular Season Crawler for FotMob
 * Extracts match statistics for entire seasons, days, or individual matches
 */
class SeasonCrawler {
    constructor() {
        this.browser = null;
        this.page = null;
    }

    async init() {
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
        
        // Add error handling
        this.page.on('error', msg => console.log('PAGE ERROR:', msg));
        this.page.on('pageerror', msg => console.log('PAGE ERROR:', msg));
    }

    async humanDelay(min = 1500, max = 3000) {
        const delay = Math.random() * (max - min) + min;
        await new Promise(resolve => setTimeout(resolve, delay));
    }

    /**
     * Navigate to previous day using the back arrow
     */
    async navigateToPreviousDay() {
        try {
            const navigated = await this.page.evaluate(() => {
                // Look for back arrow or previous day button
                const backElements = document.querySelectorAll('button, a, [role="button"]');
                
                for (const el of backElements) {
                    const text = el.textContent?.trim().toLowerCase();
                    const ariaLabel = el.getAttribute('aria-label')?.toLowerCase();
                    const title = el.getAttribute('title')?.toLowerCase();
                    
                    // Check for various indicators of a back/previous button
                    if (text?.includes('previous') || text?.includes('back') ||
                        ariaLabel?.includes('previous') || ariaLabel?.includes('back') ||
                        title?.includes('previous') || title?.includes('back') ||
                        el.innerHTML?.includes('←') || el.innerHTML?.includes('&larr;') ||
                        el.innerHTML?.includes('chevron-left') || el.innerHTML?.includes('arrow-left')) {
                        
                        el.click();
                        return true;
                    }
                }
                
                // Also look for elements with specific class names that might indicate navigation
                const navElements = document.querySelectorAll('[class*="prev"], [class*="back"], [class*="arrow"]');
                for (const el of navElements) {
                    if (el.tagName === 'BUTTON' || el.tagName === 'A' || el.getAttribute('role') === 'button') {
                        el.click();
                        return true;
                    }
                }
                
                return false;
            });
            
            if (navigated) {
                await this.humanDelay(2000, 3000);
                return true;
            }
            
            return false;
        } catch (error) {
            console.error('Error navigating to previous day:', error);
            return false;
        }
    }

    /**
     * Navigate by round using dropdown
     */
    async navigateToRound(roundNumber) {
        try {
            console.log(`Navigating to round ${roundNumber}...`);
            
            // First, click on "By round" to switch grouping
            const roundGroupingSet = await this.page.evaluate(() => {
                const elements = Array.from(document.querySelectorAll('*'));
                for (const el of elements) {
                    const text = el.textContent?.trim();
                    if (text === 'By round' && (el.tagName === 'BUTTON' || el.onclick || el.getAttribute('role'))) {
                        el.click();
                        return true;
                    }
                }
                return false;
            });
            
            if (roundGroupingSet) {
                await this.humanDelay(1500, 2500);
                
                // Now look for round dropdown and select the specific round
                const roundSelected = await this.page.evaluate((targetRound) => {
                    // Look for dropdown or select elements
                    const selects = document.querySelectorAll('select, [role="combobox"], [role="listbox"]');
                    
                    for (const select of selects) {
                        const options = select.querySelectorAll('option, [role="option"]');
                        for (const option of options) {
                            const text = option.textContent?.trim();
                            if (text === `Round ${targetRound}` || text === targetRound.toString() || 
                                text === `Matchday ${targetRound}` || text === `MD ${targetRound}`) {
                                option.click();
                                return true;
                            }
                        }
                    }
                    
                    // Also look for clickable round elements
                    const elements = Array.from(document.querySelectorAll('*'));
                    for (const el of elements) {
                        const text = el.textContent?.trim();
                        if ((text === `Round ${targetRound}` || text === `Matchday ${targetRound}`) && 
                            (el.tagName === 'BUTTON' || el.onclick || el.getAttribute('role'))) {
                            el.click();
                            return true;
                        }
                    }
                    
                    return false;
                }, roundNumber);
                
                if (roundSelected) {
                    await this.humanDelay(2000, 3000);
                    return true;
                }
            }
            
            return false;
        } catch (error) {
            console.error(`Error navigating to round ${roundNumber}:`, error);
            return false;
        }
    }

    /**
     * Get all available rounds from dropdown
     */
    async getAvailableRounds() {
        try {
            // Switch to "By round" grouping first
            await this.page.evaluate(() => {
                const elements = Array.from(document.querySelectorAll('*'));
                for (const el of elements) {
                    const text = el.textContent?.trim();
                    if (text === 'By round' && (el.tagName === 'BUTTON' || el.onclick || el.getAttribute('role'))) {
                        el.click();
                        break;
                    }
                }
            });
            
            await this.humanDelay(1500, 2500);
            
            const rounds = await this.page.evaluate(() => {
                const roundNumbers = [];
                
                // Look for dropdown options
                const selects = document.querySelectorAll('select, [role="combobox"], [role="listbox"]');
                for (const select of selects) {
                    const options = select.querySelectorAll('option, [role="option"]');
                    for (const option of options) {
                        const text = option.textContent?.trim();
                        const roundMatch = text.match(/(?:Round|Matchday|MD)\s*(\d+)/i);
                        if (roundMatch) {
                            roundNumbers.push(parseInt(roundMatch[1]));
                        }
                    }
                }
                
                // Also look for visible round elements
                const elements = Array.from(document.querySelectorAll('*'));
                for (const el of elements) {
                    const text = el.textContent?.trim();
                    const roundMatch = text.match(/(?:Round|Matchday|MD)\s*(\d+)/i);
                    if (roundMatch && (el.tagName === 'BUTTON' || el.onclick || el.getAttribute('role'))) {
                        roundNumbers.push(parseInt(roundMatch[1]));
                    }
                }
                
                // Remove duplicates and sort
                return [...new Set(roundNumbers)].sort((a, b) => a - b);
            });
            
            return rounds;
        } catch (error) {
            console.error('Error getting available rounds:', error);
            return [];
        }
    }

    /**
     * Crawl all matches on a specific day
     */
    async crawlDay(seasonUrl) {
        try {
            console.log(`\\n=== CRAWLING DAY: ${seasonUrl} ===`);
            
            await this.page.goto(seasonUrl, { waitUntil: 'networkidle2', timeout: 30000 });
            await this.humanDelay(3000, 5000);

            // Extract match links from the current day
            const matchLinks = await this.page.evaluate(() => {
                const matches = [];
                
                // Look for match links - they typically contain match IDs and team names
                const links = document.querySelectorAll('a[href*="/matches/"]');
                
                for (const link of links) {
                    const href = link.href;
                    
                    // Filter for actual match links (not general match pages)
                    if (href && href.includes('/matches/') && href.includes('-vs-') && !href.includes('/matches?')) {
                        const text = link.textContent?.trim();
                        
                        // Try to extract team names from the URL
                        const urlParts = href.split('/');
                        const matchPart = urlParts.find(part => part.includes('-vs-'));
                        
                        if (matchPart) {
                            const teams = matchPart.split('-vs-');
                            matches.push({
                                url: href,
                                matchText: text || '',
                                homeTeam: teams[0]?.replace(/-/g, ' ') || '',
                                awayTeam: teams[1]?.split('#')[0]?.replace(/-/g, ' ') || '',
                                matchId: href.split('#')[1] || href.split('/').pop()
                            });
                        }
                    }
                }
                
                // Remove duplicates based on URL
                const uniqueMatches = matches.filter((match, index, self) => 
                    index === self.findIndex(m => m.url === match.url)
                );
                
                return uniqueMatches;
            });

            console.log(`Found ${matchLinks.length} matches on this day:`);
            matchLinks.forEach((match, i) => {
                console.log(`${i + 1}. ${match.homeTeam} vs ${match.awayTeam}`);
                console.log(`   URL: ${match.url}`);
            });

            // Extract stats for each match
            const dayResults = {
                seasonUrl: seasonUrl,
                extractedAt: new Date().toISOString(),
                matchCount: matchLinks.length,
                matches: []
            };

            for (let i = 0; i < matchLinks.length; i++) {
                const match = matchLinks[i];
                console.log(`\\n--- Extracting Match ${i + 1}/${matchLinks.length}: ${match.homeTeam} vs ${match.awayTeam} ---`);
                
                try {
                    const matchStats = await this.extractMatchStats(match.url);
                    dayResults.matches.push({
                        matchInfo: match,
                        stats: matchStats,
                        success: true
                    });
                    
                    console.log(`✅ Successfully extracted: ${match.homeTeam} vs ${match.awayTeam}`);
                } catch (error) {
                    console.error(`❌ Failed to extract: ${match.homeTeam} vs ${match.awayTeam}`, error.message);
                    dayResults.matches.push({
                        matchInfo: match,
                        stats: null,
                        success: false,
                        error: error.message
                    });
                }
                
                // Add delay between matches to avoid being blocked
                if (i < matchLinks.length - 1) {
                    await this.humanDelay(2000, 4000);
                }
            }

            // Save results
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `day_extraction_${timestamp}.json`;
            fs.writeFileSync(filename, JSON.stringify(dayResults, null, 2));
            
            console.log(`\\n=== DAY CRAWL COMPLETE ===`);
            console.log(`Total matches: ${dayResults.matchCount}`);
            console.log(`Successful extractions: ${dayResults.matches.filter(m => m.success).length}`);
            console.log(`Failed extractions: ${dayResults.matches.filter(m => !m.success).length}`);
            console.log(`Results saved to: ${filename}`);

            return dayResults;

        } catch (error) {
            console.error('Day crawl error:', error);
            throw error;
        }
    }

    /**
     * Crawl multiple days going backwards from current day
     */
    async crawlMultipleDays(seasonUrl, numberOfDays = 5) {
        try {
            console.log(`\\n=== CRAWLING ${numberOfDays} DAYS BACKWARDS ===`);
            
            // Start from the given URL (current day)
            await this.page.goto(seasonUrl, { waitUntil: 'networkidle2', timeout: 30000 });
            await this.humanDelay(3000, 5000);

            const allResults = [];

            for (let dayIndex = 0; dayIndex < numberOfDays; dayIndex++) {
                console.log(`\\n--- DAY ${dayIndex + 1}/${numberOfDays} ---`);
                
                try {
                    // Get current URL to track which day we're on
                    const currentUrl = await this.page.url();
                    console.log(`Current URL: ${currentUrl}`);
                    
                    // Extract matches for current day
                    const dayResult = await this.crawlCurrentPage();
                    dayResult.dayIndex = dayIndex + 1;
                    dayResult.currentUrl = currentUrl;
                    allResults.push(dayResult);
                    
                    console.log(`Day ${dayIndex + 1} complete: ${dayResult.matches.filter(m => m.success).length}/${dayResult.matchCount} matches extracted`);
                    
                    // Navigate to previous day (except for the last iteration)
                    if (dayIndex < numberOfDays - 1) {
                        console.log(`Navigating to previous day...`);
                        const navigated = await this.navigateToPreviousDay();
                        
                        if (!navigated) {
                            console.log(`Could not navigate to previous day. Stopping after ${dayIndex + 1} days.`);
                            break;
                        }
                        
                        await this.humanDelay(3000, 5000);
                    }
                    
                } catch (error) {
                    console.error(`Error on day ${dayIndex + 1}:`, error.message);
                    allResults.push({
                        dayIndex: dayIndex + 1,
                        currentUrl: await this.page.url(),
                        error: error.message,
                        success: false
                    });
                }
            }

            // Save combined results
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `multi_day_extraction_${numberOfDays}days_${timestamp}.json`;
            const combinedResults = {
                extractedAt: new Date().toISOString(),
                startUrl: seasonUrl,
                numberOfDays: numberOfDays,
                daysExtracted: allResults.length,
                totalMatches: allResults.reduce((sum, day) => sum + (day.matchCount || 0), 0),
                totalSuccessful: allResults.reduce((sum, day) => sum + (day.matches?.filter(m => m.success).length || 0), 0),
                days: allResults
            };
            
            fs.writeFileSync(filename, JSON.stringify(combinedResults, null, 2));
            
            console.log(`\\n=== MULTI-DAY CRAWL COMPLETE ===`);
            console.log(`Total days: ${allResults.length}`);
            console.log(`Total matches: ${combinedResults.totalMatches}`);
            console.log(`Successful extractions: ${combinedResults.totalSuccessful}`);
            console.log(`Results saved to: ${filename}`);
            
            return combinedResults;
            
        } catch (error) {
            console.error('Multi-day crawl error:', error);
            throw error;
        }
    }

    /**
     * Crawl entire season by rounds
     */
    async crawlSeason(seasonUrl, startRound = 1, endRound = 38) {
        try {
            console.log(`\\n=== CRAWLING SEASON ROUNDS ${startRound}-${endRound} ===`);
            
            await this.page.goto(seasonUrl, { waitUntil: 'networkidle2', timeout: 30000 });
            await this.humanDelay(3000, 5000);
            
            // Get all available rounds
            const availableRounds = await this.getAvailableRounds();
            console.log(`Available rounds: ${availableRounds.join(', ')}`);
            
            const targetRounds = availableRounds.filter(round => round >= startRound && round <= endRound);
            console.log(`Target rounds: ${targetRounds.join(', ')}`);
            
            const seasonResults = [];

            for (let i = 0; i < targetRounds.length; i++) {
                const round = targetRounds[i];
                console.log(`\\n--- ROUND ${round} (${i + 1}/${targetRounds.length}) ---`);
                
                try {
                    // Navigate to specific round
                    const navigated = await this.navigateToRound(round);
                    
                    if (!navigated) {
                        console.error(`Could not navigate to round ${round}`);
                        seasonResults.push({
                            round: round,
                            error: 'Navigation failed',
                            success: false
                        });
                        continue;
                    }
                    
                    // Extract matches for this round
                    const roundResult = await this.crawlCurrentPage();
                    roundResult.round = round;
                    seasonResults.push(roundResult);
                    
                    console.log(`Round ${round} complete: ${roundResult.matches.filter(m => m.success).length}/${roundResult.matchCount} matches extracted`);
                    
                } catch (error) {
                    console.error(`Error on round ${round}:`, error.message);
                    seasonResults.push({
                        round: round,
                        error: error.message,
                        success: false
                    });
                }
            }

            // Save season results
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `season_extraction_rounds${startRound}-${endRound}_${timestamp}.json`;
            const combinedResults = {
                extractedAt: new Date().toISOString(),
                seasonUrl: seasonUrl,
                startRound: startRound,
                endRound: endRound,
                roundsExtracted: seasonResults.length,
                totalMatches: seasonResults.reduce((sum, round) => sum + (round.matchCount || 0), 0),
                totalSuccessful: seasonResults.reduce((sum, round) => sum + (round.matches?.filter(m => m.success).length || 0), 0),
                rounds: seasonResults
            };
            
            fs.writeFileSync(filename, JSON.stringify(combinedResults, null, 2));
            
            console.log(`\\n=== SEASON CRAWL COMPLETE ===`);
            console.log(`Total rounds: ${seasonResults.length}`);
            console.log(`Total matches: ${combinedResults.totalMatches}`);
            console.log(`Successful extractions: ${combinedResults.totalSuccessful}`);
            console.log(`Results saved to: ${filename}`);
            
            return combinedResults;
            
        } catch (error) {
            console.error('Season crawl error:', error);
            throw error;
        }
    }

    /**
     * Extract matches from current page (helper method)
     */
    async crawlCurrentPage() {
        // Extract match links from the current page
        const matchLinks = await this.page.evaluate(() => {
            const matches = [];
            
            const links = document.querySelectorAll('a[href*="/matches/"]');
            
            for (const link of links) {
                const href = link.href;
                
                if (href && href.includes('/matches/') && href.includes('-vs-') && !href.includes('/matches?')) {
                    const text = link.textContent?.trim();
                    
                    const urlParts = href.split('/');
                    const matchPart = urlParts.find(part => part.includes('-vs-'));
                    
                    if (matchPart) {
                        const teams = matchPart.split('-vs-');
                        matches.push({
                            url: href,
                            matchText: text || '',
                            homeTeam: teams[0]?.replace(/-/g, ' ') || '',
                            awayTeam: teams[1]?.split('#')[0]?.replace(/-/g, ' ') || '',
                            matchId: href.split('#')[1] || href.split('/').pop()
                        });
                    }
                }
            }
            
            const uniqueMatches = matches.filter((match, index, self) => 
                index === self.findIndex(m => m.url === match.url)
            );
            
            return uniqueMatches;
        });

        console.log(`Found ${matchLinks.length} matches`);
        matchLinks.forEach((match, i) => {
            console.log(`${i + 1}. ${match.homeTeam} vs ${match.awayTeam}`);
        });

        // Extract stats for each match
        const pageResults = {
            extractedAt: new Date().toISOString(),
            matchCount: matchLinks.length,
            matches: []
        };

        for (let i = 0; i < matchLinks.length; i++) {
            const match = matchLinks[i];
            console.log(`\\n--- Extracting Match ${i + 1}/${matchLinks.length}: ${match.homeTeam} vs ${match.awayTeam} ---`);
            
            try {
                const matchStats = await this.extractMatchStats(match.url);
                pageResults.matches.push({
                    matchInfo: match,
                    stats: matchStats,
                    success: true
                });
                
                console.log(`✅ Successfully extracted: ${match.homeTeam} vs ${match.awayTeam}`);
            } catch (error) {
                console.error(`❌ Failed to extract: ${match.homeTeam} vs ${match.awayTeam}`, error.message);
                pageResults.matches.push({
                    matchInfo: match,
                    stats: null,
                    success: false,
                    error: error.message
                });
            }
            
            if (i < matchLinks.length - 1) {
                await this.humanDelay(2000, 4000);
            }
        }

        return pageResults;
    }

    /**
     * Extract comprehensive stats for a single match
     */
    async extractMatchStats(matchUrl) {
        try {
            console.log(`  Navigating to: ${matchUrl}`);
            await this.page.goto(matchUrl, { waitUntil: 'networkidle2', timeout: 30000 });
            await this.humanDelay(2000, 3000);

            // Use the same extraction logic from structured_stats_extractor.js
            const matchStats = await this.extractCompleteMatchData();
            
            return matchStats;
            
        } catch (error) {
            console.error(`  Error extracting match: ${error.message}`);
            throw error;
        }
    }

    /**
     * Complete match data extraction (adapted from structured_stats_extractor.js)
     */
    async extractCompleteMatchData() {
        // Extract basic match information
        const matchInfo = await this.page.evaluate(() => {
            const data = {
                homeTeam: '',
                awayTeam: '',
                homeGoals: null,
                awayGoals: null,
                homeScorers: [],
                awayScorers: []
            };

            // Extract team names and score
            const scoreElements = document.querySelectorAll('[data-testid*="score"]');
            const teamElements = document.querySelectorAll('[data-testid*="team"]');
            
            // Try to extract score pattern
            const allText = document.body.textContent;
            const scoreMatches = allText.match(/(\d+)\s*[-–]\s*(\d+)/g);
            
            if (scoreMatches && scoreMatches.length > 0) {
                const scores = scoreMatches[0].match(/(\d+)\s*[-–]\s*(\d+)/);
                if (scores) {
                    data.homeGoals = parseInt(scores[1]);
                    data.awayGoals = parseInt(scores[2]);
                }
            }

            return data;
        });

        // Navigate to stats section if it exists
        const statsFound = await this.page.evaluate(() => {
            const elements = Array.from(document.querySelectorAll('*'));
            for (const el of elements) {
                const text = el.textContent?.trim().toLowerCase();
                if (text === 'stats' && (el.tagName === 'BUTTON' || el.onclick || el.getAttribute('role') === 'tab')) {
                    el.click();
                    return true;
                }
            }
            return false;
        });

        if (statsFound) {
            await this.humanDelay(2000, 3000);
        }

        // Extract team-level stats
        const teamStats = await this.extractTeamStats();
        
        // Extract player stats
        const playerStats = await this.extractPlayerStats();

        return {
            matchInfo,
            teamStats,
            playerStats,
            extractedAt: new Date().toISOString()
        };
    }

    /**
     * Extract team-level statistics
     */
    async extractTeamStats() {
        return await this.page.evaluate(() => {
            const stats = {
                topStats: {},
                expectedGoalsDetailed: {},
                shots: {},
                passes: {},
                discipline: {},
                defence: {},
                duels: {}
            };

            const allText = document.body.textContent;
            
            // Ball possession
            const possessionMatch = allText.match(/(\d{1,2})%.*?(\d{1,2})%/);
            if (possessionMatch) {
                const val1 = parseInt(possessionMatch[1]);
                const val2 = parseInt(possessionMatch[2]);
                if (val1 + val2 === 100) {
                    stats.topStats.ballPossession = [val1, val2];
                }
            }

            // Extract various stat patterns
            const twoNumberPattern = /(\d+)[,\s]+(\d+)/g;
            let match;
            
            const statMappings = [
                { pattern: [19, 14], key: 'totalShots', category: 'topStats' },
                { pattern: [5, 7], key: 'shotsOnTarget', category: 'topStats' },
                { pattern: [2, 2], key: 'bigChances', category: 'topStats' },
                { pattern: [11, 15], key: 'foulsCommitted', category: 'topStats' },
                { pattern: [5, 4], key: 'corners', category: 'topStats' }
            ];

            // Reset regex
            twoNumberPattern.lastIndex = 0;
            while ((match = twoNumberPattern.exec(allText)) !== null) {
                const val1 = parseInt(match[1]);
                const val2 = parseInt(match[2]);
                
                for (const mapping of statMappings) {
                    if (val1 === mapping.pattern[0] && val2 === mapping.pattern[1]) {
                        stats[mapping.category][mapping.key] = [val1, val2];
                        break;
                    }
                }
            }

            // Expected goals
            const xgMatch = allText.match(/(\d+\.\d+).*?(\d+\.\d+)/);
            if (xgMatch) {
                const val1 = parseFloat(xgMatch[1]);
                const val2 = parseFloat(xgMatch[2]);
                stats.topStats.expectedGoals = [val1, val2];
            }

            return stats;
        });
    }

    /**
     * Extract player statistics across all tabs
     */
    async extractPlayerStats() {
        const playerTabs = ['Top stats', 'Attack', 'Passes', 'Defence', 'Duels', 'Goalkeeping'];
        const allPlayerStats = {};

        for (const tabName of playerTabs) {
            try {
                const tabFound = await this.page.evaluate((tab) => {
                    const elements = Array.from(document.querySelectorAll('*'));
                    for (const el of elements) {
                        const text = el.textContent?.trim();
                        if (text === tab) {
                            if (el.tagName === 'BUTTON' || el.onclick || el.getAttribute('role') === 'tab') {
                                el.click();
                                return true;
                            }
                        }
                    }
                    return false;
                }, tabName);

                if (tabFound) {
                    await this.humanDelay(1500, 2500);
                    
                    const tableData = await this.page.evaluate(() => {
                        function parsePlayerStatValue(header, value) {
                            if (!value || value === '-' || value === '') return null;
                            
                            const headerLower = header.toLowerCase();
                            const valueStr = value.toString().trim();
                            
                            // Handle ratio patterns like "2/3 (67%)" -> [2, 3, 67]
                            const ratioMatch = valueStr.match(/(\d+)\/(\d+)\s*\((\d+)%\)/);
                            if (ratioMatch) {
                                return [parseInt(ratioMatch[1]), parseInt(ratioMatch[2]), parseInt(ratioMatch[3])];
                            }
                            
                            // Handle percentage patterns like "79/84 (94%)" -> [79, 94] for accurate passes
                            const accurateMatch = valueStr.match(/(\d+)\/\d+\s*\((\d+)%\)/);
                            if (accurateMatch && (headerLower.includes('accurate') || headerLower.includes('passes'))) {
                                return [parseInt(accurateMatch[1]), parseInt(accurateMatch[2])];
                            }
                            
                            // Handle simple ratios like "3/3 (100%)" for tackles -> [3, 100]  
                            const simpleRatioMatch = valueStr.match(/(\d+)\/\d+\s*\((\d+)%\)/);
                            if (simpleRatioMatch && (headerLower.includes('tackles') || headerLower.includes('dribbles') || 
                                                    headerLower.includes('duels') || headerLower.includes('aerial'))) {
                                return [parseInt(simpleRatioMatch[1]), parseInt(simpleRatioMatch[2])];
                            }
                            
                            // Float values (ratings, xG, xA, Goals prevented, etc.)
                            if (headerLower.includes('rating') || headerLower.includes('xg') || headerLower.includes('xa') ||
                                headerLower.includes('prevented') || headerLower.includes('faced')) {
                                const num = parseFloat(valueStr);
                                if (!isNaN(num)) return num;
                            }
                            
                            // Special case: fields that should be integer 0 when "0" but arrays when non-zero
                            if (valueStr === '0' && (headerLower.includes('successful dribbles') || 
                                                    headerLower.includes('accurate long balls'))) {
                                return 0;
                            }
                            
                            // Integer values
                            if (headerLower.includes('minutes') || headerLower.includes('goals') || headerLower.includes('assists') ||
                                headerLower.includes('shots') || headerLower.includes('touches') || headerLower.includes('chances') ||
                                headerLower.includes('recoveries') || headerLower.includes('interceptions') || 
                                headerLower.includes('blocks') || headerLower.includes('clearances') || headerLower.includes('saves') ||
                                headerLower.includes('conceded') || headerLower.includes('cards') || headerLower.includes('fouls') ||
                                headerLower.includes('fouled') || headerLower.includes('dribbled') || headerLower.includes('duels') ||
                                headerLower.includes('contributions') || headerLower.includes('created') || headerLower.includes('final') ||
                                headerLower.includes('crosses') || headerLower.includes('throws') || headerLower.includes('offsides') ||
                                headerLower.includes('sweeper') || headerLower.includes('claim') || headerLower.includes('headed')) {
                                
                                // Extract first number from string
                                const numMatch = valueStr.match(/(\d+)/);
                                if (numMatch) {
                                    return parseInt(numMatch[1]);
                                }
                            }
                            
                            // Return original value if no parsing needed
                            return value;
                        }

                        const tables = document.querySelectorAll('table');
                        const playerData = [];
                        
                        for (const table of tables) {
                            const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent.trim());
                            const players = [];
                            
                            table.querySelectorAll('tbody tr').forEach(row => {
                                const cells = Array.from(row.querySelectorAll('td')).map(td => td.textContent.trim());
                                if (cells.length > 0 && cells[0]) {
                                    const player = {};
                                    headers.forEach((header, i) => {
                                        if (cells[i] !== undefined) {
                                            let value = cells[i];
                                            if (header && header !== 'Player') {
                                                value = parsePlayerStatValue(header, value);
                                            }
                                            player[header] = value;
                                        }
                                    });
                                    players.push(player);
                                }
                            });

                            if (players.length > 0) {
                                playerData.push({
                                    headers: headers,
                                    players: players
                                });
                            }
                        }

                        return playerData;
                    });

                    if (tableData.length > 0) {
                        allPlayerStats[tabName] = tableData;
                        console.log(`    ${tabName}: ${tableData[0]?.players?.length || 0} players extracted`);
                    }
                }
            } catch (error) {
                console.error(`    Error extracting ${tabName}:`, error.message);
            }
        }

        return allPlayerStats;
    }

    async close() {
        if (this.browser) {
            await this.browser.close();
        }
    }
}

// Test functions

// Test navigation to previous day (crawl 2 days backwards)
async function testMultiDayCrawl() {
    const seasonUrl = 'https://www.fotmob.com/en-GB/leagues/47/matches/premier-league?season=2024-2025&group=by-date';
    
    const crawler = new SeasonCrawler();
    try {
        console.log('=== SEASON CRAWLER - MULTI-DAY TEST ===');
        await crawler.init();
        
        const results = await crawler.crawlMultipleDays(seasonUrl, 2);
        console.log(`\\n✅ Multi-day test complete! Extracted ${results.totalSuccessful} matches from ${results.daysExtracted} days`);
        
    } catch (error) {
        console.error('Multi-day test error:', error);
    } finally {
        await crawler.close();
    }
}

// Test round-based crawling
async function testRoundCrawl() {
    const seasonUrl = 'https://www.fotmob.com/en-GB/leagues/47/matches/premier-league?season=2024-2025&group=by-date';
    
    const crawler = new SeasonCrawler();
    try {
        console.log('=== SEASON CRAWLER - ROUND TEST ===');
        await crawler.init();
        
        // Test crawling rounds 1-3 only
        const results = await crawler.crawlSeason(seasonUrl, 1, 3);
        console.log(`\\n✅ Round test complete! Extracted ${results.totalSuccessful} matches from ${results.roundsExtracted} rounds`);
        
    } catch (error) {
        console.error('Round test error:', error);
    } finally {
        await crawler.close();
    }
}

// Test getting available rounds
async function testGetRounds() {
    const seasonUrl = 'https://www.fotmob.com/en-GB/leagues/47/matches/premier-league?season=2024-2025&group=by-date';
    
    const crawler = new SeasonCrawler();
    try {
        console.log('=== SEASON CRAWLER - GET ROUNDS TEST ===');
        await crawler.init();
        
        await crawler.page.goto(seasonUrl, { waitUntil: 'networkidle2', timeout: 30000 });
        await crawler.humanDelay(3000, 5000);
        
        const rounds = await crawler.getAvailableRounds();
        console.log(`Available rounds: ${rounds.join(', ')}`);
        console.log(`Total rounds available: ${rounds.length}`);
        
    } catch (error) {
        console.error('Get rounds test error:', error);
    } finally {
        await crawler.close();
    }
}

// Test single day crawl (original test)
async function testSingleDay() {
    const seasonUrl = 'https://www.fotmob.com/en-GB/leagues/47/matches/premier-league?season=2024-2025&group=by-date';
    
    const crawler = new SeasonCrawler();
    try {
        console.log('=== SEASON CRAWLER - SINGLE DAY TEST ===');
        await crawler.init();
        
        const results = await crawler.crawlDay(seasonUrl);
        console.log(`\\n✅ Test complete! Extracted ${results.matches.filter(m => m.success).length} matches successfully`);
        
    } catch (error) {
        console.error('Test error:', error);
    } finally {
        await crawler.close();
    }
}

/**
 * League configuration mapping
 */
const LEAGUES = {
    'premier-league': {
        id: 47,
        name: 'Premier League',
        country: 'England',
        maxRounds: 38
    },
    'la-liga': {
        id: 87,
        name: 'La Liga',
        country: 'Spain', 
        maxRounds: 38
    },
    'bundesliga': {
        id: 54,
        name: 'Bundesliga',
        country: 'Germany',
        maxRounds: 34
    },
    'serie-a': {
        id: 55,
        name: 'Serie A',
        country: 'Italy',
        maxRounds: 38
    },
    'ligue-1': {
        id: 53,
        name: 'Ligue 1', 
        country: 'France',
        maxRounds: 34
    },
    'champions-league': {
        id: 42,
        name: 'UEFA Champions League',
        country: 'Europe',
        maxRounds: 13
    },
    'europa-league': {
        id: 73,
        name: 'UEFA Europa League', 
        country: 'Europe',
        maxRounds: 13
    }
};

/**
 * Build season URL for a given league and season
 */
function buildSeasonUrl(leagueName, season) {
    const league = LEAGUES[leagueName.toLowerCase()];
    if (!league) {
        throw new Error(`Unknown league: ${leagueName}. Available leagues: ${Object.keys(LEAGUES).join(', ')}`);
    }
    
    return `https://www.fotmob.com/en-GB/leagues/${league.id}/matches/${leagueName}?season=${season}&group=by-date`;
}

/**
 * Main season crawling function
 */
async function crawlFullSeason(leagueName, season) {
    const league = LEAGUES[leagueName.toLowerCase()];
    if (!league) {
        console.error(`❌ Unknown league: ${leagueName}`);
        console.log(`Available leagues: ${Object.keys(LEAGUES).join(', ')}`);
        process.exit(1);
    }

    const seasonUrl = buildSeasonUrl(leagueName, season);
    console.log(`=== CRAWLING FULL SEASON ===`);
    console.log(`League: ${league.name} (${league.country})`);
    console.log(`Season: ${season}`);
    console.log(`Expected rounds: ${league.maxRounds}`);
    console.log(`URL: ${seasonUrl}`);
    console.log(`=====================================\\n`);
    
    const crawler = new SeasonCrawler();
    try {
        await crawler.init();
        
        // Crawl entire season by rounds (more reliable than day-by-day)
        const results = await crawler.crawlSeason(seasonUrl, 1, league.maxRounds);
        
        console.log(`\\n🎉 SEASON CRAWL COMPLETE!`);
        console.log(`League: ${league.name}`);
        console.log(`Season: ${season}`);
        console.log(`Rounds extracted: ${results.roundsExtracted}/${league.maxRounds}`);
        console.log(`Total matches: ${results.totalMatches}`);
        console.log(`Successful extractions: ${results.totalSuccessful}/${results.totalMatches} (${(results.totalSuccessful/results.totalMatches*100).toFixed(1)}%)`);
        
        return results;
        
    } catch (error) {
        console.error('❌ Season crawl error:', error);
        process.exit(1);
    } finally {
        await crawler.close();
    }
}

/**
 * Display usage information
 */
function showUsage() {
    console.log(`
🏆 FotMob Season Crawler
Usage: node season_crawler.js <league-name> <season> [options]

Arguments:
  league-name    League identifier (required)
  season         Season in YYYY-YYYY format (required)

Available Leagues:
  premier-league    Premier League (England) - 38 rounds
  la-liga          La Liga (Spain) - 38 rounds  
  bundesliga       Bundesliga (Germany) - 34 rounds
  serie-a          Serie A (Italy) - 38 rounds
  ligue-1          Ligue 1 (France) - 34 rounds
  champions-league UEFA Champions League - 13 rounds
  europa-league    UEFA Europa League - 13 rounds

Examples:
  node season_crawler.js premier-league 2024-2025
  node season_crawler.js la-liga 2023-2024
  node season_crawler.js bundesliga 2024-2025

Test Commands:
  node season_crawler.js test single       # Test single day
  node season_crawler.js test multi-day    # Test 2 days backwards
  node season_crawler.js test rounds       # Test rounds discovery
  node season_crawler.js test season       # Test 3 rounds extraction
`);
}

// Parse command line arguments
const args = process.argv.slice(2);

if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    showUsage();
    process.exit(0);
}

// Handle test commands
if (args[0] === 'test') {
    const testType = args[1] || 'rounds';
    
    switch (testType) {
        case 'multi-day':
            testMultiDayCrawl();
            break;
        case 'rounds':
            testGetRounds();
            break;
        case 'season':
            testRoundCrawl();
            break;
        case 'single':
        default:
            testSingleDay();
            break;
    }
} else {
    // Handle league and season crawling
    if (args.length < 2) {
        console.error('❌ Error: Both league-name and season are required');
        showUsage();
        process.exit(1);
    }
    
    const leagueName = args[0];
    const season = args[1];
    
    // Validate season format (YYYY-YYYY)
    const seasonPattern = /^\d{4}-\d{4}$/;
    if (!seasonPattern.test(season)) {
        console.error(`❌ Error: Season must be in YYYY-YYYY format (e.g., 2024-2025)`);
        console.error(`Provided: ${season}`);
        process.exit(1);
    }
    
    // Run full season crawl
    crawlFullSeason(leagueName, season);
}