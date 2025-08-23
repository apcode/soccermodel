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

// Test function to crawl a single day
async function testDayCrawl() {
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

// Run the test
testDayCrawl();