import puppeteer from 'puppeteer';

class FotMobMatchExtractor {
    constructor() {
        this.browser = null;
        this.page = null;
        this.baseUrl = 'https://www.fotmob.com';
    }

    async init() {
        console.log('Initializing FotMob match extractor...');
        
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

        console.log('FotMob match extractor initialized');
    }

    async humanDelay(min = 2000, max = 4000) {
        const delay = Math.random() * (max - min) + min;
        await new Promise(resolve => setTimeout(resolve, delay));
    }

    async navigateToMatch(matchUrl) {
        console.log(`Navigating to match: ${matchUrl}`);
        
        await this.page.goto(matchUrl, {
            waitUntil: 'networkidle2',
            timeout: 30000
        });

        await this.humanDelay(3000, 5000);
        
        // Take screenshot for debugging
        await this.page.screenshot({ path: 'match_page_debug.png' });
        console.log('Match page loaded, screenshot saved');
    }

    async analyzePageStructure() {
        console.log('Analyzing FotMob match page structure...');
        
        const analysis = await this.page.evaluate(() => {
            const structure = {
                title: document.title,
                url: window.location.href,
                headings: [],
                dataTestIds: [],
                classesWithStats: [],
                tablesFound: 0,
                possibleStatSections: []
            };

            // Get all headings
            const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
            headings.forEach(h => {
                structure.headings.push({
                    tag: h.tagName,
                    text: h.textContent.trim(),
                    classes: h.className
                });
            });

            // Get all data-testid attributes
            const testIdElements = document.querySelectorAll('[data-testid]');
            testIdElements.forEach(el => {
                structure.dataTestIds.push({
                    testId: el.getAttribute('data-testid'),
                    text: el.textContent.trim().substring(0, 50),
                    tag: el.tagName
                });
            });

            // Find classes that might contain stats
            const allElements = document.querySelectorAll('*');
            allElements.forEach(el => {
                if (el.className && typeof el.className === 'string') {
                    const text = el.textContent.trim().toLowerCase();
                    if (text.includes('possession') || 
                        text.includes('shots') || 
                        text.includes('xg') ||
                        text.includes('stats') ||
                        text.match(/\d+%/) ||
                        text.match(/\d+\s*-\s*\d+/)) {
                        
                        structure.classesWithStats.push({
                            classes: el.className,
                            tag: el.tagName,
                            text: text.substring(0, 100)
                        });
                    }
                }
            });

            // Count tables
            structure.tablesFound = document.querySelectorAll('table').length;

            // Look for sections that might contain match stats
            const sectionKeywords = ['overview', 'stats', 'statistics', 'lineups', 'events'];
            sectionKeywords.forEach(keyword => {
                const elements = Array.from(allElements).filter(el => 
                    el.textContent && el.textContent.toLowerCase().includes(keyword) && 
                    el.textContent.length < 100
                );
                if (elements.length > 0) {
                    structure.possibleStatSections.push({
                        keyword: keyword,
                        count: elements.length,
                        samples: elements.slice(0, 2).map(el => ({
                            text: el.textContent.trim().substring(0, 50),
                            classes: el.className,
                            tag: el.tagName
                        }))
                    });
                }
            });

            return structure;
        });

        console.log('\n=== PAGE STRUCTURE ANALYSIS ===');
        console.log('Title:', analysis.title);
        console.log('URL:', analysis.url);
        console.log(`Headings found: ${analysis.headings.length}`);
        console.log(`Data test IDs: ${analysis.dataTestIds.length}`);
        console.log(`Stats-related elements: ${analysis.classesWithStats.length}`);
        console.log(`Tables found: ${analysis.tablesFound}`);
        
        console.log('\nKey headings:');
        analysis.headings.slice(0, 5).forEach((h, i) => {
            console.log(`${i + 1}. <${h.tag}> "${h.text}"`);
        });

        console.log('\nData test IDs:');
        analysis.dataTestIds.slice(0, 10).forEach((item, i) => {
            console.log(`${i + 1}. ${item.testId} - "${item.text}"`);
        });

        console.log('\nStats-related elements:');
        analysis.classesWithStats.slice(0, 5).forEach((item, i) => {
            console.log(`${i + 1}. <${item.tag}> "${item.text}"`);
        });

        console.log('\nPossible stat sections:');
        analysis.possibleStatSections.forEach(section => {
            console.log(`- ${section.keyword}: ${section.count} elements`);
        });

        return analysis;
    }

    async clickStatsTab() {
        console.log('Looking for and clicking stats tab...');
        
        const statsTabClicked = await this.page.evaluate(() => {
            // Look for stats-related tabs or buttons
            const allElements = document.querySelectorAll('*');
            
            for (const element of allElements) {
                const text = element.textContent?.trim().toLowerCase();
                
                if ((text === 'stats' || text === 'statistics' || text.includes('match stats')) &&
                    (element.tagName === 'BUTTON' || 
                     element.tagName === 'A' || 
                     element.getAttribute('role') === 'tab' ||
                     element.onclick ||
                     element.style.cursor === 'pointer')) {
                    
                    element.click();
                    return true;
                }
            }
            
            return false;
        });

        if (statsTabClicked) {
            console.log('Successfully clicked stats tab');
            await this.humanDelay(2000, 4000);
            return true;
        } else {
            console.log('No stats tab found, continuing with current view');
            return false;
        }
    }

    async extractAvailableData() {
        console.log('Extracting all available data from current page...');
        
        const extractedData = await this.page.evaluate(() => {
            const data = {
                matchInfo: {},
                scores: [],
                statsFound: [],
                playerData: [],
                eventData: []
            };

            // Extract any visible scores
            const scorePattern = /(\d+)\s*[-:]\s*(\d+)/g;
            const allText = document.body.textContent;
            let scoreMatch;
            while ((scoreMatch = scorePattern.exec(allText)) !== null) {
                data.scores.push({
                    homeScore: parseInt(scoreMatch[1]),
                    awayScore: parseInt(scoreMatch[2]),
                    full: scoreMatch[0]
                });
            }

            // Extract team names from various sources
            const teamElements = document.querySelectorAll('[class*="team"], [data-testid*="team"]');
            teamElements.forEach(el => {
                const text = el.textContent.trim();
                if (text && text.length > 2 && text.length < 50 && !text.includes('vs')) {
                    data.matchInfo.possibleTeamName = text;
                }
            });

            // Extract percentage stats (like possession)
            const percentages = allText.match(/(\d+)%/g);
            if (percentages) {
                data.statsFound.push({
                    type: 'percentages',
                    values: percentages.slice(0, 10)
                });
            }

            // Look for xG or expected goals
            const xgPattern = /(\d+\.?\d*)\s*xg/gi;
            const xgMatches = [];
            let xgMatch;
            while ((xgMatch = xgPattern.exec(allText)) !== null) {
                xgMatches.push(parseFloat(xgMatch[1]));
            }
            if (xgMatches.length > 0) {
                data.statsFound.push({
                    type: 'xG',
                    values: xgMatches
                });
            }

            // Extract any visible tables
            const tables = document.querySelectorAll('table');
            tables.forEach((table, index) => {
                const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent.trim());
                const rowCount = table.querySelectorAll('tbody tr').length;
                
                if (headers.length > 0 || rowCount > 0) {
                    data.playerData.push({
                        tableIndex: index,
                        headers: headers,
                        rowCount: rowCount
                    });
                }
            });

            // Look for time-stamped events (goals, cards, etc.)
            const timePattern = /(\d{1,2})['']?\s*(⚽|🟨|🟥|goal|card)/gi;
            let timeMatch;
            while ((timeMatch = timePattern.exec(allText)) !== null) {
                data.eventData.push({
                    minute: timeMatch[1],
                    type: timeMatch[2],
                    context: allText.substring(timeMatch.index - 20, timeMatch.index + 50)
                });
            }

            return data;
        });

        console.log('\n=== EXTRACTED DATA SUMMARY ===');
        console.log(`Scores found: ${extractedData.scores.length}`);
        console.log(`Stat types found: ${extractedData.statsFound.length}`);
        console.log(`Tables found: ${extractedData.playerData.length}`);
        console.log(`Events found: ${extractedData.eventData.length}`);

        if (extractedData.scores.length > 0) {
            console.log('Sample scores:', extractedData.scores.slice(0, 3));
        }

        if (extractedData.statsFound.length > 0) {
            console.log('Stats found:');
            extractedData.statsFound.forEach(stat => {
                console.log(`- ${stat.type}: ${stat.values.slice(0, 5).join(', ')}`);
            });
        }

        if (extractedData.playerData.length > 0) {
            console.log('Player tables:');
            extractedData.playerData.forEach((table, i) => {
                console.log(`- Table ${i + 1}: ${table.rowCount} rows, headers: ${table.headers.slice(0, 3).join(', ')}`);
            });
        }

        return extractedData;
    }

    async testFullExtraction() {
        try {
            await this.init();
            
            // Test with the Liverpool vs Crystal Palace match
            const testMatchUrl = 'https://www.fotmob.com/en-GB/matches/liverpool-vs-crystal-palace/2tmp8g';
            
            await this.navigateToMatch(testMatchUrl);
            
            const pageStructure = await this.analyzePageStructure();
            
            // Try to click stats tab if available
            await this.clickStatsTab();
            
            const extractedData = await this.extractAvailableData();
            
            const fullResults = {
                matchUrl: testMatchUrl,
                extractedAt: new Date().toISOString(),
                pageStructure: pageStructure,
                extractedData: extractedData
            };

            console.log('\n=== FULL EXTRACTION TEST COMPLETE ===');
            
            // Save comprehensive results
            const fs = await import('fs');
            fs.writeFileSync('fotmob_match_analysis.json', JSON.stringify(fullResults, null, 2));
            console.log('Complete analysis saved to fotmob_match_analysis.json');
            
            return fullResults;
            
        } catch (error) {
            console.error('Full extraction test error:', error);
            throw error;
        } finally {
            await this.close();
        }
    }

    async close() {
        console.log('Closing browser...');
        if (this.browser) {
            await this.browser.close();
        }
    }
}

// Test the FotMob-specific extractor
const extractor = new FotMobMatchExtractor();
extractor.testFullExtraction().catch(console.error);