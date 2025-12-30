import { exec } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { S3Client, GetObjectCommand, PutObjectCommand } from '@aws-sdk/client-s3';
import { config } from 'dotenv';

// Load environment variables
config();

// Validate required environment variables
const requiredEnvVars = [
    'CLOUDFLARE_JURISDICTION_ENDPOINT',
    'R2_ACCESS_KEY_ID',
    'R2_SECRET_ACCESS_KEY',
    'R2_BUCKET_NAME'
];

const missingVars = requiredEnvVars.filter(varName => !process.env[varName]);
if (missingVars.length > 0) {
    console.error('❌ Missing required environment variables:', missingVars.join(', '));
    console.error('Please ensure these are set in your .env file or GitHub Secrets');
    process.exit(1);
}

console.log('✅ Environment variables loaded:');
console.log(`   - CLOUDFLARE_JURISDICTION_ENDPOINT: ${process.env.CLOUDFLARE_JURISDICTION_ENDPOINT}`);
console.log(`   - R2_ACCESS_KEY_ID: ${process.env.R2_ACCESS_KEY_ID?.substring(0, 8)}...`);
console.log(`   - R2_SECRET_ACCESS_KEY: ${process.env.R2_SECRET_ACCESS_KEY ? '[SET]' : '[NOT SET]'}`);
console.log(`   - R2_BUCKET_NAME: ${process.env.R2_BUCKET_NAME}`);

// Configure Cloudflare R2 client
const r2Client = new S3Client({
    region: 'auto',
    endpoint: process.env.CLOUDFLARE_JURISDICTION_ENDPOINT,
    credentials: {
        accessKeyId: process.env.R2_ACCESS_KEY_ID,
        secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
    }
});

async function downloadFromR2(key) {
    const command = new GetObjectCommand({
        Bucket: process.env.R2_BUCKET_NAME,
        Key: key
    });

    const response = await r2Client.send(command);
    
    // Convert stream to buffer
    const chunks = [];
    for await (const chunk of response.Body) {
        chunks.push(chunk);
    }
    return Buffer.concat(chunks);
}

async function uploadToR2(key, body, contentType) {
    const command = new PutObjectCommand({
        Bucket: process.env.R2_BUCKET_NAME,
        Key: key,
        Body: body,
        ContentType: contentType
    });

    await r2Client.send(command);
}

async function downloadWeights() {
    try {
        console.log('📥 Downloading trained weights from Cloudflare R2...');
        const weightsDir = path.join('PPO_preTrained', 'UAVEnv');
        
        // Create directory if it doesn't exist
        await fs.mkdir(weightsDir, { recursive: true });

        // List of weight files to download (all three PPO types)
        const weightFiles = [
            'Vanilla_PPO_UAV_Weights.pth',
            'AR_PPO_UAV_Weights.pth',
            'NS_PPO_UAV_Weights.pth'
        ];

        let downloadCount = 0;
        let errorCount = 0;

        for (const file of weightFiles) {
            try {
                const key = `PPO_preTrained/UAVEnv/${file}`;
                console.log(`📥 Downloading ${file}...`);
                
                const fileContent = await downloadFromR2(key);
                const localPath = path.join(weightsDir, file);
                
                await fs.writeFile(localPath, fileContent);
                console.log(`✅ Downloaded ${file}`);
                downloadCount++;
            } catch (err) {
                console.error(`❌ Failed to download ${file}:`, err.message);
                errorCount++;
            }
        }

        console.log(`\n🎉 Weights download complete!`);
        console.log(`   ✅ Successful: ${downloadCount} files`);
        console.log(`   ❌ Failed: ${errorCount} files`);

        if (downloadCount === 0) {
            throw new Error('No weights were downloaded successfully. Training job may have failed or weights were not uploaded to R2.');
        }

        return { downloadCount, errorCount };
    } catch (err) {
        console.error('❌ Error downloading weights from Cloudflare R2:', err);
        throw err;
    }
}

async function executePythonScript() {
    return new Promise((resolve, reject) => {
        const pythonCommands = ['python3', 'python', 'py'];
        let currentIndex = 0;
        const startTime = Date.now();
        let lastOutputTime = Date.now();

        function tryNextCommand() {
            if (currentIndex >= pythonCommands.length) {
                reject(new Error('No Python interpreter found (tried: python3, python, py)'));
                return;
            }

            const cmd = pythonCommands[currentIndex];
            console.log(`🐍 Trying ${cmd}...`);

            const child = exec(`${cmd} -u uav_comparison_test_new.py`, {
                env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' },
                maxBuffer: 50 * 1024 * 1024 // 50MB buffer
            });

            let hasOutput = false;

            // Stream stdout in real-time
            child.stdout.on('data', (data) => {
                hasOutput = true;
                lastOutputTime = Date.now();
                process.stdout.write(data.toString());
            });

            // Stream stderr in real-time
            child.stderr.on('data', (data) => {
                const errorMsg = data.toString();
                hasOutput = true;
                lastOutputTime = Date.now();
                if (!errorMsg.includes('was not found') && !errorMsg.includes('not recognized')) {
                    process.stderr.write(errorMsg);
                }
            });

            child.on('error', (error) => {
                const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(2);
                console.error(`\n❌ Comparison test error after ${elapsed} minutes:`, error.message);
                
                if (error.message.includes('ENOENT') || error.message.includes('not found')) {
                    console.log(`   ❌ ${cmd} not available`);
                    currentIndex++;
                    tryNextCommand();
                } else {
                    reject(error);
                }
            });

            child.on('close', (code, signal) => {
                const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(2);
                
                if (code === 0) {
                    console.log(`\n✅ Comparison test completed successfully after ${elapsed} minutes`);
                    resolve({ success: true, pythonCmd: cmd });
                } else if (!hasOutput || code === 9009) {
                    // 9009 is Windows error code for "command not found"
                    console.log(`   ❌ ${cmd} not available`);
                    currentIndex++;
                    tryNextCommand();
                } else {
                    console.error(`\n❌ Comparison test terminated after ${elapsed} minutes`);
                    console.error(`   Exit code: ${code}`);
                    console.error(`   Signal: ${signal || 'none'}`);
                    
                    if (code === null) {
                        reject(new Error(`Comparison test was killed after ${elapsed} minutes. Signal: ${signal || 'none'}`));
                    } else {
                        reject(new Error(`Comparison test failed with exit code ${code} after ${elapsed} minutes`));
                    }
                }
            });

            // Log heartbeat every 5 minutes to show progress
            const heartbeat = setInterval(() => {
                const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(2);
                const timeSinceLastOutput = ((Date.now() - lastOutputTime) / 1000).toFixed(0);
                console.log(`\n💓 Heartbeat: Test running for ${elapsed} minutes (last output ${timeSinceLastOutput}s ago)`);
            }, 5 * 60 * 1000); // Every 5 minutes

            child.on('close', () => {
                clearInterval(heartbeat);
            });
        }

        tryNextCommand();
    });
}

async function processMapXMLFiles(pythonCmd) {
    return new Promise((resolve, reject) => {
        console.log('🔧 Processing map XML files...');
        console.log('   ➕ Adding boundaries...');

        const addBoundaries = exec(`${pythonCmd} -u add_boundaries.py`, {
            env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
        });

        addBoundaries.stdout.on('data', (data) => {
            process.stdout.write('   ' + data.toString());
        });

        addBoundaries.stderr.on('data', (data) => {
            process.stderr.write('   ' + data.toString());
        });

        addBoundaries.on('close', (code) => {
            if (code === 0) {
                console.log('   ➖ Removing reflectance...');

                const removeReflectance = exec(`${pythonCmd} -u remove_reflectance.py`, {
                    env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
                });

                removeReflectance.stdout.on('data', (data) => {
                    process.stdout.write('   ' + data.toString());
                });

                removeReflectance.stderr.on('data', (data) => {
                    process.stderr.write('   ' + data.toString());
                });

                removeReflectance.on('close', (code) => {
                    if (code === 0) {
                        console.log('✅ Map XML processing complete');
                        resolve({ success: true });
                    } else {
                        reject(new Error(`remove_reflectance.py exited with code ${code}`));
                    }
                });

                removeReflectance.on('error', (error) => {
                    reject(error);
                });
            } else {
                reject(new Error(`add_boundaries.py exited with code ${code}`));
            }
        });

        addBoundaries.on('error', (error) => {
            reject(error);
        });
    });
}

async function collectAgentsData() {
    try {
        const agentsDir = 'Agents';
        const data = {
            agents: {},
            summary: null
        };

        // Read results_summary.csv
        try {
            const summaryPath = path.join(agentsDir, 'results_summary.csv');
            const summaryContent = await fs.readFile(summaryPath, 'utf-8');
            data.summary = summaryContent;
        } catch (err) {
            console.log('⚠️ No summary file found');
        }

        // Read each agent's data
        const agentFolders = await fs.readdir(agentsDir);

        for (const agentName of agentFolders) {
            if (agentName === 'results_summary.csv') continue;

            const agentPath = path.join(agentsDir, agentName);
            const stat = await fs.stat(agentPath);

            if (!stat.isDirectory()) continue;

            data.agents[agentName] = {};

            // Read obstacle folders
            const obstacleFolders = await fs.readdir(agentPath);

            for (const obstacleFolder of obstacleFolders) {
                const obstaclePath = path.join(agentPath, obstacleFolder);
                const obstacleStat = await fs.stat(obstaclePath);

                if (!obstacleStat.isDirectory()) continue;

                data.agents[agentName][obstacleFolder] = {
                    maps: {}
                };

                // Read map folders (map_1, map_2, etc.)
                const mapFolders = await fs.readdir(obstaclePath);

                for (const mapFolder of mapFolders) {
                    const mapPath = path.join(obstaclePath, mapFolder);
                    const mapStat = await fs.stat(mapPath);

                    if (!mapStat.isDirectory()) continue;

                    data.agents[agentName][obstacleFolder].maps[mapFolder] = {
                        map_xml: null,
                        map_metadata: null,
                        trajectories: {}
                    };

                    // Read map.xml
                    try {
                        const mapXMLPath = path.join(mapPath, 'map.xml');
                        const mapXMLContent = await fs.readFile(mapXMLPath, 'utf-8');
                        data.agents[agentName][obstacleFolder].maps[mapFolder].map_xml = mapXMLContent;
                    } catch (err) {
                        console.log(`⚠️  No map.xml found for ${agentName}/${obstacleFolder}/${mapFolder}`);
                    }

                    // Read map_metadata.json
                    try {
                        const metadataPath = path.join(mapPath, 'map_metadata.json');
                        const metadataContent = await fs.readFile(metadataPath, 'utf-8');
                        data.agents[agentName][obstacleFolder].maps[mapFolder].map_metadata = JSON.parse(metadataContent);
                    } catch (err) {
                        console.log(`⚠️  No map_metadata.json found for ${agentName}/${obstacleFolder}/${mapFolder}`);
                    }

                    // Read trajectories
                    const trajectoriesPath = path.join(mapPath, 'trajectories');
                    try {
                        const trajectoryFiles = await fs.readdir(trajectoriesPath);

                        for (const trajFile of trajectoryFiles) {
                            if (!trajFile.endsWith('.json')) continue;

                            const trajFilePath = path.join(trajectoriesPath, trajFile);
                            const trajContent = await fs.readFile(trajFilePath, 'utf-8');
                            data.agents[agentName][obstacleFolder].maps[mapFolder].trajectories[trajFile] = JSON.parse(trajContent);
                        }
                    } catch (err) {
                        console.log(`⚠️  No trajectories found for ${agentName}/${obstacleFolder}/${mapFolder}`);
                    }
                }
            }
        }

        return data;
    } catch (err) {
        console.error('❌ Error collecting agents data:', err);
        throw err;
    }
}

async function saveAgentsData(data) {
    try {
        console.log('🔄 Uploading agents data to Cloudflare R2...');
        let uploadCount = 0;
        let errorCount = 0;

        // Upload results_summary.csv
        if (data.summary) {
            try {
                const key = 'Agents/results_summary.csv';
                await uploadToR2(key, data.summary, 'text/csv');
                console.log('✅ Uploaded results_summary.csv');
                uploadCount++;
            } catch (err) {
                console.error('❌ Failed to upload results_summary.csv:', err.message);
                errorCount++;
            }
        }

        // Upload each agent's data
        for (const [agentName, agentData] of Object.entries(data.agents)) {
            console.log(`📤 Uploading ${agentName} data...`);

            // Upload obstacle folders
            for (const [obstacleFolder, obstacleData] of Object.entries(agentData)) {
                
                // Upload map folders (map_1, map_2, etc.)
                if (obstacleData.maps && Object.keys(obstacleData.maps).length > 0) {
                    for (const [mapFolder, mapData] of Object.entries(obstacleData.maps)) {
                        const basePath = `Agents/${agentName}/${obstacleFolder}/${mapFolder}`;

                        // Upload map.xml
                        if (mapData.map_xml) {
                            try {
                                const key = `${basePath}/map.xml`;
                                await uploadToR2(key, mapData.map_xml, 'application/xml');
                                uploadCount++;
                            } catch (err) {
                                console.error(`❌ Failed to upload ${basePath}/map.xml:`, err.message);
                                errorCount++;
                            }
                        }

                        // Upload map_metadata.json
                        if (mapData.map_metadata) {
                            try {
                                const key = `${basePath}/map_metadata.json`;
                                await uploadToR2(key, JSON.stringify(mapData.map_metadata, null, 2), 'application/json');
                                uploadCount++;
                            } catch (err) {
                                console.error(`❌ Failed to upload ${basePath}/map_metadata.json:`, err.message);
                                errorCount++;
                            }
                        }

                        // Upload trajectories
                        if (mapData.trajectories && Object.keys(mapData.trajectories).length > 0) {
                            for (const [trajFile, trajData] of Object.entries(mapData.trajectories)) {
                                try {
                                    const key = `${basePath}/trajectories/${trajFile}`;
                                    await uploadToR2(key, JSON.stringify(trajData, null, 2), 'application/json');
                                    uploadCount++;
                                } catch (err) {
                                    console.error(`❌ Failed to upload ${basePath}/trajectories/${trajFile}:`, err.message);
                                    errorCount++;
                                }
                            }
                        }
                    }
                }
            }

            console.log(`✅ Uploaded ${agentName} data to Cloudflare R2`);
        }

        console.log(`\n🎉 Upload complete!`);
        console.log(`   ✅ Successful: ${uploadCount} files`);
        console.log(`   ❌ Failed: ${errorCount} files`);

        if (errorCount > 0) {
            throw new Error(`Failed to upload ${errorCount} files`);
        }

        return { uploadCount, errorCount };
    } catch (err) {
        console.error('❌ Error uploading agents data to Cloudflare R2:', err);
        throw err;
    }
}

async function main() {
    console.log('🚀 Starting UAV Testing and Data Upload Process...');
    console.log(`⏰ Timestamp: ${new Date().toISOString()}\n`);

    try {
        // Step 1: Download trained weights from R2
        console.log('📥 Step 1: Downloading trained weights...');
        const weightsDownloadResult = await downloadWeights();
        console.log(`✅ Weights downloaded: ${weightsDownloadResult.downloadCount} files\n`);

        // Step 2: Execute the UAV comparison test
        console.log('🚁 Step 2: Starting UAV comparison test...');
        const { pythonCmd } = await executePythonScript();
        console.log('✅ UAV comparison test completed!\n');

        // Step 3: Process map XML files (add boundaries and remove reflectance)
        console.log('🔧 Step 3: Processing map XML files...');
        await processMapXMLFiles(pythonCmd);

        // Step 4: Collect all generated data
        console.log('📊 Step 4: Collecting generated data...');
        const agentsData = await collectAgentsData();

        console.log(`📦 Data collected: ${Object.keys(agentsData.agents).length} agents`);

        // Step 5: Upload to Cloudflare R2
        console.log('☁️  Step 5: Uploading test results and agents data...');
        const uploadResult = await saveAgentsData(agentsData);

        console.log('\n✅ Testing and upload process completed successfully!');
        console.log(`📊 Total files uploaded: ${uploadResult.uploadCount}`);
        process.exit(0);

    } catch (error) {
        console.error(`\n❌ Process failed:`, error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

// Run the main function
main();
