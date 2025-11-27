import { exec } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

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
                    map_xml: null,
                    map_metadata: null,
                    trajectories: {}
                };

                // Read map.xml
                try {
                    const mapXmlPath = path.join(obstaclePath, 'map.xml');
                    const mapXml = await fs.readFile(mapXmlPath, 'utf-8');
                    data.agents[agentName][obstacleFolder].map_xml = mapXml;
                } catch (err) {
                    console.log(`⚠️ No map.xml for ${agentName}/${obstacleFolder}`);
                }

                // Read map_metadata.json
                try {
                    const metadataPath = path.join(obstaclePath, 'map_metadata.json');
                    const metadata = await fs.readFile(metadataPath, 'utf-8');
                    data.agents[agentName][obstacleFolder].map_metadata = JSON.parse(metadata);
                } catch (err) {
                    console.log(`⚠️ No metadata for ${agentName}/${obstacleFolder}`);
                }

                // Read trajectories
                const trajectoriesPath = path.join(obstaclePath, 'trajectories');
                try {
                    const trajectoryFiles = await fs.readdir(trajectoriesPath);

                    for (const trajFile of trajectoryFiles) {
                        const trajPath = path.join(trajectoriesPath, trajFile);
                        const trajContent = await fs.readFile(trajPath, 'utf-8');
                        data.agents[agentName][obstacleFolder].trajectories[trajFile] = JSON.parse(trajContent);
                    }
                } catch (err) {
                    console.log(`⚠️ No trajectories for ${agentName}/${obstacleFolder}`);
                }
            }
        }

        return data;
    } catch (err) {
        console.error('❌ Error collecting agents data:', err);
        throw err;
    }
}

async function runTrainingScript() {
    return new Promise((resolve, reject) => {
        const pythonCommands = ['python3', 'python', 'py'];
        let currentIndex = 0;

        function tryNextCommand() {
            if (currentIndex >= pythonCommands.length) {
                reject(new Error('No Python interpreter found (tried: python3, python, py)'));
                return;
            }

            const cmd = pythonCommands[currentIndex];
            console.log(`🐍 Trying ${cmd}...`);

            const child = exec(`${cmd} -u training.py --ppo_type ns`, {
                env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
            });

            let hasOutput = false;

            // Stream stdout in real-time
            child.stdout.on('data', (data) => {
                hasOutput = true;
                process.stdout.write(data.toString());
            });

            // Stream stderr in real-time
            child.stderr.on('data', (data) => {
                const errorMsg = data.toString();
                if (!errorMsg.includes('was not found') && !errorMsg.includes('not recognized')) {
                    process.stderr.write(errorMsg);
                }
            });

            child.on('error', (error) => {
                if (error.message.includes('ENOENT') || error.message.includes('not found')) {
                    console.log(`   ❌ ${cmd} not available`);
                    currentIndex++;
                    tryNextCommand();
                } else {
                    reject(error);
                }
            });

            child.on('close', (code) => {
                if (code === 0) {
                    resolve({ success: true, pythonCmd: cmd });
                } else if (!hasOutput) {
                    console.log(`   ❌ ${cmd} not available`);
                    currentIndex++;
                    tryNextCommand();
                } else {
                    reject(new Error(`Training script exited with code ${code}`));
                }
            });
        }

        tryNextCommand();
    });
}

async function executePythonScript() {
    return new Promise((resolve, reject) => {
        const pythonCommands = ['python3', 'python', 'py'];
        let currentIndex = 0;

        function tryNextCommand() {
            if (currentIndex >= pythonCommands.length) {
                reject(new Error('No Python interpreter found (tried: python3, python, py)'));
                return;
            }

            const cmd = pythonCommands[currentIndex];
            console.log(`🐍 Trying ${cmd}...`);

            const child = exec(`${cmd} -u uav_comparison_test_new.py`, {
                env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
            });

            let hasOutput = false;

            // Stream stdout in real-time
            child.stdout.on('data', (data) => {
                hasOutput = true;
                process.stdout.write(data.toString());
            });

            // Stream stderr in real-time
            child.stderr.on('data', (data) => {
                const errorMsg = data.toString();
                if (!errorMsg.includes('was not found') && !errorMsg.includes('not recognized')) {
                    process.stderr.write(errorMsg);
                }
            });

            child.on('error', (error) => {
                if (error.message.includes('ENOENT') || error.message.includes('not found')) {
                    console.log(`   ❌ ${cmd} not available`);
                    currentIndex++;
                    tryNextCommand();
                } else {
                    reject(error);
                }
            });

            child.on('close', (code) => {
                if (code === 0) {
                    resolve({ success: true, pythonCmd: cmd });
                } else if (!hasOutput) {
                    console.log(`   ❌ ${cmd} not available`);
                    currentIndex++;
                    tryNextCommand();
                } else {
                    reject(new Error(`Python script exited with code ${code}`));
                }
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

async function uploadToR2(key, body, contentType) {
    const command = new PutObjectCommand({
        Bucket: process.env.R2_BUCKET_NAME,
        Key: key,
        Body: body,
        ContentType: contentType
    });

    await r2Client.send(command);
}

async function uploadWeights() {
    try {
        console.log('🔄 Uploading trained weights to Cloudflare R2...');
        const weightsDir = path.join('PPO_preTrained', 'UAVEnv');
        let uploadCount = 0;
        let errorCount = 0;

        // Check if directory exists
        try {
            await fs.access(weightsDir);
        } catch (err) {
            console.error(`❌ Weights directory not found: ${weightsDir}`);
            throw new Error(`Weights directory not found: ${weightsDir}`);
        }

        // Read all files in the weights directory
        const files = await fs.readdir(weightsDir);
        
        for (const file of files) {
            const filePath = path.join(weightsDir, file);
            const stat = await fs.stat(filePath);
            
            if (stat.isFile()) {
                try {
                    const fileContent = await fs.readFile(filePath);
                    const key = `PPO_preTrained/UAVEnv/${file}`;
                    
                    // Determine content type based on file extension
                    let contentType = 'application/octet-stream';
                    if (file.endsWith('.pth')) {
                        contentType = 'application/octet-stream';
                    }
                    
                    await uploadToR2(key, fileContent, contentType);
                    console.log(`✅ Uploaded ${file}`);
                    uploadCount++;
                } catch (err) {
                    console.error(`❌ Failed to upload ${file}:`, err.message);
                    errorCount++;
                }
            }
        }

        console.log(`\n🎉 Weights upload complete!`);
        console.log(`   ✅ Successful: ${uploadCount} files`);
        console.log(`   ❌ Failed: ${errorCount} files`);

        if (errorCount > 0) {
            throw new Error(`Failed to upload ${errorCount} weight files`);
        }

        return { uploadCount, errorCount };
    } catch (err) {
        console.error('❌ Error uploading weights to Cloudflare R2:', err);
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
                const basePath = `Agents/${agentName}/${obstacleFolder}`;

                // Upload map.xml
                if (obstacleData.map_xml) {
                    try {
                        const key = `${basePath}/map.xml`;
                        await uploadToR2(key, obstacleData.map_xml, 'application/xml');
                        uploadCount++;
                    } catch (err) {
                        console.error(`❌ Failed to upload ${basePath}/map.xml:`, err.message);
                        errorCount++;
                    }
                }

                // Upload map_metadata.json
                if (obstacleData.map_metadata) {
                    try {
                        const key = `${basePath}/map_metadata.json`;
                        await uploadToR2(key, JSON.stringify(obstacleData.map_metadata, null, 2), 'application/json');
                        uploadCount++;
                    } catch (err) {
                        console.error(`❌ Failed to upload ${basePath}/map_metadata.json:`, err.message);
                        errorCount++;
                    }
                }

                // Upload trajectories
                if (obstacleData.trajectories && Object.keys(obstacleData.trajectories).length > 0) {
                    for (const [trajFile, trajData] of Object.entries(obstacleData.trajectories)) {
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
    console.log('🚀 Starting UAV Training, Data Generation and Upload Process...');
    console.log(`⏰ Timestamp: ${new Date().toISOString()}\n`);

    try {
        // Step 1: Run training script
        console.log('🎓 Step 1: Running training script with NS PPO...');
        const { pythonCmd } = await runTrainingScript();
        console.log('✅ Training completed successfully!\n');

        // Step 2: Upload trained weights
        console.log('📦 Step 2: Uploading trained weights...');
        const weightsUploadResult = await uploadWeights();
        console.log(`✅ Weights uploaded: ${weightsUploadResult.uploadCount} files\n`);

        // Step 3: Execute the UAV comparison test
        console.log('🚁 Step 3: Starting UAV comparison test...');
        await executePythonScript();
        console.log('✅ UAV comparison test completed!\n');

        // Step 4: Process map XML files (add boundaries and remove reflectance)
        console.log('🔧 Step 4: Processing map XML files...');
        await processMapXMLFiles(pythonCmd);

        // Step 5: Collect all generated data
        console.log('📊 Step 5: Collecting generated data...');
        const agentsData = await collectAgentsData();

        console.log(`📦 Data collected: ${Object.keys(agentsData.agents).length} agents`);

        // Step 6: Upload to Cloudflare R2
        console.log('☁️  Step 6: Uploading agents data...');
        const uploadResult = await saveAgentsData(agentsData);

        console.log('\n✅ Process completed successfully!');
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