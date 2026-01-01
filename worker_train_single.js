import { exec } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { config } from 'dotenv';

// Load environment variables
config();

// Get PPO type from command line or environment variable
const PPO_TYPE = process.argv[2] || process.env.PPO_TYPE || 'ns';

// Validate PPO type
const validPPOTypes = ['vanilla', 'ar', 'ns'];
if (!validPPOTypes.includes(PPO_TYPE)) {
    console.error(`❌ Invalid PPO type: ${PPO_TYPE}`);
    console.error(`   Valid options: ${validPPOTypes.join(', ')}`);
    process.exit(1);
}

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
console.log(`   - PPO_TYPE: ${PPO_TYPE}`);

// Configure Cloudflare R2 client
const r2Client = new S3Client({
    region: 'auto',
    endpoint: process.env.CLOUDFLARE_JURISDICTION_ENDPOINT,
    credentials: {
        accessKeyId: process.env.R2_ACCESS_KEY_ID,
        secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
    }
});

async function runTrainingScript(ppoType) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`🎓 Training ${ppoType.toUpperCase()} PPO`);
    console.log(`${'='.repeat(60)}\n`);

    return new Promise((resolve, reject) => {
        const pythonCommands = ['python3', 'python', 'py'];
        let currentIndex = 0;
        const startTime = Date.now();
        let lastOutputTime = Date.now();
        let pythonCmd = null;

        function tryNextCommand() {
            if (currentIndex >= pythonCommands.length) {
                reject(new Error('No Python interpreter found (tried: python3, python, py)'));
                return;
            }

            const cmd = pythonCommands[currentIndex];
            console.log(`🐍 Trying ${cmd}...`);

            const child = exec(`${cmd} -u training.py --ppo_type ${ppoType}`, {
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
                console.error(`\n❌ Training process error after ${elapsed} minutes:`, error.message);
                
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
                    console.log(`\n✅ ${ppoType.toUpperCase()} PPO training completed successfully after ${elapsed} minutes`);
                    pythonCmd = cmd;
                    resolve({ success: true, pythonCmd: cmd, ppoType });
                } else if (!hasOutput) {
                    console.log(`   ❌ ${cmd} not available`);
                    currentIndex++;
                    tryNextCommand();
                } else {
                    // Detailed error logging
                    console.error(`\n❌ Training script terminated after ${elapsed} minutes`);
                    console.error(`   Exit code: ${code}`);
                    console.error(`   Signal: ${signal || 'none'}`);
                    
                    if (code === null) {
                        console.error(`   Reason: Process was killed (likely timeout or resource limit)`);
                        console.error(`   This typically happens due to:`);
                        console.error(`     - GitHub Actions 6-hour job timeout`);
                        console.error(`     - Out of memory (runner has ~7GB available)`);
                        console.error(`     - Manual cancellation`);
                        reject(new Error(`Training process was killed after ${elapsed} minutes. Exit code: ${code}, Signal: ${signal || 'none'}`));
                    } else {
                        console.error(`   Reason: Script exited with error code ${code}`);
                        reject(new Error(`Training script failed with exit code ${code} after ${elapsed} minutes`));
                    }
                }
            });

            // Log heartbeat every 5 minutes to show progress
            const heartbeat = setInterval(() => {
                const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(2);
                const timeSinceLastOutput = ((Date.now() - lastOutputTime) / 1000).toFixed(0);
                console.log(`\n💓 Heartbeat: Training running for ${elapsed} minutes (last output ${timeSinceLastOutput}s ago)`);
            }, 5 * 60 * 1000); // Every 5 minutes

            child.on('close', () => {
                clearInterval(heartbeat);
            });
        }

        tryNextCommand();
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

async function uploadWeights(ppoType) {
    try {
        console.log(`🔄 Uploading ${ppoType.toUpperCase()} trained weights to Cloudflare R2...`);
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

        // Map PPO type to weight filename
        const ppoTypeMap = {
            'vanilla': 'Vanilla_PPO_UAV_Weights.pth',
            'ar': 'AR_PPO_UAV_Weights.pth',
            'ns': 'NS_PPO_UAV_Weights.pth'
        };

        const weightFile = ppoTypeMap[ppoType];
        const filePath = path.join(weightsDir, weightFile);

        try {
            const fileContent = await fs.readFile(filePath);
            const key = `PPO_preTrained/UAVEnv/${weightFile}`;
            
            await uploadToR2(key, fileContent, 'application/octet-stream');
            console.log(`✅ Uploaded ${weightFile}`);
            uploadCount++;
        } catch (err) {
            console.error(`❌ Failed to upload ${weightFile}:`, err.message);
            errorCount++;
        }

        console.log(`\n🎉 Weights upload complete!`);
        console.log(`   ✅ Successful: ${uploadCount} files`);
        console.log(`   ❌ Failed: ${errorCount} files`);

        if (errorCount > 0) {
            throw new Error(`Failed to upload ${ppoType} weight files`);
        }

        return { uploadCount, errorCount };
    } catch (err) {
        console.error('❌ Error uploading weights to Cloudflare R2:', err);
        throw err;
    }
}

async function uploadTrainingArtifacts(ppoType) {
    try {
        console.log(`🔄 Uploading ${ppoType.toUpperCase()} training plots and logs to Cloudflare R2...`);
        let uploadCount = 0;
        let errorCount = 0;

        // Define training artifacts to upload based on PPO type
        const artifacts = [
            // Goal achievement plot for this PPO type
            { file: `goal_achievement_${ppoType.toUpperCase()}.png`, type: 'image/png', desc: `${ppoType.toUpperCase()} PPO Goal Achievement` },
            
            // Episode rewards pickle file
            { file: `episode_rewards_${ppoType}.pkl`, type: 'application/octet-stream', desc: `${ppoType.toUpperCase()} PPO Episode Rewards` },
            
            // Curriculum learning log
            { file: `curriculum_learning_log_${ppoType}.csv`, type: 'text/csv', desc: `${ppoType.toUpperCase()} PPO Curriculum Log` },
        ];

        // Add NS-specific artifacts
        if (ppoType === 'ns') {
            artifacts.push(
                { file: 'rdr_rules_usage_NS.png', type: 'image/png', desc: 'RDR Rules Combined Usage' },
                { file: 'r1_rule_usage_NS.png', type: 'image/png', desc: 'R1 Rule Usage' },
                { file: 'r2_rule_usage_NS.png', type: 'image/png', desc: 'R2 Rule Usage' }
            );
        }

        // Obstacle detection log (common)
        artifacts.push({ file: 'obstacle_detection_log.csv', type: 'text/csv', desc: 'Obstacle Detection Log' });

        for (const artifact of artifacts) {
            try {
                // Check if file exists
                await fs.access(artifact.file);
                
                const fileContent = await fs.readFile(artifact.file);
                const key = `TrainingArtifacts/${artifact.file}`;
                
                await uploadToR2(key, fileContent, artifact.type);
                console.log(`✅ Uploaded ${artifact.desc}: ${artifact.file}`);
                uploadCount++;
            } catch (err) {
                // Only log error if it's not a file not found error
                if (err.code !== 'ENOENT') {
                    console.error(`❌ Failed to upload ${artifact.file}:`, err.message);
                    errorCount++;
                } else {
                    console.log(`⚠️  ${artifact.file} not found (may not be generated for this PPO type)`);
                }
            }
        }

        console.log(`\n🎉 Training artifacts upload complete!`);
        console.log(`   ✅ Successful: ${uploadCount} files`);
        console.log(`   ⚠️  Skipped/Failed: ${errorCount} files`);

        return { uploadCount, errorCount };
    } catch (err) {
        console.error('❌ Error uploading training artifacts:', err);
        throw err;
    }
}

async function main() {
    console.log(`🚀 Starting UAV Training for ${PPO_TYPE.toUpperCase()} PPO...`);
    console.log(`⏰ Timestamp: ${new Date().toISOString()}\n`);

    try {
        // Step 1: Run training script for specific PPO type
        console.log(`🎓 Step 1: Running training for ${PPO_TYPE.toUpperCase()} PPO...`);
        await runTrainingScript(PPO_TYPE);
        console.log(`✅ ${PPO_TYPE.toUpperCase()} PPO training completed successfully!\n`);

        // Step 2: Upload trained weights
        console.log('📦 Step 2: Uploading trained weights...');
        const weightsUploadResult = await uploadWeights(PPO_TYPE);
        console.log(`✅ Weights uploaded: ${weightsUploadResult.uploadCount} files\n`);

        // Step 3: Upload training plots and logs
        console.log('📊 Step 3: Uploading training plots and logs...');
        const artifactsUploadResult = await uploadTrainingArtifacts(PPO_TYPE);
        console.log(`✅ Training artifacts uploaded: ${artifactsUploadResult.uploadCount} files\n`);

        console.log(`\n✅ ${PPO_TYPE.toUpperCase()} PPO training and upload process completed successfully!`);
        process.exit(0);

    } catch (error) {
        console.error(`\n❌ Process failed:`, error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

// Run the main function
main();
