import { exec } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
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

const PPO_TYPE = 'vanilla';

async function runTrainingScript() {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`🤖 Training ${PPO_TYPE.toUpperCase()} PPO`);
    console.log(`${'='.repeat(60)}\n`);

    const result = await new Promise((resolve, reject) => {
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

            const child = exec(`${cmd} -u training.py --ppo_type ${PPO_TYPE}`, {
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
                    console.log(`\n✅ ${PPO_TYPE.toUpperCase()} PPO training completed successfully after ${elapsed} minutes`);
                    resolve({ success: true, pythonCmd: cmd });
                } else if (!hasOutput) {
                    console.log(`   ❌ ${cmd} not available`);
                    currentIndex++;
                    tryNextCommand();
                } else {
                    console.error(`\n❌ Training script terminated after ${elapsed} minutes`);
                    console.error(`   Exit code: ${code}`);
                    console.error(`   Signal: ${signal || 'none'}`);
                    
                    if (code === null) {
                        console.error(`   Reason: Process was killed (likely timeout or resource limit)`);
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
    
    console.log(`\n${'='.repeat(60)}`);
    console.log(`✅ ${PPO_TYPE.toUpperCase()} PPO training completed`);
    console.log(`${'='.repeat(60)}\n`);
    
    return result;
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
        console.log(`🔄 Uploading ${PPO_TYPE.toUpperCase()} PPO weights to Cloudflare R2...`);
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
        
        // Filter for vanilla PPO weight files
        const vanillaFiles = files.filter(file => file.toLowerCase().includes('vanilla'));
        
        for (const file of vanillaFiles) {
            const filePath = path.join(weightsDir, file);
            const stat = await fs.stat(filePath);
            
            if (stat.isFile()) {
                try {
                    const fileContent = await fs.readFile(filePath);
                    const key = `PPO_preTrained/UAVEnv/${file}`;
                    
                    await uploadToR2(key, fileContent, 'application/octet-stream');
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

async function uploadTrainingArtifacts() {
    try {
        console.log(`🔄 Uploading ${PPO_TYPE.toUpperCase()} PPO training plots and logs to Cloudflare R2...`);
        let uploadCount = 0;
        let errorCount = 0;

        // Define training artifacts to upload for Vanilla PPO
        const artifacts = [
            { file: 'goal_achievement_VANILLA.png', type: 'image/png', desc: 'Vanilla PPO Goal Achievement' },
            { file: 'episode_rewards_vanilla.pkl', type: 'application/octet-stream', desc: 'Vanilla PPO Episode Rewards' },
            { file: 'curriculum_learning_log_vanilla.csv', type: 'text/csv', desc: 'Vanilla PPO Curriculum Log' },
        ];

        for (const artifact of artifacts) {
            try {
                await fs.access(artifact.file);
                
                const fileContent = await fs.readFile(artifact.file);
                const key = `TrainingArtifacts/${artifact.file}`;
                
                await uploadToR2(key, fileContent, artifact.type);
                console.log(`✅ Uploaded ${artifact.desc}: ${artifact.file}`);
                uploadCount++;
            } catch (err) {
                if (err.code !== 'ENOENT') {
                    console.error(`❌ Failed to upload ${artifact.file}:`, err.message);
                    errorCount++;
                } else {
                    console.log(`⚠️  ${artifact.file} not found (may not be generated yet)`);
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
    console.log(`🚀 Starting ${PPO_TYPE.toUpperCase()} PPO Training and Weight Upload Process...`);
    console.log(`⏰ Timestamp: ${new Date().toISOString()}\n`);

    try {
        // Step 1: Run training script
        console.log(`🎓 Step 1: Running ${PPO_TYPE.toUpperCase()} PPO training...`);
        await runTrainingScript();
        console.log(`✅ ${PPO_TYPE.toUpperCase()} PPO training completed successfully!\n`);

        // Step 2: Upload trained weights
        console.log('📦 Step 2: Uploading trained weights...');
        const weightsUploadResult = await uploadWeights();
        console.log(`✅ Weights uploaded: ${weightsUploadResult.uploadCount} files\n`);

        // Step 3: Upload training plots and logs
        console.log('📊 Step 3: Uploading training plots and logs...');
        const artifactsUploadResult = await uploadTrainingArtifacts();
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
