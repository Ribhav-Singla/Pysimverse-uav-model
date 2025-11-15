import { Worker } from 'bullmq';
import IORedis from 'ioredis';
import 'dotenv/config.js';

const connection = new IORedis(process.env.UPSTASH_REDIS_URL, {
    maxRetriesPerRequest: null,
    enableReadyCheck: false,
    enableOfflineQueue: true,
    tls: {
        rejectUnauthorized: false
    },
    retryStrategy: (times) => {
        if (times > 5) {
            console.error('❌ Failed to connect to Redis after 5 retries');
            process.exit(1);
        }
        return Math.min(times * 100, 3000);
    }
});

connection.on('error', (err) => {
    console.error('❌ Redis connection error:', err.message);
});

connection.on('connect', () => {
    console.log('✅ Connected to Redis');
});

const worker = new Worker("refreshDataQueue", async job => {
    console.log(`🔄 Processing job ID: ${job.id} with data:`, job.data)
    // Simulate data refresh processing
    await new Promise(resolve => setTimeout(resolve, 2000));
    console.log(`✅ Job ID: ${job.id} completed`);
}, { connection });

worker.on('completed', job => {
    console.log(`🎉 Job ID: ${job.id} has been completed successfully.`);
});

worker.on('failed', (job, err) => {
    console.error(`❌ Job ID: ${job.id} has failed with error: ${err.message}`);
});