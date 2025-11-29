import { S3Client, ListObjectsV2Command, DeleteObjectsCommand } from '@aws-sdk/client-s3';
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

console.log('✅ Environment variables loaded');
console.log(`   - R2_BUCKET_NAME: ${process.env.R2_BUCKET_NAME}`);
console.log(`   - Endpoint: ${process.env.CLOUDFLARE_JURISDICTION_ENDPOINT}\n`);

// Configure Cloudflare R2 client
const r2Client = new S3Client({
    region: 'auto',
    endpoint: process.env.CLOUDFLARE_JURISDICTION_ENDPOINT,
    credentials: {
        accessKeyId: process.env.R2_ACCESS_KEY_ID,
        secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
    }
});

async function deleteAgentsDirectory() {
    try {
        console.log('🗑️  Starting deletion of Agents/ directory from R2...\n');
        
        let continuationToken = undefined;
        let totalDeleted = 0;
        let totalListed = 0;

        do {
            // List objects with the Agents/ prefix
            console.log('📋 Listing objects...');
            const listCommand = new ListObjectsV2Command({
                Bucket: process.env.R2_BUCKET_NAME,
                Prefix: 'Agents/',
                ContinuationToken: continuationToken,
                MaxKeys: 1000 // Maximum allowed by S3 API
            });

            const listResponse = await r2Client.send(listCommand);
            
            if (!listResponse.Contents || listResponse.Contents.length === 0) {
                console.log('✅ No more objects found with Agents/ prefix');
                break;
            }

            const objectsToDelete = listResponse.Contents.map(obj => ({ Key: obj.Key }));
            totalListed += objectsToDelete.length;
            
            console.log(`   Found ${objectsToDelete.length} objects in this batch`);
            console.log(`   Total found so far: ${totalListed} objects\n`);

            // Delete objects in batches (max 1000 per request)
            console.log('🗑️  Deleting batch...');
            const deleteCommand = new DeleteObjectsCommand({
                Bucket: process.env.R2_BUCKET_NAME,
                Delete: {
                    Objects: objectsToDelete,
                    Quiet: false
                }
            });

            const deleteResponse = await r2Client.send(deleteCommand);
            
            const deletedCount = deleteResponse.Deleted?.length || 0;
            const errorCount = deleteResponse.Errors?.length || 0;
            
            totalDeleted += deletedCount;
            
            console.log(`   ✅ Deleted: ${deletedCount} objects`);
            if (errorCount > 0) {
                console.log(`   ❌ Errors: ${errorCount} objects`);
                deleteResponse.Errors?.forEach(error => {
                    console.error(`      - ${error.Key}: ${error.Message}`);
                });
            }
            console.log(`   Total deleted: ${totalDeleted} objects\n`);

            // Check if there are more objects to list
            continuationToken = listResponse.IsTruncated ? listResponse.NextContinuationToken : undefined;

        } while (continuationToken);

        console.log('\n✅ Deletion complete!');
        console.log(`📊 Summary:`);
        console.log(`   - Total objects listed: ${totalListed}`);
        console.log(`   - Total objects deleted: ${totalDeleted}`);
        
        return { totalListed, totalDeleted };

    } catch (error) {
        console.error('\n❌ Error deleting Agents/ directory:', error);
        throw error;
    }
}

// Run the deletion
console.log('⚠️  WARNING: This will delete all objects under the Agents/ prefix in R2');
console.log('⏳ Starting in 3 seconds... (Press Ctrl+C to cancel)\n');

setTimeout(async () => {
    try {
        await deleteAgentsDirectory();
        process.exit(0);
    } catch (error) {
        console.error('Process failed:', error.message);
        process.exit(1);
    }
}, 3000);
