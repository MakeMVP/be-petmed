#!/bin/bash
set -e

echo "Initializing LocalStack S3..."

# Create the S3 bucket for documents
awslocal s3 mb s3://petmed-documents

# Set CORS for the bucket
awslocal s3api put-bucket-cors --bucket petmed-documents --cors-configuration '{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET", "PUT", "POST", "DELETE"],
      "AllowedOrigins": ["http://localhost:3000"],
      "ExposeHeaders": ["ETag"]
    }
  ]
}'

echo "LocalStack S3 initialized successfully!"
