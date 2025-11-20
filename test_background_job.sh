#!/bin/bash

# Test Best Practice Background Job Implementation
# This demonstrates the proper way to handle long-running tasks

echo "=== Testing Background Job Implementation ==="
echo ""

# 1. Start the job
echo "1️⃣  Starting background job..."
RESPONSE=$(curl -s -X POST "http://localhost:9322/indexing/process-bucket?max_concurrent=20&batch_size=50&client_id=1")
echo "$RESPONSE" | jq '.'

# Extract job_id
JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
echo ""
echo "✅ Job started with ID: $JOB_ID"
echo "📊 Status URL: http://localhost:9322/indexing/process-bucket/$JOB_ID"
echo ""

# 2. Monitor progress
echo "2️⃣  Monitoring progress (checking every 10 seconds)..."
echo "   Press Ctrl+C to stop monitoring (job will continue running)"
echo ""

while true; do
    sleep 10
    
    STATUS=$(curl -s "http://localhost:9322/indexing/process-bucket/$JOB_ID?client_id=1")
    
    JOB_STATUS=$(echo "$STATUS" | jq -r '.status')
    PROGRESS=$(echo "$STATUS" | jq -r '.progress.current')
    TOTAL=$(echo "$STATUS" | jq -r '.progress.total')
    PERCENTAGE=$(echo "$STATUS" | jq -r '.progress.percentage')
    CURRENT_FILE=$(echo "$STATUS" | jq -r '.progress.current_file')
    
    clear
    echo "=== Job Status: $JOB_STATUS ==="
    echo ""
    echo "Progress: $PROGRESS/$TOTAL files ($PERCENTAGE%)"
    echo "Current file: $CURRENT_FILE"
    echo ""
    
    # Show last processing logs
    echo "📝 Recent logs:"
    docker logs ai-dok-llm_interaction_service-1 --tail 5 2>&1 | grep -E "Processing|Successfully|Progress"
    echo ""
    
    # Show resources
    echo "💾 Container Resources:"
    docker stats ai-dok-llm_interaction_service-1 --no-stream --format "CPU: {{.CPUPerc}}  Memory: {{.MemUsage}}"
    echo ""
    
    if [ "$JOB_STATUS" == "completed" ]; then
        echo "✅ Job completed successfully!"
        echo ""
        echo "Final result:"
        echo "$STATUS" | jq '.result'
        break
    elif [ "$JOB_STATUS" == "failed" ]; then
        echo "❌ Job failed!"
        echo ""
        echo "Error:"
        echo "$STATUS" | jq '.error'
        break
    fi
    
    echo "Press Ctrl+C to stop monitoring (job will continue)"
done

echo ""
echo "=== Test Complete ===" echo ""
echo "Key Benefits of This Implementation:"
echo "✅ HTTP worker returns immediately (no timeout)"
echo "✅ Job runs in separate process (isolated)"
echo "✅ Progress tracking via GET endpoint"
echo "✅ Can check status anytime (survives restarts)"
echo "✅ No more WORKER TIMEOUT errors!"
