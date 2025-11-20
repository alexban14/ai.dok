#!/bin/bash

# Monitor Docker container resources and processing logs
echo "=== Monitoring ai-dok-llm_interaction_service-1 ===" echo ""

while true; do
    clear
    echo "=== $(date) ==="
    echo ""
    
    # Docker stats (one-shot)
    echo "📊 CONTAINER RESOURCES:"
    docker stats ai-dok-llm_interaction_service-1 --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
    echo ""
    
    # Last processing logs
    echo "📝 RECENT PROCESSING LOGS:"
    docker logs ai-dok-llm_interaction_service-1 --tail 15 2>&1 | grep -E "Processing file|Successfully processed|Failed|CRITICAL|ERROR|TIMEOUT"
    echo ""
    
    # Worker status
    echo "🔧 WORKER STATUS:"
    docker exec ai-dok-llm_interaction_service-1 ps aux | grep -E "PID|gunicorn|uvicorn" | head -5
    echo ""
    
    echo "Press Ctrl+C to stop monitoring..."
    sleep 5
done
