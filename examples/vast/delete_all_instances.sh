#!/bin/bash
# Delete all VastAI instances
# This script will destroy ALL instances associated with your API key

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           Delete All VastAI Instances                          ║"
echo "║                                                                ║"
echo "║  WARNING: This will destroy ALL your instances!               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment if it exists
if [ -d ".new-sdk" ]; then
    source .new-sdk/bin/activate
fi

# Get all instance IDs
echo "[INFO] Fetching all instances..."
INSTANCE_IDS=$(python3 -c "
from vastai import VastAI
v = VastAI()
instances = v.show_instances()

# Handle different return types
if isinstance(instances, list):
    ids = [str(inst['id']) for inst in instances]
elif isinstance(instances, dict) and 'instances' in instances:
    ids = [str(inst['id']) for inst in instances['instances']]
else:
    ids = []

if ids:
    print(' '.join(ids))
" 2>/dev/null)

if [ -z "$INSTANCE_IDS" ]; then
    echo "[INFO] No instances found to delete."
    exit 0
fi

# Count instances
INSTANCE_COUNT=$(echo $INSTANCE_IDS | wc -w)
echo "[INFO] Found $INSTANCE_COUNT instance(s) to delete: $INSTANCE_IDS"
echo ""

# Ask for confirmation
read -p "Are you sure you want to delete ALL $INSTANCE_COUNT instances? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "[INFO] Deletion cancelled."
    exit 0
fi

# Delete all instances
echo "[INFO] Deleting instances..."
python3 -c "
from vastai import VastAI
v = VastAI()
ids = [int(id) for id in '$INSTANCE_IDS'.split()]
print(f'Destroying {len(ids)} instances: {ids}')
result = v.destroy_instances(ids=ids)
print(f'Result: {result}')
print('All instances destroyed!')
"

echo "[INFO] Done!"
