#!/usr/bin/env python3
"""
Quick script to delete all VastAI instances.
WARNING: This will destroy ALL instances!
"""

import sys
from vastai import VastAI


def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║           Delete All VastAI Instances                          ║")
    print("║                                                                ║")
    print("║  WARNING: This will destroy ALL your instances!               ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

    v = VastAI()

    # Get all instances
    print("[INFO] Fetching all instances...")
    result = v.show_instances()

    # Handle different return types
    instances = []
    if isinstance(result, list):
        instances = result
    elif isinstance(result, dict) and 'instances' in result:
        instances = result['instances']
    elif isinstance(result, str) and not result.strip():
        instances = []

    if not instances:
        print("[INFO] No instances found to delete.")
        return 0

    # Show instance details
    instance_ids = [inst['id'] for inst in instances]
    print(f"[INFO] Found {len(instances)} instance(s) to delete:")
    for inst in instances:
        status = inst.get('actual_status', 'unknown')
        label = inst.get('label', 'unlabeled')
        print(f"  - Instance {inst['id']}: {status} ({label})")
    print()

    # Ask for confirmation
    if '-y' not in sys.argv and '--yes' not in sys.argv:
        response = input(f"Are you sure you want to delete ALL {len(instances)} instances? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("[INFO] Deletion cancelled.")
            return 0

    # Delete all instances
    print(f"[INFO] Destroying {len(instance_ids)} instances...")
    result = v.destroy_instances(ids=instance_ids)
    print(f"[INFO] Result: {result}")
    print("[INFO] All instances destroyed!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
