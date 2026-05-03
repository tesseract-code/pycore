#!/usr/bin/env python3
"""
Utility script to check TCP log server status.

Usage:
    python check_log_server.py                 # Find any active server
    python check_log_server.py 9020            # Check specific port
    python check_log_server.py --scan          # Scan all ports in range
    python check_log_server.py --json          # Output as JSON
"""
import json
from pycore.log.port import PortManager


def check_specific_port(port: int, output_json: bool = False) -> bool:
    """Check if server is running on a specific port."""
    is_running = PortManager.is_server_running(port)

    if output_json:
        result = {
            'port': port,
            'running': is_running,
            'status': 'active' if is_running else 'inactive'
        }
        print(json.dumps(result, indent=2))
    else:
        if is_running:
            print(f"✓ TCP Log Server is RUNNING on port {port}")
        else:
            print(f"✗ No server running on port {port}")

    return is_running


def find_active_server(output_json: bool = False) -> bool:
    """Find any active server."""
    port = PortManager.find_active_server_port()

    if output_json:
        result = {
            'found': port is not None,
            'port': port,
            'status': 'active' if port else 'not_found'
        }
        print(json.dumps(result, indent=2))
    else:
        if port:
            print(f"✓ Found active TCP Log Server on port {port}")
        else:
            print("✗ No active TCP Log Server found")

    return port is not None


def scan_all_ports(output_json: bool = False) -> list:
    """Scan all ports in the default range."""
    active_servers = []

    for port in PortManager.PORT_RANGE:
        if PortManager.is_server_running(port):
            active_servers.append(port)

    if output_json:
        result = {
            'scanned_range': {
                'start': min(PortManager.PORT_RANGE),
                'end': max(PortManager.PORT_RANGE)
            },
            'active_servers': active_servers,
            'count': len(active_servers)
        }
        print(json.dumps(result, indent=2))
    else:
        if active_servers:
            print(f"✓ Found {len(active_servers)} active server(s):")
            for port in active_servers:
                print(f"  • Port {port}")
        else:
            print(
                f"✗ No active servers found in range {min(PortManager.PORT_RANGE)}-{max(PortManager.PORT_RANGE)}")

    return active_servers


def list_lock_files(output_json: bool = False):
    """List all lock files (may include stale locks)."""
    from pathlib import Path
    import tempfile

    lock_dir = Path(tempfile.gettempdir())
    lock_files = list(lock_dir.glob("tcp_log_server_*.lock"))

    lock_info = []
    for lock_file in lock_files:
        try:
            port = int(lock_file.read_text().strip())
            is_running = PortManager.is_server_running(port)
            lock_info.append({
                'file': str(lock_file),
                'port': port,
                'running': is_running,
                'stale': not is_running
            })
        except (ValueError, IOError):
            lock_info.append({
                'file': str(lock_file),
                'port': None,
                'running': False,
                'stale': True
            })

    if output_json:
        result = {
            'lock_files': lock_info,
            'count': len(lock_info)
        }
        print(json.dumps(result, indent=2))
    else:
        if lock_info:
            print(f"Found {len(lock_info)} lock file(s):")
            for info in lock_info:
                status = "ACTIVE" if info['running'] else "STALE"
                port_str = f"port {info['port']}" if info['port'] else "invalid"
                print(f"  [{status}] {port_str}: {info['file']}")
        else:
            print("No lock files found")


def clean_stale_locks(output_json: bool = False):
    """Remove stale lock files."""
    from pathlib import Path
    import tempfile

    lock_dir = Path(tempfile.gettempdir())
    lock_files = list(lock_dir.glob("tcp_log_server_*.lock"))

    removed = []
    failed = []

    for lock_file in lock_files:
        try:
            port = int(lock_file.read_text().strip())
            if not PortManager.is_server_running(port):
                lock_file.unlink()
                removed.append({'file': str(lock_file), 'port': port})
        except Exception as e:
            failed.append({'file': str(lock_file), 'error': str(e)})

    if output_json:
        result = {
            'removed': removed,
            'failed': failed,
            'removed_count': len(removed),
            'failed_count': len(failed)
        }
        print(json.dumps(result, indent=2))
    else:
        if removed:
            print(f"✓ Removed {len(removed)} stale lock file(s):")
            for info in removed:
                print(f"  • Port {info['port']}: {info['file']}")
        else:
            print("No stale lock files to remove")

        if failed:
            print(f"\n✗ Failed to remove {len(failed)} file(s):")
            for info in failed:
                print(f"  • {info['file']}: {info['error']}")



