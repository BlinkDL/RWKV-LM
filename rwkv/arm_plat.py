# by FL 
def is_raspberry_pi():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        
        # Check if we are on a Raspberry Pi board
        if 'Raspberry Pi' not in cpuinfo:
            return None
        
        # Parse the revision field
        for line in cpuinfo.splitlines():
            if line.startswith('Revision'):
                revision = line.split(':')[1].strip()
                return map_revision_to_version(revision)
    except FileNotFoundError:
        # Not running on Linux or /proc/cpuinfo doesn't exist
        return None


def map_revision_to_version(revision):
    # Simplified mapping for major versions based on revision codes
    # Add more revisions as needed for different Pi models
    version_map = {
        '900092': '0',        # Pi Zero
        '9000c1': '0w',       # Pi Zero W
        'a02082': '3b',       # Pi 3 Model B
        'a22082': '3b',       # Pi 3 Model B
        'a020d3': '3b+',      # Pi 3 Model B+
        'a03111': '4b',       # Pi 4 Model B
        'b03111': '4b',       # Pi 4 Model B
        'c03111': '4b',       # Pi 4 Model B
        '9020e0': '4b,8g',      # Pi 4 Model B 8GB
        'c04170': '5,4g',
        'c03140': '5',        # Pi 5
    }

    # Return the board version or None if unknown
    return version_map.get(revision, None)

##############################
    
# cf above    
def is_odroid():
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
        if "odroid" in cpuinfo or any(model in cpuinfo for model in []):
            return True
    except FileNotFoundError:
        # /proc/cpuinfo might not exist on non-Linux systems
        return False
    return False
    