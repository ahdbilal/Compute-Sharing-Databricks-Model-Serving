import os
from packaging.version import parse as parse_version

# It is recommended to maintain consistent package versions across all logged models to prevent potential conflicts.
# In cases where differences exist, the following function merges 'requirements.txt' files from each model, prioritizing the highest version number when a conflict arises.

def merge_requirements(directory):
    requirements = {}

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.startswith('requirements') and filename.endswith('.txt'):
                with open(os.path.join(root, filename), 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '==' in line:
                            name, version = line.split('==')
                            if name not in requirements or parse_version(version) > parse_version(requirements.get(name, "0")):
                                requirements[name] = version
                        else:
                            requirements[line] = requirements.get(line)

    return [f"{name}=={version}" if version else name for name, version in requirements.items()]