#!/usr/bin/env python3
"""Build evidence probes using an existing Release/Humble CMake build's dependencies."""
import argparse
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('build', type=Path, help='Existing build/n3mapping directory')
parser.add_argument('output', type=Path)
parser.add_argument('--node-source', type=Path, help='Optional frozen Humble node source')
args = parser.parse_args()
build = args.build.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
sources = Path(__file__).resolve().parent
node_source = (args.node_source or sources.parents[2] / 'humble/src/n3mapping_node.cpp').resolve()
for source, target, extra in (
    ('replay_probe', 'test_localization_episode', []),
    ('humble_replay_probe', 'n3mapping_node', [f'-DN3M_NODE_SOURCE="{node_source}"']),
):
    generated = build / f'CMakeFiles/{target}.dir'
    flags = (generated / 'flags.make').read_text()
    includes = shlex.split(next(line.split('=', 1)[1] for line in flags.splitlines()
                               if line.startswith('CXX_INCLUDES')))
    link = shlex.split((generated / 'link.txt').read_text())
    libraries = link[link.index('-o') + 2:]
    libraries = [str(build / item) if not item.startswith(('/', '-')) else item
                 for item in libraries]
    if source == 'humble_replay_probe':
        libraries.append('-lsqlite3')
    subprocess.run([link[0], '-O3', '-DNDEBUG', '-std=c++17', '-fPIC', '-fopenmp',
                    *includes, *extra, str(sources / f'{source}.cpp'),
                    '-o', str(output / source), *libraries], check=True)
