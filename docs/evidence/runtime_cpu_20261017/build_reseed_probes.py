#!/usr/bin/env python3
"""Reproduce rejected reseeding experiments from the 114937a runtime sources.

Neither the runtime library nor the Product profile in the repository is changed.
Use LD_PRELOAD=OUTPUT/guarded_veto.so with reloc_free_space_mode=kill for core
experiments. guarded_product.so also overrides the frozen Product factory for
the DDS probe; ordinary ROS overrides do not change that factory's profile.
The alias probe is diagnostic: two sparse-copy cases fail in baseline too.
The guarded/occlusion combination fails an independent SpRobo corridor case by
about 32 m while baseline abstains. These builds reproduce that failure; the
historical CPU reduction does not qualify the combination for production.
"""
import argparse
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('build', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--occlusion-only', action='store_true',
                    help='Build only the subsequent occlusion-aware combination')
args = parser.parse_args()
build, output = args.build.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
evidence = Path(__file__).resolve().parent
repo = evidence.parents[2]

policy = (repo / 'src/relocalization_decision_policy.cpp').read_text()
bypass = '      config_.reloc_free_space_mode == "kill" || !free_space_best ||'
assert policy.count(bypass) == 1, 'This probe requires the 114937a decision policy'
policy = policy.replace(bypass, '      !free_space_best ||')
(output / 'guard_decision.cpp').write_text(policy)

factory = (repo / 'src/config.cpp').read_text().split(
    'std::string runtimeConfigCanonical', 1)[0] + '}\n'
assert factory.count('config.reloc_free_space_mode = "veto";') == 1
factory = factory.replace('#include <cmath>', '#include <cmath>\n#include <cstdio>')
factory = factory.replace('config.reloc_free_space_mode = "veto";',
    'config.reloc_free_space_mode = "kill";\n'
    '    std::fprintf(stderr, "PROBE_FREE_SPACE_MODE=kill\\n");')
(output / 'guard_product.cpp').write_text(factory)
occlusion_factory = factory.replace('config.reloc_free_space_mode = "kill";',
    'config.reloc_free_space_mode = "kill";\n'
    '    config.reloc_visibility_occlusion_aware = true;').replace(
        'PROBE_FREE_SPACE_MODE=kill',
        'PROBE_FREE_SPACE_MODE=kill PROBE_OCCLUSION_AWARE=true')
(output / 'guard_occlusion_product.cpp').write_text(occlusion_factory)
edge = (evidence / 'reseed_guard_probe.cpp').read_text().replace(
    '#include "n3mapping/world_localizing.h"',
    '#include "n3mapping/world_localizing.h"\n#include "n3mapping/relocalization_order.h"')
edge = edge.replace('  const auto* free_best =',
    '  if (ranking.top2 && compareRelocalizationScore(\n'
    '      ranking.top1_visibility, ranking.top2_visibility) >= 0) return;\n'
    '  const auto* free_best =')
(output / 'guard_edge.cpp').write_text(edge)

generated = build / 'CMakeFiles/test_localization_episode.dir'
flags = (generated / 'flags.make').read_text()
includes = shlex.split(next(line.split('=', 1)[1] for line in flags.splitlines()
                           if line.startswith('CXX_INCLUDES')))
link = shlex.split((generated / 'link.txt').read_text())
libraries = [str(build / item) if not item.startswith(('/', '-')) else item
             for item in link[link.index('-o') + 2:]]
base = [link[0], '-O3', '-DNDEBUG', '-std=c++17', '-fPIC', '-fopenmp', *includes]
guard = [str(evidence / 'reseed_guard_probe.cpp'), str(output / 'guard_decision.cpp')]
occlusion = [str(output / 'guard_edge.cpp'), str(output / 'guard_decision.cpp'),
             str(evidence / 'reseed_occlusion_probe.cpp')]
targets = [
    ('guarded_veto.so', guard, ['-shared']),
    ('guarded_product.so', guard + [str(output / 'guard_product.cpp')], ['-shared']),
    ('alias_probe', [str(evidence / 'reseed_alias_probe.cpp')], []),
    ('guard_occlusion.so', occlusion, ['-shared']),
    ('guard_occlusion_product.so',
     occlusion + [str(output / 'guard_occlusion_product.cpp')], ['-shared']),
]
for target, sources, extra in targets[3:] if args.occlusion_only else targets:
    subprocess.run([*base, *extra, *sources, '-o', str(output / target), *libraries],
                   check=True)
