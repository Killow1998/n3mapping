"""Build tiny ELF fixtures that exercise the production identity path."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fixture compilation failed:\n"
            + completed.stdout
            + completed.stderr
        )


def build_fake_product_elf(
    root: Path,
    name: str,
    *,
    commit: str,
    product_profile_sha256: str,
    runtime_node: bool,
    verified: bool = True,
    build_type: str = "Release",
    research_tools: str = "OFF",
    delegate_script: Path | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    core_source = root / "fake_core.c"
    core_library = root / "libn3mapping_core.so"
    if not core_library.exists():
        core_source.write_text(
            "int n3mapping_fake_core(void) { return 0; }\n",
            encoding="utf-8",
        )
        _run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-Wl,-soname,libn3mapping_core.so",
                "-o",
                str(core_library),
                str(core_source),
            ]
        )

    link_name = "n3mapping_core"
    touch_symbol = "n3mapping_fake_core"
    if runtime_node:
        wrapper_source = root / "fake_wrapper.c"
        wrapper_library = root / "libn3mapping_humble_wrapper.so"
        if not wrapper_library.exists():
            wrapper_source.write_text(
                "extern int n3mapping_fake_core(void);\n"
                "int n3mapping_fake_wrapper(void) {\n"
                "  return n3mapping_fake_core();\n"
                "}\n",
                encoding="utf-8",
            )
            _run(
                [
                    "cc",
                    "-shared",
                    "-fPIC",
                    "-Wl,-soname,libn3mapping_humble_wrapper.so",
                    "-Wl,--disable-new-dtags,-rpath,$ORIGIN",
                    "-L",
                    str(root),
                    "-o",
                    str(wrapper_library),
                    str(wrapper_source),
                    "-ln3mapping_core",
                ]
            )
        link_name = "n3mapping_humble_wrapper"
        touch_symbol = "n3mapping_fake_wrapper"

    identity = json.dumps(
        {
            "schema": "n3mapping_product_build_identity_v2",
            "commit": commit,
            "product_profile_sha256": product_profile_sha256,
            "build_type": build_type,
            "research_tools": research_tools,
            "verified": verified,
        },
        sort_keys=True,
    )
    source = root / f"{name}.c"
    executable = root / name
    delegate = ""
    if delegate_script is not None:
        delegate = f"""
  char **child = calloc((size_t)argc + 3U, sizeof(char *));
  if (!child) return 70;
  child[0] = "/usr/bin/python3";
  child[1] = "-B";
  child[2] = {json.dumps(str(delegate_script.resolve()))};
  for (int index = 1; index < argc; ++index) {{
    child[index + 2] = argv[index];
  }}
  execv(child[0], child);
  return 71;
"""
    source.write_text(
        f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern int {touch_symbol}(void);

int main(int argc, char **argv) {{
  (void){touch_symbol}();
  for (int index = 1; index < argc; ++index) {{
    if (strcmp(argv[index], "--build-identity-json") == 0) {{
      puts({json.dumps(identity)});
      return 0;
    }}
  }}
{delegate}
  return 0;
}}
""",
        encoding="utf-8",
    )
    _run(
        [
            "cc",
            "-Wl,--disable-new-dtags,-rpath,$ORIGIN",
            "-L",
            str(root),
            "-o",
            str(executable),
            str(source),
            f"-l{link_name}",
        ]
    )
    return executable
