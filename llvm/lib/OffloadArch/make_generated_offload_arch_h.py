#
#  make_generated_offload_arch_h.py - Create the fle generated_offload_arch.h
#
# Copyright (c) 2024 ADVANCED MICRO DEVICES, INC.
#
# AMD is granting you permission to use this software and documentation (if any) (collectively, the
# Materials) pursuant to the terms and conditions of the Software License Agreement included with the
# Materials.  If you do not have a copy of the Software License Agreement, contact your AMD
# representative for a copy.
#
# You agree that you will not reverse engineer or decompile the Materials, in whole or in part, except for
# example code which is provided in source code form and as allowed by applicable law.
#
# WARRANTY DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND.  AMD DISCLAIMS ALL WARRANTIES, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING BUT NOT
# LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, TITLE, NON-INFRINGEMENT, THAT THE SOFTWARE WILL RUN UNINTERRUPTED OR ERROR-
# FREE OR WARRANTIES ARISING FROM CUSTOM OF TRADE OR COURSE OF USAGE.  THE ENTIRE RISK
# ASSOCIATED WITH THE USE OF THE SOFTWARE IS ASSUMED BY YOU.  Some jurisdictions do not
# allow the exclusion of implied warranties, so the above exclusion may not apply to You.
#
# LIMITATION OF LIABILITY AND INDEMNIFICATION:  AMD AND ITS LICENSORS WILL NOT,
# UNDER ANY CIRCUMSTANCES BE LIABLE TO YOU FOR ANY PUNITIVE, DIRECT, INCIDENTAL,
# INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM USE OF THE SOFTWARE OR THIS
# AGREEMENT EVEN IF AMD AND ITS LICENSORS HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGES.  In no event shall AMD's total liability to You for all damages, losses, and
# causes of action (whether in contract, tort (including negligence) or otherwise)
# exceed the amount of $100 USD.  You agree to defend, indemnify and hold harmless
# AMD and its licensors, and any of their directors, officers, employees, affiliates or
# agents from and against any and all loss, damage, liability and other expenses
# (including reasonable attorneys' fees), resulting from Your use of the Software or
# violation of the terms and conditions of this Agreement.
#
# U.S. GOVERNMENT RESTRICTED RIGHTS: The Materials are provided with "RESTRICTED RIGHTS."
# Use, duplication, or disclosure by the Government is subject to the restrictions as set
# forth in FAR 52.227-14 and DFAR252.227-7013, et seq., or its successor.  Use of the
# Materials by the Government constitutes acknowledgement of AMD's proprietary rights in them.
#
# EXPORT RESTRICTIONS: The Materials may be subject to export restrictions as stated in the
# Software License Agreement.
#

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# These are the input files
AOT_PCIID2CODENAME = ["amdgpu/pciid2codename.txt", "nvidia/pciid2codename.txt"]
AOT_CODENAME2OFFLIADARCH = ["amdgpu/codename2offloadarch.txt", "nvidia/codename2offloadarch.txt"]
AOT_CODENAME2OFFLIADARCH_AMD = ["amdgpu/codename2offloadarch.txt"]

# This is the output file which is always written to current dir
AOT_DOTH_FILE = Path.cwd() / "generated_offload_arch.h"


def write_aot_prolog(file) -> None:
    print('//  This file is generated by make_generated_offload_arch_h.py\n'
          '//  It is only included by OffloadArch.cpp\n'
          '#include <stddef.h>\n'
          '#include <stdint.h>\n', file=file, flush=True)


def write_aot_offloadarch(file, archs: Set[str]) -> None:
    print('typedef enum {', file=file)
    for arch in sorted(archs):
        print(f'  AOT_{arch.upper()},', file=file)
    print('} AOT_OFFLOADARCH;\n', file=file, flush=True)


def write_aot_codename(file, codenames: Set[str]) -> None:
    print('typedef enum {', file=file)
    for codename in sorted(codenames):
        print(f'  AOT_CN_{codename.upper()},', file=file)
    print('} AOT_CODENAME;\n', file=file, flush=True)


def write_aot_structs(file) -> None:
    print('struct AOT_CODENAME_ID_TO_STRING{\n'
          '  AOT_CODENAME codename_id;\n'
          '  const char* codename;\n'
          '};\n'
          '\n'
          'struct AOT_OFFLOADARCH_TO_STRING{\n'
          '  AOT_OFFLOADARCH offloadarch_id;\n'
          '  const char* offloadarch;\n'
          '};\n'
          '\n'
          'struct AOT_OFFLOADARCH_TO_CODENAME_ENTRY{\n'
          '  const char* offloadarch;\n'
          '  const char* codename;\n'
          '};\n'
          '\n'
          'struct AOT_TABLE_ENTRY{\n'
          '  uint16_t vendorid;\n'
          '  uint16_t devid;\n'
          '  AOT_CODENAME codename_id;\n'
          '  AOT_OFFLOADARCH offloadarch_id;\n'
          '};\n', file=file, flush=True)


def write_aot_codename_to_string(file, codenames: Set[str]) -> None:
    print('extern const AOT_CODENAME_ID_TO_STRING AOT_CODENAMES[] = {', file=file)
    for codename in sorted(codenames):
        print('  {AOT_CN_'f'{codename.upper()}, "{codename}"''},', file=file)
    print('};\n', file=file, flush=True)


def write_aot_offloadarch_to_string(file, offload_archs: Set[str]) -> None:
    print('extern const AOT_OFFLOADARCH_TO_STRING AOT_OFFLOADARCHS[] = {', file=file)
    for arch in sorted(offload_archs):
        print('  {AOT_'f'{arch.upper()}, "{arch}"''},', file=file)
    print('};\n', file=file, flush=True)


def write_amd_offloadarch_to_codename_table(file, code_names: Dict[str, List[str]]) -> None:
    print('extern const AOT_OFFLOADARCH_TO_CODENAME_ENTRY AOT_AMD_OFFLOADARCH_TO_CODENAME_TABLE[] = {', file=file)
    for code_name, offload_arch in code_names.items():
        print('    {'f'"{offload_arch[0]}", "{code_name}"''},', file=file, flush=True)
    print('};\n', file=file, flush=True)


def write_aot_table(file, offload_archs: Dict[str, List[str]], pci_ids: Dict[str, Tuple[str, str]]) -> None:
    print('extern const AOT_TABLE_ENTRY AOT_TABLE[] = {', file=file)
    for key, pci_id in pci_ids.items():
        offload_arch = offload_archs[key][0]
        for pci_vid, pci_did in pci_id:
            # Workaround, not enough information to map below correctly from txt files
            if pci_vid == '0x1002' and pci_did == '0x74a1':
                offload_arch = 'gfx941'
            print('  {'f'{pci_vid}, {pci_did}, AOT_CN_{key.upper()}, AOT_{offload_arch.upper()}''},', file=file)
    print('};\n', file=file, flush=True)


def read_codename2offloadarch(input_dir: Path, files: List[str]) -> Dict[str, List[str]]:
    result = dict()
    for file in files:
        with (input_dir / file).open() as f:
            for line in f.readlines():
                (codename, offload_arch) = line.strip().split(sep=' ', maxsplit=1)
                result.setdefault(codename, []).append(offload_arch)
    return result


def read_pciid2codename(input_path: Path, codenames: Set[str]) -> Dict[str, Tuple[str, str]]:
    result = dict()
    for file in AOT_PCIID2CODENAME:
        with (input_path / file).open() as f:
            for line in f.readlines():                
                (pci_vid, pci_did, _, _, codename, _) = \
                    line.strip() \
                        .replace(' : ', ' ') \
                        .replace(':', ' ') \
                        .split(sep=' ', maxsplit=5)
                # only proceed if we know about this codename
                if codename in codenames:
                    result.setdefault(codename, []).append((f'0x{pci_vid}', f'0x{pci_did}'))
    return result


#  ===========  Main code starts here ======================

def main(input_dir: Path):
    # we don't want to append to existing file
    AOT_DOTH_FILE.unlink(missing_ok=True)

    codename2offload_arch = read_codename2offloadarch(input_dir, AOT_CODENAME2OFFLIADARCH)

    codenames, offload_archs = set(), set()
    for [key, offload_arch] in codename2offload_arch.items():
        codenames.add(key)
        offload_archs.update(offload_arch)

    pciid2codename = read_pciid2codename(input_dir, codenames)

    codename2offload_arch_amd = read_codename2offloadarch(input_dir, AOT_CODENAME2OFFLIADARCH_AMD)

    with AOT_DOTH_FILE.open(mode="w") as f:
        write_aot_prolog(f)
        write_aot_offloadarch(f, offload_archs)
        write_aot_codename(f, codenames)
        write_aot_structs(f)
        write_aot_codename_to_string(f, codenames)
        write_aot_offloadarch_to_string(f, offload_archs)
        write_amd_offloadarch_to_codename_table(f, codename2offload_arch_amd)
        write_aot_table(f, codename2offload_arch, pciid2codename)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())

