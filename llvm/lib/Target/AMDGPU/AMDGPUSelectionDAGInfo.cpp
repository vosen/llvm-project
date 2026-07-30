//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSelectionDAGInfo.h"
#include "AMDGPUISelLowering.h"

using namespace llvm;

AMDGPUSelectionDAGInfo::~AMDGPUSelectionDAGInfo() = default;

bool AMDGPUSelectionDAGInfo::isTargetMemoryOpcode(unsigned Opcode) const {
  return Opcode >= AMDGPUISD::FIRST_MEMORY_OPCODE &&
         Opcode <= AMDGPUISD::LAST_MEMORY_OPCODE;
}

// ZLUDA changes start
bool AMDGPUSelectionDAGInfo::isTargetStrictFPOpcode(unsigned Opcode) const {
  return Opcode == AMDGPUISD::STRICT_RCP ||
         Opcode == AMDGPUISD::STRICT_RSQ ||
         Opcode == AMDGPUISD::STRICT_SQRT ||
         Opcode == AMDGPUISD::STRICT_LOG ||
         Opcode == AMDGPUISD::STRICT_EXP ||
         Opcode == AMDGPUISD::STRICT_FP_TO_FP16;
}
// ZLUDA changes end
