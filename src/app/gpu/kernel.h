#pragma once

namespace gpu
{
extern "C" void dispatchKernel(dim3 grid, dim3 block, int sbytes,
                               unsigned int *data, int arrayWidth);
}
