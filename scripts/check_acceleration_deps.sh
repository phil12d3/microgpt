#!/usr/bin/env bash
set -euo pipefail

os_name="$(uname -s)"

echo "host_os $os_name"
echo "cpu_backend available"

if [ "$os_name" = "Darwin" ]; then
  echo "metal_backend platform_supported"

  missing=0
  for framework in Metal Foundation QuartzCore; do
    path="/System/Library/Frameworks/$framework.framework"
    if [ -d "$path" ]; then
      echo "framework $framework found $path"
    else
      echo "framework $framework missing $path" >&2
      missing=1
    fi
  done

  if command -v xcrun >/dev/null 2>&1; then
    if metal_path="$(xcrun --find metal 2>/dev/null)"; then
      echo "tool metal found $metal_path"
    else
      echo "tool metal missing" >&2
      missing=1
    fi

    if metallib_path="$(xcrun --find metallib 2>/dev/null)"; then
      echo "tool metallib found $metallib_path"
    else
      echo "tool metallib missing optional"
      echo "metal_note runtime source compilation can avoid a metallib build step for the first backend pass"
    fi
  else
    echo "tool xcrun missing" >&2
    missing=1
  fi

  echo "metal_ldlibs -framework Metal -framework Foundation -framework QuartzCore"

  if [ "$missing" -ne 0 ]; then
    echo "metal_backend unavailable"
    echo "install or repair Xcode Command Line Tools before building the Metal backend" >&2
    exit 1
  fi

  echo "metal_backend available"
else
  echo "metal_backend unavailable_non_darwin"
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "cuda_backend tool_found $(command -v nvcc)"
else
  echo "cuda_backend unavailable_nvcc_missing"
fi
