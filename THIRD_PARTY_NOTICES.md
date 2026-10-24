# Third-party software notices

The repository's MIT licence covers project-owned code and trained policies. It does not replace the licences of the third-party or vendor files listed below. Data in the associated [Zenodo record](https://doi.org/10.5281/zenodo.23601833), including the figure source workbook, are CC BY 4.0.

| Component | Included files | Licence and source |
|---|---|---|
| LIBSVM 3.33 | `robot-controller/svm.cpp`, `robot-controller/svm.h` | BSD 3-Clause; full copyright, conditions and disclaimer in [LIBSVM COPYRIGHT.txt](LICENSES/LIBSVM%20COPYRIGHT.txt). [Upstream project](https://github.com/cjlin1/libsvm). |
| Hikvision HCNetSDK C# declarations | `operator-console/CHCNetSDK.cs` | Vendor-derived interop declarations, excluded from this repository's MIT licence. Hikvision SDK terms apply. Obtain the SDK and its applicable terms from the [Hikvision SDK download page](https://www.hikvision.com/en/support/download/sdk/). No Hikvision native SDK binaries are included. |
| Orocos KDL 1.5.1 | `simulation/native/source/vendor/kdl/` and compiled native backends | LGPL 2.1; original per-file notices retained. Full licence: [KDL_COPYING](simulation/native/source/KDL_COPYING). [Upstream](https://github.com/orocos/orocos_kinematics_dynamics). The corresponding source and build script are included so the libraries can be rebuilt. |
| Eigen 3.3.7 | `simulation/native/source/vendor/eigen/Eigen/` | Original header notices retained; MPL 2.0 text in [EIGEN COPYING.MPL2.txt](simulation/native/licenses/EIGEN%20COPYING.MPL2.txt). Per-file notices govern any exceptions. [Upstream release](https://gitlab.com/libeigen/eigen/-/tree/3.3.7). |
| MinGW-w64 winpthreads | `simulation/native/runtime/libwinpthread-1.dll` | Original copyright and licence in [WINPTHREADS COPYING.txt](simulation/native/runtime/WINPTHREADS%20COPYING.txt). |
| GCC runtime incorporated in Windows native libraries | The supplied `.dll` backends | GCC runtime notices and GCC Runtime Library Exception are retained under `simulation/native/licenses/GCC runtime/`. |

OpenCV, Python packages, hardware drivers and other separately installed dependencies retain their own licences. See [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md). Project-owned planning headers and proprietary device SDK files listed there are not included in this distribution.
