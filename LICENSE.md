# License

**Lang3D-XL: Language Embedded 3D Gaussians for Large-scale Scenes**

## Summary

This software is available for **non-commercial research and evaluation purposes only**.

Due to dependencies on code with restrictive licenses (particularly 3D Gaussian Splatting and related components from Inria and MPII), this project **cannot be used for commercial purposes** without explicit written permission from all relevant rights holders.

---

## License Terms

This repository builds upon multiple codebases with different licenses. The most restrictive terms apply to the combined work:

### 1. Core 3D Gaussian Splatting Components

Code derived from:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Inria & MPII)
- [Feature3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs) (inherits Gaussian Splatting license)
- [LangSplat](https://github.com/minghanqin/LangSplat) (inherits Gaussian Splatting license)

**License**: Gaussian-Splatting License  
**Rights Holders**: Inria and Max Planck Institut for Informatik (MPII)

**Terms**:
- ✅ **Permitted**: Non-commercial use for research and evaluation purposes
- ❌ **Prohibited**: Commercial use, exploitation, or distribution without explicit consent
- ⚠️ **Required**: Citation of relevant papers (see below)
- ⚠️ **Required**: Retention of all copyright, patent, trademark, and attribution notices

Full license text: [Gaussian-Splatting LICENSE.md](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)

### 2. SIBR Viewers

**License**: SIBR License  
**Rights Holders**: Inria and UCA

**Terms**:
- ✅ **Permitted**: Non-commercial use for research and evaluation purposes
- ❌ **Prohibited**: Commercial use without explicit consent

Full license text: [SIBR LICENSE.md](https://gitlab.inria.fr/sibr/sibr_core/-/blob/master/LICENSE.md)

### 3. Third-Party Dependencies (Permissive Licenses)

The following dependencies have permissive licenses, but their use in this project is still governed by the most restrictive terms above:

- **[CLIP](https://github.com/openai/CLIP)** (OpenAI): MIT License
- **[Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)** (Meta AI): Apache 2.0 License
- **[gsplat](https://github.com/nerfstudio-project/gsplat)**: Apache 2.0 License
- **[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)** (NVIDIA): BSD 3-Clause License

---

## Usage Restrictions

⚠️ **IMPORTANT**: This software inherits the most restrictive licenses from its dependencies.

### ✅ Permitted Uses
- Academic and scientific research
- Evaluation and testing for research purposes
- Educational purposes
- Publications in academic journals and conferences

### ❌ Prohibited Uses
- Commercial exploitation of any kind
- Integration into commercial products or services
- Distribution as part of commercial software
- Use in production systems or commercial environments
- Any use that generates revenue without explicit permission

### ⚠️ Requirements
- **Attribution**: You must retain all copyright, license, and attribution notices
- **Citation**: You must cite the relevant papers when using this software (see Citation section below)
- **License Inclusion**: When redistributing, you must include a complete copy of this license
- **Non-commercial Derivative Works**: Any derivative works must also be non-commercial and subject to the same restrictions

---

## Commercial Use

**To obtain permission for commercial use**, you must contact all relevant rights holders:

1. **For Gaussian Splatting components** (Inria):
   - Email: stip-sophia.transfert@inria.fr
   - Website: https://www.inria.fr/en

2. **For MPII components**:
   - Max Planck Institut for Informatik
   - Website: https://www.mpi-inf.mpg.de/

Unauthorized commercial use constitutes a violation of the license and may result in legal action.

## Disclaimer

THE SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTIES OF ANY NATURE, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT.

IN NO EVENT SHALL THE AUTHORS, COPYRIGHT HOLDERS, OR THEIR INSTITUTIONS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
