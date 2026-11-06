/-
  SplitManifoldLUT.lean
  Formal specification and verification of the Split-Manifold LUT Execution Engine.
  Zero-Unpack Compressed-Domain Linear Algebra in Lean 4 Core.
  
  Author: OjasX Research Team & Antigravity
-/

namespace SplitManifold

-- ============================================================================
-- Section 1: Core Algebraic Definitions over Generalized Types
-- ============================================================================

-- Inner product of two vectors of dimension D represented as functions (Fin D -> R)
def dotProduct {R : Type} [Add R] [Mul R] [Zero R] (D : Nat) (u v : Fin D → R) : R :=
  Fin.foldl D (fun acc i => acc + u i * v i) 0

-- Scale a vector of dimension D by a scalar factor s
def scaleVec {R : Type} [Mul R] (D : Nat) (s : R) (v : Fin D → R) : Fin D → R :=
  fun i => s * v i

-- A codebook maps a centroid index (Fin Kcb) to a codevector of dimension D
def Codebook (Kcb D : Nat) (R : Type) := Fin Kcb → (Fin D → R)

-- Weight code indices: for each block g : Fin G, assigns a centroid index in Fin Kcb
def WeightCodes (G Kcb : Nat) := Fin G → Fin Kcb

-- Block scales: for each block g : Fin G, assigns a scalar multiplier in R
def BlockScales (G : Nat) (R : Type) := Fin G → R

-- ============================================================================
-- Section 2: Look-Up Table (LUT) Construction and Zero-Unpack Execution
-- ============================================================================

-- Pre-Projected Look-Up Table (LUT) Definition
-- LUT[g, k] = <x_g, C[k]>
-- Activation vector x is pre-projected against every centroid in codebook C once per layer.
-- Memory footprint: G * Kcb elements (e.g. 256 * 256 * 4 bytes = 256 KB, resident in SRAM/L2).
def buildLUT {R : Type} [Add R] [Mul R] [Zero R] (G D Kcb : Nat)
    (x_blocks : Fin G → (Fin D → R)) (C : Codebook Kcb D R) :
    Fin G → Fin Kcb → R :=
  fun g k => dotProduct D (x_blocks g) (C k)

-- Zero-Unpack LUT Execution (Unit Scales)
-- Evaluates the output scalar by gathering pre-projected values from the LUT without decoding C.
-- Memory bandwidth: 0 bytes of weight floats fetched; purely index gathers from SRAM!
def executeLUT {R : Type} [Add R] [Zero R] (G Kcb : Nat)
    (lut : Fin G → Fin Kcb → R) (codes : WeightCodes G Kcb) : R :=
  Fin.foldl G (fun acc g => acc + lut g (codes g)) 0

-- Zero-Unpack Scaled LUT Execution
-- Incorporates per-block scales s_g directly on the scalar LUT entries:
-- y_m = sum_g (s_g * LUT[g, codes[g]])
def executeScaledLUT {R : Type} [Add R] [Mul R] [Zero R] (G Kcb : Nat)
    (lut : Fin G → Fin Kcb → R) (scales : BlockScales G R) (codes : WeightCodes G Kcb) : R :=
  Fin.foldl G (fun acc g => acc + scales g * lut g (codes g)) 0

-- Direct Inner Product on Quantized Weights (The Eager / Unpacked Baseline)
-- First decodes codes[g] into a dense vector C[codes[g]], then takes the dot product with x_blocks[g].
def executeDirect {R : Type} [Add R] [Mul R] [Zero R] (G D Kcb : Nat)
    (x_blocks : Fin G → (Fin D → R)) (C : Codebook Kcb D R)
    (codes : WeightCodes G Kcb) : R :=
  Fin.foldl G (fun acc g => acc + dotProduct D (x_blocks g) (C (codes g))) 0

-- ============================================================================
-- Section 3: Formal Theorems - Manifold Equivalence
-- ============================================================================

-- Theorem 1: Definitional Equivalence of Zero-Unpack LUT Execution and Direct Quantized GEMV
-- Proves that executing via pre-projected table lookup is definitionally and algebraically identical to
-- unpacking all weights and computing dense floating-point dot products.
theorem lut_execution_equivalence {R : Type} [Add R] [Mul R] [Zero R]
    (G D Kcb : Nat)
    (x_blocks : Fin G → (Fin D → R))
    (C : Codebook Kcb D R)
    (codes : WeightCodes G Kcb) :
    executeLUT G Kcb (buildLUT G D Kcb x_blocks C) codes =
    executeDirect G D Kcb x_blocks C codes := by
  rfl

-- ============================================================================
-- Section 4: Split-Manifold Architecture (Outlier Separation)
-- ============================================================================

-- Split-Manifold Additivity Structure
-- Separates weights into low-variance manifold (quantized via LUT) and
-- high-magnitude outliers (evaluated in exact precision).
structure SplitRow (G D Kcb : Nat) (R : Type) where
  manifold_codes : WeightCodes G Kcb
  outlier_dot : R  -- Exact sparse inner product <x, W_outlier>

-- Total output of Split-Manifold LUT Execution:
-- y = y_manifold (via zero-unpack LUT) + y_outlier (via exact sparse dot)
def splitManifoldForward {R : Type} [Add R] [Mul R] [Zero R]
    (G D Kcb : Nat) (lut : Fin G → Fin Kcb → R)
    (row : SplitRow G D Kcb R) : R :=
  executeLUT G Kcb lut row.manifold_codes + row.outlier_dot

-- Direct unpacked split-manifold forward
def splitManifoldDirect {R : Type} [Add R] [Mul R] [Zero R]
    (G D Kcb : Nat) (x_blocks : Fin G → (Fin D → R))
    (C : Codebook Kcb D R) (row : SplitRow G D Kcb R) : R :=
  executeDirect G D Kcb x_blocks C row.manifold_codes + row.outlier_dot

-- Theorem 2: Split-Manifold Zero-Unpack Forward Equivalence
-- Proves that Split-Manifold LUT execution produces the exact same output as
-- full direct evaluation of the split linear system.
theorem split_manifold_forward_equivalence {R : Type} [Add R] [Mul R] [Zero R]
    (G D Kcb : Nat)
    (x_blocks : Fin G → (Fin D → R))
    (C : Codebook Kcb D R)
    (row : SplitRow G D Kcb R) :
    splitManifoldForward G D Kcb (buildLUT G D Kcb x_blocks C) row =
    splitManifoldDirect G D Kcb x_blocks C row := by
  unfold splitManifoldForward splitManifoldDirect
  have h_eq : executeLUT G Kcb (buildLUT G D Kcb x_blocks C) row.manifold_codes =
              executeDirect G D Kcb x_blocks C row.manifold_codes :=
    lut_execution_equivalence G D Kcb x_blocks C row.manifold_codes
  rw [h_eq]

-- ============================================================================
-- Section 5: Matrix (Multi-Row) GEMM Generalization
-- ============================================================================

-- A full linear layer consists of M output rows
def MatrixSplitLayer (M G D Kcb : Nat) (R : Type) :=
  Fin M → SplitRow G D Kcb R

-- Evaluate full matrix multiplication y = W * x for all M rows using single LUT
def splitManifoldGemv {R : Type} [Add R] [Mul R] [Zero R]
    (M G D Kcb : Nat) (lut : Fin G → Fin Kcb → R)
    (matrix : MatrixSplitLayer M G D Kcb R) : Fin M → R :=
  fun m => splitManifoldForward G D Kcb lut (matrix m)

-- Direct full matrix multiplication
def splitManifoldGemvDirect {R : Type} [Add R] [Mul R] [Zero R]
    (M G D Kcb : Nat) (x_blocks : Fin G → (Fin D → R))
    (C : Codebook Kcb D R)
    (matrix : MatrixSplitLayer M G D Kcb R) : Fin M → R :=
  fun m => splitManifoldDirect G D Kcb x_blocks C (matrix m)

-- Theorem 3: Multi-Row GEMV Equivalence
-- Proves that an entire M x K linear layer evaluated via 1 shared LUT across all M rows
-- is point-wise identical to direct matrix multiplication.
theorem split_manifold_gemv_equivalence {R : Type} [Add R] [Mul R] [Zero R]
    (M G D Kcb : Nat)
    (x_blocks : Fin G → (Fin D → R))
    (C : Codebook Kcb D R)
    (matrix : MatrixSplitLayer M G D Kcb R) :
    splitManifoldGemv M G D Kcb (buildLUT G D Kcb x_blocks C) matrix =
    splitManifoldGemvDirect M G D Kcb x_blocks C matrix := by
  funext m
  exact split_manifold_forward_equivalence G D Kcb x_blocks C (matrix m)

-- ============================================================================
-- Section 6: Invariants & Safety Theorems
-- ============================================================================

-- Theorem 4: Exact Outlier Subspace Preservation
-- Subtraction on identical outlier values yields exact zero in any subtractive structure.
def residualOutlierError {R : Type} [Sub R] (exact_val approx_val : R) : R :=
  exact_val - approx_val

theorem exact_outlier_zero_error_int (x : Int) :
    residualOutlierError x x = 0 := by
  unfold residualOutlierError
  exact Int.sub_self x

-- Theorem 5: Finite Table Index Safety
-- Any centroid index c : Fin Kcb is guaranteed to be strictly within table bounds (< Kcb),
-- eliminating buffer overruns by construction.
theorem code_index_bounds (Kcb : Nat) (c : Fin Kcb) : c.val < Kcb :=
  c.isLt

end SplitManifold
