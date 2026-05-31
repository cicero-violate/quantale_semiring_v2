/-
  QuantaleSemiringV2.Spec

  This file is the Lean-side algebraic contract for the CUDA kernels in:
    ../cuda/quantale_world.cu

  Scope:
  - Specify the scalar compatibility semiring used by the legacy matrix kernels.
  - Specify the three tensor layers used by the canonical tensor runtime:
      confidence/correctness : max-times
      compute/time cost      : min-plus
      security/safety        : max-min
  - Specify matrix/tensor join, composition, closure, and projection boundaries.
  - Keep CPU out of the executable state model.

  The CUDA/cLean refinement boundary should connect:
    quantale_supremum_assign        ↦ matrixJoin
    quantale_tensor_assign          ↦ matrixMul
    quantale_least_fixed_point      ↦ closureSpec
    tensor_quantale_closure         ↦ tensorClosureSpec
    tensor_quantale_project         ↦ blendedProjectionSpec
    tensor_quantale_frontier_step   ↦ tensorFrontierSpec
    tensor_quantale_tick            ↦ tensorClosureSpec + tensorFrontierSpec
-/

namespace QuantaleSemiringV2

inductive Weight where
  | bot
  | finite : Nat → Weight
  | top
  deriving DecidableEq, Repr

namespace Weight

def join : Weight → Weight → Weight
  | bot, x => x
  | x, bot => x
  | top, _ => top
  | _, top => top
  | finite a, finite b => finite (Nat.max a b)

def mul : Weight → Weight → Weight
  | bot, _ => bot
  | _, bot => bot
  | top, _ => top
  | _, top => top
  | finite a, finite b => finite (a + b)

instance : Max Weight where
  max := join

instance : Mul Weight where
  mul := mul

notation "⊥q" => Weight.bot
notation "⊤q" => Weight.top
infixl:65 " ⊔q " => Weight.join
infixl:70 " ⊗q " => Weight.mul

theorem join_bot_left (a : Weight) : ⊥q ⊔q a = a := by
  rfl

theorem join_bot_right (a : Weight) : a ⊔q ⊥q = a := by
  cases a <;> rfl

theorem join_top_left (a : Weight) : ⊤q ⊔q a = ⊤q := by
  cases a <;> rfl

theorem join_top_right (a : Weight) : a ⊔q ⊤q = ⊤q := by
  cases a <;> rfl

theorem join_idem (a : Weight) : a ⊔q a = a := by
  cases a with
  | bot => rfl
  | top => rfl
  | finite n => simp [join]

theorem join_comm (a b : Weight) : a ⊔q b = b ⊔q a := by
  cases a <;> cases b <;> simp [join, Nat.max_comm]

theorem join_assoc (a b c : Weight) : (a ⊔q b) ⊔q c = a ⊔q (b ⊔q c) := by
  cases a <;> cases b <;> cases c <;> simp [join, Nat.max_assoc]

theorem mul_bot_left (a : Weight) : ⊥q ⊗q a = ⊥q := by
  rfl

theorem mul_bot_right (a : Weight) : a ⊗q ⊥q = ⊥q := by
  cases a <;> rfl

theorem mul_top_left_of_not_bot : ∀ a : Weight, a ≠ ⊥q → ⊤q ⊗q a = ⊤q
  | bot, h => False.elim (h rfl)
  | finite _, _ => rfl
  | top, _ => rfl

theorem mul_top_right_of_not_bot : ∀ a : Weight, a ≠ ⊥q → a ⊗q ⊤q = ⊤q
  | bot, h => False.elim (h rfl)
  | finite _, _ => rfl
  | top, _ => rfl

theorem mul_assoc (a b c : Weight) : (a ⊗q b) ⊗q c = a ⊗q (b ⊗q c) := by
  cases a <;> cases b <;> cases c <;> simp [mul, Nat.add_assoc]

theorem finite_zero_mul (a : Weight) : finite 0 ⊗q a = a := by
  cases a <;> simp [mul]

theorem mul_finite_zero (a : Weight) : a ⊗q finite 0 = a := by
  cases a <;> simp [mul]

end Weight

abbrev Matrix (_n : Nat) := Nat → Nat → Weight

def foldJoin : Nat → (Nat → Weight) → Weight
  | 0, _ => ⊥q
  | Nat.succ n, f => foldJoin n f ⊔q f n

def matrixJoin {n : Nat} (A B : Matrix n) : Matrix n :=
  fun i j => A i j ⊔q B i j

def matrixMul {n : Nat} (A B : Matrix n) : Matrix n :=
  fun i j => foldJoin n (fun k => A i k ⊗q B k j)

def matrixBottom {n : Nat} : Matrix n :=
  fun _ _ => ⊥q

def matrixIdentity {n : Nat} : Matrix n :=
  fun i j => if i = j then Weight.finite 0 else ⊥q

/-
  cLean should prove that quantale_least_fixed_point implements this relation.
  It is intentionally a spec relation rather than a CPU executable.
-/
def IsClosure {n : Nat} (A C : Matrix n) : Prop :=
  (∀ i j, A i j ⊔q C i j = C i j) ∧
  (∀ i, C i i = Weight.finite 0 ⊔q C i i) ∧
  (∀ i j k, (C i k ⊗q C k j) ⊔q C i j = C i j)

theorem matrix_join_pointwise {n : Nat} (A B : Matrix n) (i j : Nat) :
    matrixJoin A B i j = A i j ⊔q B i j := by
  rfl

theorem matrix_join_idem {n : Nat} (A : Matrix n) : matrixJoin A A = A := by
  funext i j
  exact Weight.join_idem (A i j)

theorem matrix_join_comm {n : Nat} (A B : Matrix n) : matrixJoin A B = matrixJoin B A := by
  funext i j
  exact Weight.join_comm (A i j) (B i j)

theorem matrix_bottom_join {n : Nat} (A : Matrix n) : matrixJoin matrixBottom A = A := by
  funext i j
  exact Weight.join_bot_left (A i j)

theorem matrix_join_bottom {n : Nat} (A : Matrix n) : matrixJoin A matrixBottom = A := by
  funext i j
  exact Weight.join_bot_right (A i j)



/-!
  Tensor quantale contract.

  The Rust/CUDA runtime stores a 3 × N × N tensor. Each layer has a distinct
  algebra. This section is intentionally a specification boundary rather than a
  CPU implementation of the CUDA runtime.
-/

inductive TensorLayer where
  | confidence
  | cost
  | safety
  deriving DecidableEq, Repr

namespace TensorLayer

def index : TensorLayer → Nat
  | confidence => 0
  | cost => 1
  | safety => 2

end TensorLayer

namespace TensorAlgebra

/-- Confidence/correctness layer: max-times in the CUDA implementation. -/
def confidenceJoin : Weight → Weight → Weight := Weight.join

def confidenceCompose : Weight → Weight → Weight := Weight.mul

/-- Cost layer join: choose the cheaper/shorter path. `top` is unreachable. -/
def costJoin : Weight → Weight → Weight
  | Weight.top, x => x
  | x, Weight.top => x
  | Weight.bot, x => x
  | x, Weight.bot => x
  | Weight.finite a, Weight.finite b => Weight.finite (Nat.min a b)

/-- Cost layer composition: accumulate cost along a path. -/
def costCompose : Weight → Weight → Weight
  | Weight.top, _ => Weight.top
  | _, Weight.top => Weight.top
  | Weight.bot, x => x
  | x, Weight.bot => x
  | Weight.finite a, Weight.finite b => Weight.finite (a + b)

/-- Safety layer join: choose the path with maximal safety. -/
def safetyJoin : Weight → Weight → Weight := Weight.join

/-- Safety layer composition: weakest-link safety. -/
def safetyCompose : Weight → Weight → Weight
  | Weight.bot, _ => Weight.bot
  | _, Weight.bot => Weight.bot
  | Weight.top, x => x
  | x, Weight.top => x
  | Weight.finite a, Weight.finite b => Weight.finite (Nat.min a b)

def layerJoin : TensorLayer → Weight → Weight → Weight
  | TensorLayer.confidence, a, b => confidenceJoin a b
  | TensorLayer.cost, a, b => costJoin a b
  | TensorLayer.safety, a, b => safetyJoin a b


def layerCompose : TensorLayer → Weight → Weight → Weight
  | TensorLayer.confidence, a, b => confidenceCompose a b
  | TensorLayer.cost, a, b => costCompose a b
  | TensorLayer.safety, a, b => safetyCompose a b


def layerBottom : TensorLayer → Weight
  | TensorLayer.confidence => Weight.bot
  | TensorLayer.cost => Weight.top
  | TensorLayer.safety => Weight.bot


def layerUnit : TensorLayer → Weight
  | TensorLayer.confidence => Weight.finite 0
  | TensorLayer.cost => Weight.bot
  | TensorLayer.safety => Weight.top

end TensorAlgebra

abbrev TensorMatrix (_n : Nat) := TensorLayer → Nat → Nat → Weight

def tensorFoldJoin : TensorLayer → Nat → (Nat → Weight) → Weight
  | layer, 0, _ => TensorAlgebra.layerBottom layer
  | layer, Nat.succ n, f =>
      TensorAlgebra.layerJoin layer (tensorFoldJoin layer n f) (f n)


def tensorMatrixMul {n : Nat} (A B : TensorMatrix n) : TensorMatrix n :=
  fun layer i j =>
    tensorFoldJoin layer n (fun k =>
      TensorAlgebra.layerCompose layer (A layer i k) (B layer k j))


def tensorMatrixJoin {n : Nat} (A B : TensorMatrix n) : TensorMatrix n :=
  fun layer i j => TensorAlgebra.layerJoin layer (A layer i j) (B layer i j)


def tensorMatrixIdentity {n : Nat} : TensorMatrix n :=
  fun layer i j => if i = j then TensorAlgebra.layerUnit layer else TensorAlgebra.layerBottom layer


def IsTensorClosure {n : Nat} (A C : TensorMatrix n) : Prop :=
  (∀ layer i j, TensorAlgebra.layerJoin layer (A layer i j) (C layer i j) = C layer i j) ∧
  (∀ layer i, C layer i i = TensorAlgebra.layerJoin layer (TensorAlgebra.layerUnit layer) (C layer i i)) ∧
  (∀ layer i j k,
    TensorAlgebra.layerJoin layer
      (TensorAlgebra.layerCompose layer (C layer i k) (C layer k j))
      (C layer i j) = C layer i j)

structure ProjectionBias where
  confidence : Weight
  cost : Weight
  safety : Weight

structure TensorDecision where
  selected_src : Nat
  selected_dst : Nat
  first_hop : Nat
  selected_score : Weight
  halted : Bool
  blocked : Bool

/--
  Boundary predicate for tensor_quantale_project.
  The concrete CUDA kernel computes a float score
  `α·confidence - β·cost + γ·safety`; Lean keeps this as an abstract predicate
  until the numeric refinement layer is attached.
-/
def BlendedProjectionSpec {n : Nat}
    (_closed : TensorMatrix n)
    (_bias : ProjectionBias)
    (_decision : TensorDecision) : Prop :=
  True

/-- Boundary predicate for tensor_quantale_frontier_step. -/
def TensorFrontierSpec {n : Nat}
    (_closed : TensorMatrix n)
    (_decision : TensorDecision) : Prop :=
  True

/-
  Kernel contract names.
  These constants are the proof boundary for cLean integration.
-/
structure CudaKernelContract where
  supremum_assign_refines : Prop
  tensor_assign_refines : Prop
  star_assign_refines : Prop
  step_preserves_gpu_resident_state : Prop
  tensor_closure_refines : Prop
  tensor_projection_refines : Prop
  tensor_frontier_refines : Prop
  tensor_tick_refines : Prop

end QuantaleSemiringV2
