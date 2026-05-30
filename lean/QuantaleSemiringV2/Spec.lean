/-
  QuantaleSemiringV2.Spec

  This file is the Lean-side algebraic contract for the CUDA kernels in:
    ../cuda/quantale_world.cu

  Scope:
  - Prove the max-plus carrier laws used by the kernel.
  - Specify matrix join, matrix multiplication, and closure.
  - Keep CPU out of the executable state model.

  The CUDA/cLean refinement boundary should connect:
    quantale_join_assign    ↦ matrixJoin
    quantale_mul_assign     ↦ matrixMul
    quantale_closure_assign ↦ closureSpec
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
  cLean should prove that quantale_closure_assign implements this relation.
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

/-
  Kernel contract names.
  These constants are the proof boundary for cLean integration.
-/
structure CudaKernelContract where
  join_assign_refines : Prop
  mul_assign_refines : Prop
  closure_assign_refines : Prop
  step_preserves_gpu_resident_state : Prop

end QuantaleSemiringV2
