(** Standalone exact codec laws for OJASX Vector Quantization Runner.
    Canonical rational arithmetic avoids adding classical-real axioms. No
    floating-point, statistical quality, HF-coverage, or speed theorem is asserted. *)
Require Import Coq.Arith.PeanoNat.
Require Import QArith.Qcanon Ring Field Lia.

Open Scope Qc_scope.

Definition vq_pack_nibbles (a b : nat -> bool) (i : nat) : bool :=
  if (i <? 4)%nat then a i else b (i - 4)%nat.

Definition vq_unpack_left (packed : nat -> bool) (i : nat) : bool := packed i.
Definition vq_unpack_right (packed : nat -> bool) (i : nat) : bool := packed (4 + i)%nat.

Theorem vq_pack_left_roundtrip :
  forall a b i, (i < 4)%nat -> vq_unpack_left (vq_pack_nibbles a b) i = a i.
Proof.
  intros a b i H.
  unfold vq_unpack_left, vq_pack_nibbles.
  apply Nat.ltb_lt in H.
  rewrite H.
  reflexivity.
Qed.

Theorem vq_pack_right_roundtrip :
  forall a b i, (i < 4)%nat -> vq_unpack_right (vq_pack_nibbles a b) i = b i.
Proof.
  intros a b i H.
  unfold vq_unpack_left, vq_unpack_right, vq_pack_nibbles.
  replace ((4 + i <? 4)%nat) with false.
  2:{ symmetry; apply Nat.ltb_ge; lia. }
  replace (4 + i - 4)%nat with i by lia.
  reflexivity.
Qed.

Theorem vq_residual_reconstruction :
  forall (v q0 q1 q2 q3 r0 r1 r2 r3 r4 : Qc),
    r0 = v ->
    r1 = r0 - q0 ->
    r2 = r1 - q1 ->
    r3 = r2 - q2 ->
    r4 = r3 - q3 ->
    v = q0 + q1 + q2 + q3 + r4.
Proof.
  intros; subst; ring.
Qed.

Theorem vq_zero_residual_exact :
  forall (v q0 q1 r0 r1 r2 : Qc),
    r0 = v -> r1 = r0 - q0 -> r2 = r1 - q1 -> r2 = 0 ->
    v = q0 + q1.
Proof.
  intros v q0 q1 r0 r1 r2 H0 H1 H2 Hz.
  rewrite H0 in H1.
  rewrite H1 in H2.
  rewrite Hz in H2.
  symmetry in H2.
  replace v with (v - q0 - q1 + q0 + q1) by ring.
  rewrite H2.
  ring.
Qed.

Theorem vq_nonzero_row_scale_roundtrip :
  forall (w s n : Qc), s <> 0 -> n = w / s -> n * s = w.
Proof.
  intros; subst; field; assumption.
Qed.

Definition vq_distance2 (x y : Qc) := (x - y) * (x - y).

Theorem vq_zero_anchor_nonexpanding :
  forall (r q : Qc),
    (vq_distance2 r q <= vq_distance2 r 0)%Qc ->
    ((r - q) * (r - q) <= r * r)%Qc.
Proof.
  intros r q H.
  unfold vq_distance2 in H.
  replace (r - 0) with r in H by ring.
  exact H.
Qed.

Theorem vq_error_budget_nonnegative :
  forall (error budget : Qc), (0 <= error -> error <= budget -> 0 <= budget)%Qc.
Proof.
  intros error budget Herror Hbudget.
  eapply Qcle_trans; eauto.
Qed.

Print Assumptions vq_pack_left_roundtrip.
Print Assumptions vq_pack_right_roundtrip.
Print Assumptions vq_residual_reconstruction.
Print Assumptions vq_zero_residual_exact.
Print Assumptions vq_nonzero_row_scale_roundtrip.
Print Assumptions vq_zero_anchor_nonexpanding.
Print Assumptions vq_error_budget_nonnegative.
