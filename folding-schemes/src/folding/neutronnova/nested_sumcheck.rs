use crate::folding::neutronnova::zerocheck::ZeroCheckInstance;
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::rand::Rng;
use ark_std::sync::Arc;
use ark_std::test_rng;

#[derive(Debug)]
pub struct NestedSumCheckInstance<F: PrimeField> {
    pub G: Vec<Arc<DenseMultilinearExtension<F>>>,
    pub F_poly: Vec<(F, Vec<usize>)>,
    pub z: Vec<F>,
    pub tau: F,
    pub e1: DenseMultilinearExtension<F>,
    pub e2: DenseMultilinearExtension<F>,
}

impl<F: PrimeField> From<ZeroCheckInstance<F>> for NestedSumCheckInstance<F> {
    fn from(zc: ZeroCheckInstance<F>) -> Self {
        // Step 1: Generate a random challenge tau
        // TODO: Use a proper RNG
        let mut rng = test_rng();

        let tau_int = rng.gen::<u8>() as usize;
        let tau = F::from(tau_int as u64);

        // Step 2: Compute Te as a power of tau
        let l = 1 << zc.F_poly.len();
        let m = 1 << l;

        let sqrt_m_usize = (m as f64).sqrt() as usize; // Convert to usize for loop control

        let mut e1_values = Vec::with_capacity(sqrt_m_usize);
        let mut e2_values = Vec::with_capacity(sqrt_m_usize);

        // Compute e1 as τ^0, τ^1, ..., τ^(sqrt(m)-1)
        // Compute e2 as τ^0, τ^sqrt(m), ..., τ^{(sqrt(m)-1) *sqrt(m)}
        for i in 0..sqrt_m_usize {
            let power_of_tau_e1 = tau_int.pow(i as u32);
            e1_values.push(F::from(power_of_tau_e1 as u64));

            let power_of_tau_e2 = power_of_tau_e1 * tau_int;
            e2_values.push(F::from(power_of_tau_e2 as u64));
        }

        let e1 = DenseMultilinearExtension::from_evaluations_vec(zc.F_poly.len(), e1_values);
        let e2 = DenseMultilinearExtension::from_evaluations_vec(zc.F_poly.len(), e2_values);

        Self {
            G: zc.G,
            F_poly: zc.F_poly,
            z: zc.z,
            tau,
            e1,
            e2,
        }
    }
}

impl<F: PrimeField> NestedSumCheckInstance<F> {
    pub fn compute_T(&self) -> F {
        let l = self.F_poly.len();
        // T = \sum_{x \in {0,1}^l} e(x) * q(x)
        let mut T = F::zero();
        for x in 0..(1 << l) {
            let x_bits: Vec<bool> = (0..l).map(|i| (x >> i) & 1 == 1).collect();
            let x_f: Vec<F> = x_bits
                .iter()
                .map(|&b| if b { F::one() } else { F::zero() })
                .collect();

            let q_x = self.evaluate_q(&x_f);
            T += self.e1.evaluate(&x_f).unwrap() * self.e2.evaluate(&x_f).unwrap() * q_x;
        }
        T
    }

    // Evaluate Q(x) = F(G(x)) using the precomputed G values
    pub fn evaluate_q(&self, x: &[F]) -> F {
        let g_values: Vec<F> = self
            .G
            .iter()
            .map(|mle| mle.evaluate(x).unwrap()) // Evaluate each MLE at x
            .collect();

        let mut result = F::zero();
        for (coeff, indices) in self.F_poly.iter() {
            let product = indices.iter().map(|&i| g_values[i]).product::<F>();
            result += *coeff * product;
        }
        result
    }

    pub fn fold(&self, other: &Self) -> Self {
        // Step 0: Check that instances are compatible for folding
        assert_eq!(
            self.G.len(),
            other.G.len(),
            "Cannot fold instances: G vectors have different lengths"
        );
        assert_eq!(
            self.F_poly.len(),
            other.F_poly.len(),
            "Cannot fold instances: F_poly vectors have different lengths"
        );
        assert_eq!(
            self.z.len(),
            other.z.len(),
            "Cannot fold instances: z vectors have different lengths"
        );
        assert_eq!(
            self.e1.num_vars,
            other.e1.num_vars,
            "Cannot fold instances: e1 have different number of variables"
        );
        assert_eq!(
            self.e2.num_vars,
            other.e2.num_vars,
            "Cannot fold instances: e2 have different number of variables"
        );

        // Step 1: Generate random challenges gamma and rho
        let mut rng = test_rng();
        let gamma = F::rand(&mut rng);
        // let rho = F::rand(&mut rng);

        // Step 2: Fold the G vectors
        let folded_G = self
            .G
            .iter()
            .zip(&other.G)
            .map(|(g1, g2)| {
                // Ensure the number of variables matches
                assert_eq!(
                    g1.num_vars, g2.num_vars,
                    "MLEs in G have different numbers of variables"
                );

                // Combine the evaluations
                let combined_evals = g1
                    .evaluations
                    .iter()
                    .zip(&g2.evaluations)
                    .map(|(e1, e2)| *e1 + gamma * (*e2))
                    .collect::<Vec<F>>();

                // Create a new MLE with the combined evaluations
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    g1.num_vars,
                    combined_evals,
                ))
            })
            .collect::<Vec<_>>();

        // Step 3: Fold the F_poly vectors
        let mut folded_F_poly = Vec::new();
        for ((coeff1, indices1), (coeff2, indices2)) in self.F_poly.iter().zip(&other.F_poly) {
            // Assuming that indices1 and indices2 are the same
            assert_eq!(
                indices1, indices2,
                "Cannot fold instances: F_poly indices do not match"
            );
            let new_coeff = *coeff1 + gamma * (*coeff2);
            folded_F_poly.push((new_coeff, indices1.clone()));
        }

        // Step 4: Fold the z vectors
        let folded_z = self
            .z
            .iter()
            .zip(&other.z)
            .map(|(z1, z2)| *z1 + gamma * (*z2))
            .collect::<Vec<F>>();

        // Step 5: Fold tau
        let folded_tau = self.tau + gamma * other.tau;

        // Step 6: Fold e1 and e2
        let folded_e1_evals = self
            .e1
            .evaluations
            .iter()
            .zip(&other.e1.evaluations)
            .map(|(e1_val, e2_val)| *e1_val + gamma * (*e2_val))
            .collect::<Vec<F>>();

        let folded_e2_evals = self
            .e2
            .evaluations
            .iter()
            .zip(&other.e2.evaluations)
            .map(|(e1_val, e2_val)| *e1_val + gamma * (*e2_val))
            .collect::<Vec<F>>();

        let folded_e1 =
            DenseMultilinearExtension::from_evaluations_vec(self.e1.num_vars, folded_e1_evals);
        let folded_e2 =
            DenseMultilinearExtension::from_evaluations_vec(self.e2.num_vars, folded_e2_evals);

        // Step 7: Return the new folded instance
        Self {
            G: folded_G,
            F_poly: folded_F_poly,
            z: folded_z,
            tau: folded_tau,
            e1: folded_e1,
            e2: folded_e2,
        }
    }
}

// pub struct PowerCheckInstance<F: PrimeField> {
//     pub F_poly: Vec<(F, Vec<usize>)>,
//     pub e1: DenseMultilinearExtension<F>,
//     pub e2: DenseMultilinearExtension<F>,
//     pub g2: DenseMultilinearExtension<F>,
//     pub g3: DenseMultilinearExtension<F>,
// }

// impl<F: PrimeField> From<NestedSumCheckInstance<F>> for PowerCheckInstance<F> {
//     fn from(nsc: NestedSumCheckInstance<F>) -> Self {
//         let sqrt_m = nsc.e1.evaluations.len();

//         // Compute g2 based on Construction 2
//         let mut g2_values = vec![F::one()]; // g2(0) = 1
//         g2_values.extend_from_slice(&nsc.e1.evaluations[..sqrt_m - 1]); // g2(i) = e_(i-1) for 1 <= i < sqrt(m)
//         g2_values.push(F::one()); // g2(sqrt(m)) = 1
//         g2_values.push(nsc.e1.evaluations[sqrt_m - 2]); // g2(sqrt(m) + 1) = e_(sqrt(m) - 2)
//         g2_values.push(nsc.e1.evaluations[sqrt_m - 1]); // g2(sqrt(m) + 2) = e_(sqrt(m) - 1)
//         g2_values.push(nsc.e2.evaluations[0]); // g2(sqrt(m) + 3) = e_sqrt(m+1)

//         // Convert g2 to a DenseMultilinearExtension
//         let g2 = DenseMultilinearExtension::from_evaluations_vec(sqrt_m, g2_values);

//         // Compute g3 based on Construction 2
//         let mut g3_values = vec![F::one()]; // g3(0) = 1
//         for i in 1..sqrt_m {
//             g3_values.push(nsc.tau.pow([i as u64])); // g3(i) = tau^i for 1 <= i < sqrt(m)
//         }
//         g3_values.push(F::one()); // g3(sqrt(m)) = 1
//         g3_values.push(nsc.tau); // g3(sqrt(m) + 1) = tau
//         g3_values.push(nsc.e1.evaluations[sqrt_m - 1]); // g3(sqrt(m) + 2) = e_(sqrt(m) - 1)
//         g3_values.extend(nsc.e1.evaluations.iter().skip(sqrt_m - 2).take(2).cloned()); // g3(sqrt(m) + 3 and 4) = e_(sqrt(m) - 2), e_(sqrt(m) - 1)

//         // Convert g3 to a DenseMultilinearExtension
//         let g3 = DenseMultilinearExtension::from_evaluations_vec(sqrt_m, g3_values);

//         let F_poly = vec![
//             (F::one(), vec![0]),          // Represents y1 with coefficient +1
//             (F::one().neg(), vec![1, 2]), // Represents -y2 * y3
//         ];

//         Self {
//             F_poly,
//             e1: nsc.e1,
//             e2: nsc.e2,
//             g2,
//             g3,
//         }
//     }
// }

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::arith::ccs::tests::{get_test_ccs, get_test_z};
    use crate::folding::neutronnova::zerocheck::ZeroCheckInstance;
    use ark_pallas::Fr;
    use ark_std::Zero;

    #[test]
    fn test_eval_nestedsumcheck_from_ccs() {
        let ccs = get_test_ccs::<Fr>();
        let z = get_test_z::<Fr>(3);

        let zc = ZeroCheckInstance::<Fr>::from_ccs(ccs.clone(), &z);
        let nsc: NestedSumCheckInstance<Fr> = zc.into();

        let T = nsc.compute_T();
        assert_eq!(T, Fr::zero(), "T is not zero: {:?}", T);
    }

    #[test]
    fn test_fold_nsc() {
        let ccs1 = get_test_ccs::<Fr>();
        let z1 = get_test_z::<Fr>(3);
        let zc1 = ZeroCheckInstance::<Fr>::from_ccs(ccs1.clone(), &z1);
        let nsc1: NestedSumCheckInstance<Fr> = zc1.into();

        let ccs2 = get_test_ccs::<Fr>();
        let z2 = get_test_z::<Fr>(3);
        let zc2 = ZeroCheckInstance::<Fr>::from_ccs(ccs2.clone(), &z2);
        let nsc2: NestedSumCheckInstance<Fr> = zc2.into();

        let nsc = nsc1.fold(&nsc2);
        let T = nsc.compute_T();
        assert_eq!(T, Fr::zero(), "T is not zero: {:?}", T);
    }
}
