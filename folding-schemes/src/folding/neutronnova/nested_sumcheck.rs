use crate::folding::neutronnova::zerocheck::ZeroCheckInstance;
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::sync::Arc;

pub struct NestedSumCheckInstance<F: PrimeField> {
    pub G: Vec<Arc<DenseMultilinearExtension<F>>>,
    pub F_poly: Vec<(F, Vec<usize>)>,
    pub z: Vec<F>,
    pub tau: F,
    pub e: DenseMultilinearExtension<F>,
}

impl<F: PrimeField> From<ZeroCheckInstance<F>> for NestedSumCheckInstance<F> {
    fn from(zc: ZeroCheckInstance<F>) -> Self {
        // Step 1: Generate a random challenge tau
        // TODO: Use a proper RNG
        let mut rng = ark_std::test_rng();
        let tau = F::rand(&mut rng);

        // Step 2: Compute Te as a power of tau
        let mut evaluations = vec![];
        let l = zc.F_poly.len(); // Assuming l is the length of z

        for x in 0..(1 << l) {
            let mut eq_sum = F::zero();
            let x_bits: Vec<bool> = (0..l).map(|i| (x >> i) & 1 == 1).collect();

            for z in 0..(1 << l) {
                let z_bits: Vec<bool> = (0..l).map(|i| (z >> i) & 1 == 1).collect();

                // Compute eq(z, x) using the equality polynomial eq(z, x)
                let eq_z_x = x_bits
                    .iter()
                    .zip(z_bits.iter())
                    .fold(F::one(), |acc, (&xi, &zi)| {
                        let xi_f = if xi { F::one() } else { F::zero() };
                        let zi_f = if zi { F::one() } else { F::zero() };
                        acc * (xi_f * zi_f + (F::one() - xi_f) * (F::one() - zi_f))
                    });

                // Compute τ^Z, where Z is the decimal representation of z
                let z_decimal = z as u64;
                let tau_z = tau.pow([z_decimal]);

                // Accumulate the value of eq(z, x) * τ^Z
                eq_sum += eq_z_x * tau_z;
            }
            evaluations.push(eq_sum);
        }

        // Create e as a DenseMultilinearExtension based on the computed evaluations
        let e = DenseMultilinearExtension::from_evaluations_vec(l, evaluations);

        Self {
            G: zc.G,
            F_poly: zc.F_poly,
            z: zc.z,
            tau,
            e,
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
            T += self.e.evaluate(&x_f).unwrap() * q_x;
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

    // pub fn fold(&self, other: &Self) -> Self {
    //     let mut rng = ark_std::test_rng();
    //     let gamma = F::rand(&mut rng);
    // }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::arith::ccs::tests::{get_test_ccs, get_test_z};
    use crate::folding::neutronnova::zerocheck::ZeroCheckInstance;
    use ark_pallas::Fr;
    use ark_std::Zero;

    #[test]
    fn test_eval_nestedsumcheck_from_css() {
        let ccs = get_test_ccs::<Fr>();
        let z = get_test_z::<Fr>(3);

        let zc = ZeroCheckInstance::<Fr>::from_ccs(ccs.clone(), &z);
        let nsc: NestedSumCheckInstance<Fr> = zc.into();

        let T = nsc.compute_T();
        assert_eq!(T, Fr::zero(), "T is not zero: {:?}", T);
    }
}
