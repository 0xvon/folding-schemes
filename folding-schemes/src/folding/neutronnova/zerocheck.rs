use crate::{arith::ccs::CCS, utils::vec::mat_vec_mul};
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::sync::Arc;

pub struct ZeroCheckInstance<F: PrimeField> {
    pub G: Vec<Arc<DenseMultilinearExtension<F>>>,
    pub F_poly: Vec<(F, Vec<usize>)>,
    pub z: Vec<F>,
}

impl<F: PrimeField> ZeroCheckInstance<F> {
    // Computes Q(x) = F(G(z)) as a VirtualPolynomial
    // where F(m_1, m_2, ..., m_n) = \sum^q c_i * \prod_{j \in S_i} m_j
    // and G(z) = [M_1(z), M_2(z), ..., M_n(z)]
    pub fn from_ccs(ccs: CCS<F>, z: &[F]) -> Self {
        // Get the list of MLEs for G(z)
        let g_mles: Vec<Arc<DenseMultilinearExtension<F>>> = ccs
            .M
            .iter()
            .map(|matrix| {
                let evaluations = mat_vec_mul(matrix, z).unwrap(); // Perform matrix-vector multiplication
                let mle = DenseMultilinearExtension::from_evaluations_vec(ccs.s, evaluations);
                Arc::new(mle)
            })
            .collect();

        let f_poly: Vec<(F, Vec<usize>)> = ccs
            .c
            .iter()
            .zip(ccs.S.iter())
            .map(|(&coeff, indices)| (coeff, indices.clone()))
            .collect();

        Self {
            G: g_mles,
            F_poly: f_poly,
            z: z.to_vec(),
        }
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
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::arith::ccs::tests::{get_test_ccs, get_test_z};
    use ark_pallas::Fr;
    use ark_std::{One, Zero};

    #[test]
    fn test_eval_zerocheck_from_css() {
        let ccs = get_test_ccs::<Fr>();
        let z = get_test_z::<Fr>(3);

        let zc = ZeroCheckInstance::<Fr>::from_ccs(ccs.clone(), &z);

        assert_eq!(
            zc.G.len(),
            ccs.M.len(),
            "Mismatch in number of MLEs and matrices"
        );

        // Check that Q(x) evaluates to zero for all {0,1}^s combinations
        let x_list = [
            [Fr::zero(), Fr::zero()],
            [Fr::zero(), Fr::one()],
            [Fr::one(), Fr::zero()],
            [Fr::one(), Fr::one()],
        ];
        for x in x_list.iter() {
            let q_value = zc.evaluate_q(x);
            assert_eq!(
                q_value,
                Fr::zero(),
                "Q(x) evaluated at x = {:?} is not zero: {:?}",
                x,
                q_value
            );
        }
    }
}
