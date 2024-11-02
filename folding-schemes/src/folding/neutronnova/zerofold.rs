use crate::{arith::ccs::CCS, utils::vec::mat_vec_mul};
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::sync::Arc;

pub struct ZeroFoldFromCCS<F: PrimeField> {
    pub G: Vec<Arc<DenseMultilinearExtension<F>>>,
    pub ccs: CCS<F>,
}

impl<F: PrimeField> ZeroFoldFromCCS<F> {
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

        Self {
            G: g_mles,
            ccs: ccs.clone(),
        }
    }

    pub fn evaluate_q(&self, x: &[F]) -> F {
        let g_values: Vec<F> = self
            .G
            .iter()
            .map(|mle| mle.evaluate(x).unwrap()) // 各 MLE を z で評価
            .collect();

        let mut result = F::zero();
        for (coeff, indices) in self.ccs.c.iter().zip(self.ccs.S.iter()) {
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
    fn test_eval_zerofold_from_css() {
        let ccs = get_test_ccs::<Fr>();
        let z = get_test_z::<Fr>(3);

        let zc = ZeroFoldFromCCS::<Fr>::from_ccs(ccs.clone(), &z);

        assert_eq!(
            zc.G.len(),
            ccs.M.len(),
            "Mismatch in number of MLEs and matrices"
        );

        // Step 4: Print the result and assert its correctness
        let x_list = [
            [Fr::zero(), Fr::zero()],
            [Fr::zero(), Fr::one()],
            [Fr::one(), Fr::zero()],
            [Fr::one(), Fr::one()],
        ];
        for x in x_list.iter() {
            let q_value = zc.evaluate_q(x);
            assert_eq!(q_value, Fr::zero(), "Q(z) evaluated at z = 0 is not zero");
        }
    }
}
